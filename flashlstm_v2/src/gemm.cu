#include "gemm.h"

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace flstm {
namespace {

using CutlassHalf = cutlass::half_t;
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;

#if defined(FLASHLSTM_CUDA_ARCH) && FLASHLSTM_CUDA_ARCH >= 90
using DefaultArch = cutlass::arch::Sm90;
#elif defined(FLASHLSTM_CUDA_ARCH) && FLASHLSTM_CUDA_ARCH >= 89
using DefaultArch = cutlass::arch::Sm89;
#else
using DefaultArch = cutlass::arch::Sm80;
#endif

template <typename LayoutA, typename LayoutB, typename OpClass>
using GemmKernel = cutlass::gemm::device::Gemm<
    CutlassHalf,
    LayoutA,
    CutlassHalf,
    LayoutB,
    float,
    ColumnMajor,
    float,
    OpClass,
    DefaultArch
>;

template <
    typename LayoutA,
    typename LayoutB,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int Stages,
    int AlignmentA = 8,
    int AlignmentB = 8>
using TensorOpGemm = cutlass::gemm::device::Gemm<
    CutlassHalf,
    LayoutA,
    CutlassHalf,
    LayoutB,
    float,
    ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    DefaultArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,
    AlignmentA,
    AlignmentB,
    false,
    cutlass::arch::OpMultiplyAdd>;

template <typename LayoutA, typename LayoutB>
using SimtGemm = cutlass::gemm::device::Gemm<
    CutlassHalf,
    LayoutA,
    CutlassHalf,
    LayoutB,
    float,
    ColumnMajor,
    float,
    cutlass::arch::OpClassSimt,
    DefaultArch,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAdd>;

template <typename Gemm>
struct GemmContext {
    Gemm op;
    void *workspace{nullptr};
    size_t workspace_bytes{0};

    ~GemmContext() {
        if (workspace != nullptr) {
            cudaFree(workspace);
            workspace = nullptr;
            workspace_bytes = 0;
        }
    }

    void *GetWorkspace(size_t required) {
        if (required == 0) {
            return nullptr;
        }
        if (required > workspace_bytes) {
            if (workspace != nullptr) {
                cudaFree(workspace);
                workspace = nullptr;
                workspace_bytes = 0;
            }
            cudaError_t err = cudaMalloc(&workspace, required);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cutlass workspace alloc failed: ")
                                         + cudaGetErrorString(err));
            }
            workspace_bytes = required;
        }
        return workspace;
    }
};

template <
    typename PrimaryTensor,
    typename SecondaryTensor,
    typename SimtTensor>
void RunGemmWithFallback(
    GemmContext<PrimaryTensor> &primary_ctx,
    GemmContext<SecondaryTensor> &secondary_ctx,
    GemmContext<SimtTensor> &simt_ctx,
    int m,
    int n,
    int k,
    const __half *A,
    int lda,
    const __half *B,
    int ldb,
    float *C,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream,
    const char *what
) {
    auto attempt = [&](auto &ctx, auto const &label, std::string &errors) -> bool {
        auto &gemm = ctx.op;
        using GemmType = std::decay_t<decltype(gemm)>;
        typename GemmType::Arguments args(
            {m, n, k},
            {reinterpret_cast<CutlassHalf const *>(A), lda},
            {reinterpret_cast<CutlassHalf const *>(B), ldb},
            {C, ldc},
            {C, ldc},
            {alpha, beta}
        );
        cutlass::Status status = gemm.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            if (!errors.empty()) {
                errors += "; ";
            }
            errors += std::string(label) + " unsupported: " + cutlassGetStatusString(status);
            return false;
        }
        size_t workspace_required = std::decay_t<decltype(gemm)>::get_workspace_size(args);
        void *workspace = ctx.GetWorkspace(workspace_required);
        status = gemm.initialize(args, workspace, stream);
        if (status == cutlass::Status::kSuccess) {
            status = gemm.run(stream);
        }
        if (status == cutlass::Status::kSuccess) {
            return true;
        }
        if (!errors.empty()) {
            errors += "; ";
        }
        errors += std::string(label) + " failed: " + cutlassGetStatusString(status);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            errors += std::string(" (cuda: ") + cudaGetErrorString(err) + ")";
            cudaGetLastError();
        }
        return false;
    };

    std::string failures;
    if (attempt(primary_ctx, "tensorop", failures)) {
        return;
    }
    if (attempt(secondary_ctx, "tensorop_small", failures)) {
        return;
    }
    if (attempt(simt_ctx, "simt", failures)) {
        return;
    }
    throw std::runtime_error(std::string(what) + ": " + failures);
}

} // namespace

void GemmTN(
    int m,
    int n,
    int k,
    const __half *A,
    int lda,
    const __half *B,
    int ldb,
    float *C,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    using Primary = TensorOpGemm<
        RowMajor,
        ColumnMajor,
        cutlass::gemm::GemmShape<128, 64, 64>,
        cutlass::gemm::GemmShape<64, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        3>;
    using Secondary = TensorOpGemm<
        RowMajor,
        ColumnMajor,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<32, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        2>;
    using Simt = SimtGemm<RowMajor, ColumnMajor>;
    static GemmContext<Primary> primary_ctx;
    static GemmContext<Secondary> secondary_ctx;
    static GemmContext<Simt> simt_ctx;
    RunGemmWithFallback(
        primary_ctx,
        secondary_ctx,
        simt_ctx,
        m,
        n,
        k,
        A,
        lda,
        B,
        ldb,
        C,
        ldc,
        alpha,
        beta,
        stream,
        "cutlass GemmTN");
}

void GemmNN(
    int m,
    int n,
    int k,
    const __half *A,
    int lda,
    const __half *B,
    int ldb,
    float *C,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    using Primary = TensorOpGemm<
        ColumnMajor,
        ColumnMajor,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        4>;
    using Secondary = TensorOpGemm<
        ColumnMajor,
        ColumnMajor,
        cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        3>;
    using Simt = SimtGemm<ColumnMajor, ColumnMajor>;
    static GemmContext<Primary> primary_ctx;
    static GemmContext<Secondary> secondary_ctx;
    static GemmContext<Simt> simt_ctx;
    RunGemmWithFallback(
        primary_ctx,
        secondary_ctx,
        simt_ctx,
        m,
        n,
        k,
        A,
        lda,
        B,
        ldb,
        C,
        ldc,
        alpha,
        beta,
        stream,
        "cutlass GemmNN");
}

void GemmNT(
    int m,
    int n,
    int k,
    const __half *A,
    int lda,
    const __half *B,
    int ldb,
    float *C,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    using Primary = TensorOpGemm<
        ColumnMajor,
        RowMajor,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        3>;
    using Secondary = TensorOpGemm<
        ColumnMajor,
        RowMajor,
        cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        2>;
    using Simt = SimtGemm<ColumnMajor, RowMajor>;
    static GemmContext<Primary> primary_ctx;
    static GemmContext<Secondary> secondary_ctx;
    static GemmContext<Simt> simt_ctx;
    RunGemmWithFallback(
        primary_ctx,
        secondary_ctx,
        simt_ctx,
        m,
        n,
        k,
        A,
        lda,
        B,
        ldb,
        C,
        ldc,
        alpha,
        beta,
        stream,
        "cutlass GemmNT");
}

} // namespace flstm
