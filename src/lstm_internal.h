#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/device_kernel.h>
#include <cutlass/epilogue/thread/conversion_op.h>
#include <cutlass/reduction/thread/reduction_operators.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace flashlstm::internal {

__device__ inline float sigmoidf(const float x) {
    return 1.0f / (1.0f + ::expf(-x));
}

using ElementInput = cutlass::half_t;
using ElementOutput = float;

constexpr int kElementsPer128Bits =
    128 / cutlass::sizeof_bits<ElementOutput>::value;

template <typename ElementAccumulator>
struct CutlassComputeSelector {
    using Type = ElementAccumulator;
};

template <>
struct CutlassComputeSelector<cutlass::half_t> {
    using Type = float;
};

template <typename ElementAccumulator>
using CutlassLinearCombinationOp =
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kElementsPer128Bits,
        ElementAccumulator,
        typename CutlassComputeSelector<ElementAccumulator>::Type>;

template <typename ElementAccumulator>
using CutlassGemmLargeT = cutlass::gemm::device::Gemm<
    ElementInput,
    cutlass::layout::RowMajor,
    ElementInput,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassLinearCombinationOp<ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd>;

template <typename ElementAccumulator>
using CutlassGemmSmallT = cutlass::gemm::device::Gemm<
    ElementInput,
    cutlass::layout::RowMajor,
    ElementInput,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassLinearCombinationOp<ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    4,
    4,
    false,
    cutlass::arch::OpMultiplyAdd>;

template <typename ElementAccumulator>
using CutlassGemmCudnnLikeT = cutlass::gemm::device::Gemm<
    ElementInput,
    cutlass::layout::RowMajor,
    ElementInput,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassLinearCombinationOp<ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd>;

template <typename ElementAccumulator>
using CutlassGemmCudnnSplitKT = cutlass::gemm::device::GemmSplitKParallel<
    ElementInput,
    cutlass::layout::RowMajor,
    ElementInput,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassLinearCombinationOp<ElementAccumulator>,
    cutlass::epilogue::thread::Convert<
        ElementOutput,
        kElementsPer128Bits,
        ElementAccumulator>,
    cutlass::reduction::thread::ReduceAdd<
        ElementOutput,
        ElementAccumulator,
        kElementsPer128Bits>,
    cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
    5,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;

template <typename ElementAccumulator>
inline typename CutlassGemmLargeT<ElementAccumulator>::Arguments
make_input_projection_arguments(
    int total_rows,
    int gates,
    int input,
    ElementInput* activations,
    int activations_stride,
    ElementInput* packed_weights,
    float* input_gates) {
    using Gemm = CutlassGemmLargeT<ElementAccumulator>;
    return typename Gemm::Arguments{
        {total_rows, gates, input},
        {activations, activations_stride},
        {packed_weights, activations_stride},
        {input_gates, gates},
        {input_gates, gates},
        {1.0f, 0.0f},
        1};
}

template <typename ElementAccumulator>
inline std::size_t input_projection_workspace_size(
    int total_rows,
    int gates,
    int input,
    ElementInput* activations,
    int activations_stride,
    ElementInput* packed_weights,
    float* input_gates) {
    using Gemm = CutlassGemmLargeT<ElementAccumulator>;
    auto args = make_input_projection_arguments<ElementAccumulator>(
        total_rows,
        gates,
        input,
        activations,
        activations_stride,
        packed_weights,
        input_gates);
    return Gemm::get_workspace_size(args);
}

}  // namespace flashlstm::internal
