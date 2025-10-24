#include "flashlstm/lstm_api.h"
#include "lstm_internal.h"
#include "error_checks.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <utility>
#include <limits>
#include <new>
#include <type_traits>
#include <cstdint>
#include <cstring>

#include "nvtx_profinst.h"

// TODO: remove once graph experiments stabilise.
bool g_flashlstm_force_graph_mode = true;

namespace flashlstm::kernels {
    using namespace flashlstm::internal;
    __global__ void populate_activations_kernel(const float *__restrict__ x,
                                                const float *__restrict__ h0,
                                                ElementInput *__restrict__ activations,
                                                std::size_t seq_len,
                                                std::size_t batch,
                                                std::size_t input_size,
                                                std::size_t hidden_size,
                                                std::size_t activations_stride);

    __global__ void combine_bias_kernel(const float *__restrict__ b_ih,
                                        const float *__restrict__ b_hh,
                                        float *__restrict__ bias_out,
                                        std::size_t elements);

    __global__ void pack_weights_kernel(const float *__restrict__ weight_ih,
                                        const float *__restrict__ weight_hh,
                                        ElementInput *__restrict__ packed,
                                        int hidden_size,
                                        int input_size,
                                        int input_hidden,
                                        int input_hidden_stride);

    __global__ void lstm_pointwise_kernel(const float *gates,
                                          int split_k_slices,
                                          int slice_stride,
                                          const float *input_gates,
                                          const float *bias,
                                          const float *c_prev,
                                          float *c_next,
                                          float *h_next,
                                          ElementInput *activations,
                                          int input_size,
                                          int activations_stride,
                                          ElementInput *next_hidden_tail,
                                          float *output_t,
                                          int batch_size,
                                          int hidden_size,
                                          float *gates_out);
} // namespace flashlstm::internal

namespace {
    using namespace flashlstm::internal;

    template<typename Pointer>
    void free_if_needed(Pointer *&ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }

    constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    inline dim3 make_grid_dim(std::size_t elements, int threads = 256) {
        if (elements == 0) {
            return dim3(0);
        }
        const std::size_t blocks = (elements + static_cast<std::size_t>(threads) - 1) / threads;
        const std::size_t capped = std::min<std::size_t>(
            blocks,
            std::numeric_limits<unsigned int>::max());
        return dim3(static_cast<unsigned int>(capped));
    }

    inline cudaError_t launch_populate_activations(const float *x,
                                                   const float *h0,
                                                   ElementInput *activations,
                                                   std::size_t seq_len,
                                                   std::size_t batch,
                                                   std::size_t input_size,
                                                   std::size_t hidden_size,
                                                   std::size_t activations_stride,
                                                   cudaStream_t stream) {
        const std::size_t padded_elements = seq_len * batch * activations_stride;
        if (padded_elements > 0) {
            cudaError_t memset_status = cudaMemsetAsync(
                activations, 0, padded_elements * sizeof(ElementInput), stream);
            if (memset_status != cudaSuccess) {
                return memset_status;
            }
        }
        const std::size_t input_elements = seq_len * batch * input_size;
        const std::size_t hidden_elements =
                (h0 != nullptr) ? batch * hidden_size : 0;
        const std::size_t total = input_elements + hidden_elements;
        if (total == 0) {
            return cudaSuccess;
        }
        constexpr int threads = 256;
        dim3 grid = make_grid_dim(total, threads);
        if (grid.x == 0) {
            return cudaErrorInvalidConfiguration;
        }
        flashlstm::kernels::populate_activations_kernel<<<grid, threads, 0, stream>>>(
            x, h0, activations, seq_len, batch, input_size, hidden_size, activations_stride);
        return cudaGetLastError();
    }

    cudaError_t launch_combine_bias(const float *b_ih,
                                    const float *b_hh,
                                    float *bias_out,
                                    std::size_t elements,
                                    cudaStream_t stream) {
        if (elements == 0) {
            return cudaSuccess;
        }
        constexpr int threads = 256;
        dim3 grid = make_grid_dim(elements, threads);
        if (grid.x == 0) {
            return cudaErrorInvalidConfiguration;
        }
        flashlstm::kernels::combine_bias_kernel<<<grid, threads, 0, stream>>>(b_ih, b_hh, bias_out, elements);
        return cudaGetLastError();
    }

    inline int compute_input_proj_chunk_timesteps(int chunk_size, std::size_t seq_len) {
        if (seq_len == 0) {
            return 0;
        }
        if (chunk_size <= 0) {
            return static_cast<int>(seq_len);
        }
        const int clamped = std::min<int>(chunk_size, static_cast<int>(seq_len));
        return std::max(1, clamped);
    }

    inline cudaError_t launch_pack_weights_kernel(const float *weight_ih,
                                                  const float *weight_hh,
                                                  ElementInput *packed,
                                                  int hidden_size,
                                                  int input_size,
                                                  int input_hidden,
                                                  int input_hidden_stride,
                                                  cudaStream_t stream) {
        const std::size_t elements =
                static_cast<std::size_t>(4 * hidden_size) * input_hidden_stride;
        if (elements == 0) {
            return cudaSuccess;
        }
        constexpr int threads = 256;
        dim3 grid = make_grid_dim(elements, threads);
        if (grid.x == 0) {
            return cudaErrorInvalidConfiguration;
        }
        flashlstm::kernels::pack_weights_kernel<<<grid, threads, 0, stream>>>(
            weight_ih,
            weight_hh,
            packed,
            hidden_size,
            input_size,
            input_hidden,
            input_hidden_stride);
        return cudaGetLastError();
    }

    struct GemmPlan {
        bool use_splitk = false;
        bool use_cudnn_like = false;
        bool use_small = false;
        int split_k_slices = 1;
        std::size_t workspace_bytes = 0;
    };

    struct LstmBuffersImpl {
        lstm_compute_precision_t precision;
        bool use_fp16_accumulator;
        std::size_t seq_len;
        std::size_t seq_len_capacity;
        std::size_t batch;
        std::size_t input_size;
        std::size_t hidden_size;
        int input_proj_chunk_size;
        std::size_t input_hidden_size;
        std::size_t input_hidden_stride;
        std::size_t weight_elements;
        std::size_t activations_elements;
        std::size_t bias_elements;
        std::size_t state_elements;
        std::size_t gates_elements;
        std::size_t input_gates_elements;

        ElementInput *packed_weights;
        ElementInput *activations;
        float *bias;
        float *h_prev;
        float *h_next;
        float *c_prev;
        float *c_next;
        float *gates;
        float *input_gates;
        float *splitk_output;
        void *workspace;
        std::size_t workspace_size;
        void *input_workspace;
        std::size_t input_workspace_size;
        std::size_t splitk_output_elements;
        float *x_staging;
        std::size_t x_staging_elements;
        float *b_ih_staging;
        float *b_hh_staging;
        float *output_buffer;
        std::size_t output_buffer_elements;
        float *hn_buffer;
        float *cn_buffer;

        bool weights_packed;
        GemmPlan plan;
        GemmPlan graph_plan;
        bool graph_plan_valid;
        lstm_execution_mode_t execution_mode;
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        std::size_t graph_seq_len;

        cudaStream_t stream;
        int multi_processor_count;

        struct GraphMemsetNodeStorage {
            cudaMemsetParams params{};
            cudaGraphNode_t node = nullptr;
            bool enabled = false;
        };

        struct GraphPopulateNodeStorage {
            cudaKernelNodeParams params{};
            void *args[8]{};
            const float *x = nullptr;
            const float *h0 = nullptr;
            ElementInput *activations = nullptr;
            std::size_t seq_len = 0;
            std::size_t batch = 0;
            std::size_t input_size = 0;
            std::size_t hidden_size = 0;
            std::size_t activations_stride = 0;
            cudaGraphNode_t node = nullptr;
        };

        struct GraphCombineBiasNodeStorage {
            cudaKernelNodeParams params{};
            void *args[4]{};
            const float *b_ih = nullptr;
            const float *b_hh = nullptr;
            float *bias_out = nullptr;
            std::size_t elements = 0;
            cudaGraphNode_t node = nullptr;
        };

        struct GraphKernelNodeStorage {
            cudaKernelNodeParams params{};
            std::vector<void *> arg_ptrs;
            std::vector<std::uint8_t> arg_storage;
            cudaGraphNode_t node = nullptr;
        };

        struct GraphPointwiseNodeStorage {
            cudaKernelNodeParams params{};
            void *args[16]{};
            const float *gates = nullptr;
            int split_k_slices = 0;
            int slice_stride = 0;
            const float *input_gates = nullptr;
            const float *bias = nullptr;
            const float *c_prev = nullptr;
            float *c_next = nullptr;
            float *h_next = nullptr;
            ElementInput *activations = nullptr;
            int input_size = 0;
            int activations_stride = 0;
            ElementInput *next_hidden_tail = nullptr;
            float *output_t = nullptr;
            int batch_size = 0;
            int hidden_size = 0;
            float *gates_out = nullptr;
            cudaGraphNode_t node = nullptr;
        };

        struct GraphMemcpyNodeStorage {
            cudaMemcpy3DParms params{};
            cudaGraphNode_t node = nullptr;
            bool enabled = false;
        };

        GraphMemsetNodeStorage graph_activations_memset;
        GraphPopulateNodeStorage graph_populate_node;
        GraphCombineBiasNodeStorage graph_combine_bias_node;
        std::vector<GraphKernelNodeStorage> graph_input_projection_nodes;
        std::vector<GraphKernelNodeStorage> graph_hidden_gemm_nodes;
        std::vector<GraphPointwiseNodeStorage> graph_pointwise_nodes;
        GraphMemcpyNodeStorage graph_hn_memcpy_node;
        GraphMemcpyNodeStorage graph_cn_memcpy_node;
    };

    template<typename Gemm>
    cutlass::Status prepare_cutlass_gemm_kernel_node(
        typename Gemm::Arguments const &args,
        void *workspace,
        LstmBuffersImpl::GraphKernelNodeStorage &storage) {
        using Kernel = typename Gemm::GemmKernel;
        using ThreadblockSwizzle = typename Gemm::ThreadblockSwizzle;
        using ThreadblockShape = typename Gemm::ThreadblockShape;

        ThreadblockSwizzle swizzle;
        auto grid_shape = swizzle.get_tiled_shape(
            args.problem_size,
            {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            args.split_k_slices);

        if (Gemm::kSplitKSerial) {
            if (args.split_k_slices > 1) {
                if (!workspace) {
                    return cutlass::Status::kErrorWorkspaceNull;
                }
                size_t bytes = Gemm::get_workspace_size(args);
                cudaError_t memset_status = cudaMemset(workspace, 0, bytes);
                if (memset_status != cudaSuccess) {
                    return cutlass::Status::kErrorInternal;
                }
            }
        } else {
            if (args.split_k_slices > 1) {
                return cutlass::Status::kErrorInvalidProblem;
            }
        }

        typename Kernel::Params params(
            args.problem_size,
            grid_shape,
            args.ref_A.non_const_ref(),
            args.ref_B.non_const_ref(),
            args.ref_C.non_const_ref(),
            args.ref_D,
            args.epilogue,
            static_cast<int *>(workspace),
            args.gather_A_indices,
            args.gather_B_indices,
            args.scatter_D_indices);

        storage.arg_storage.resize(sizeof(params));
        auto *stored_params = reinterpret_cast<typename Kernel::Params *>(storage.arg_storage.data());
        *stored_params = params;

        storage.arg_ptrs.resize(1);
        storage.arg_ptrs[0] = stored_params;

        dim3 grid = swizzle.get_grid_shape(grid_shape);
        dim3 block(Kernel::kThreadCount, 1, 1);
        int smem_size = int(sizeof(typename Kernel::SharedStorage));

        if (smem_size >= (48 << 10)) {
            cudaError_t attr_status = cudaFuncSetAttribute(
                cutlass::Kernel<Kernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size);
            if (attr_status != cudaSuccess) {
                return cutlass::Status::kErrorInternal;
            }
        }

        storage.params.func = reinterpret_cast<void *>(cutlass::Kernel<Kernel>);
        storage.params.gridDim = grid;
        storage.params.blockDim = block;
        storage.params.sharedMemBytes = smem_size;
        storage.params.kernelParams = storage.arg_ptrs.data();
        storage.params.extra = nullptr;

        return cutlass::Status::kSuccess;
    }

    template<typename SplitKGemm>
    cutlass::Status prepare_splitk_gemm_kernel_node(
        typename SplitKGemm::Arguments const &args,
        float *workspace,
        LstmBuffersImpl::GraphKernelNodeStorage &storage) {
        using Kernel = typename SplitKGemm::GemmKernel;
        using ThreadblockSwizzle = typename SplitKGemm::ThreadblockSwizzle;
        using ThreadblockShape = typename SplitKGemm::ThreadblockShape;

        if (workspace == nullptr) {
            return cutlass::Status::kErrorWorkspaceNull;
        }

        ThreadblockSwizzle swizzle;
        auto grid_shape = swizzle.get_tiled_shape(
            args.problem_size,
            {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            args.split_k_slices);

        typename Kernel::Params params(
            args.problem_size,
            grid_shape,
            args.ref_A.non_const_ref(),
            args.ref_B.non_const_ref(),
            args.ref_D.non_const_ref(),
            args.convert,
            int64_t(args.problem_size.m()) * args.problem_size.n());

        storage.arg_storage.resize(sizeof(params));
        auto *stored_params = reinterpret_cast<typename Kernel::Params *>(storage.arg_storage.data());
        *stored_params = params;

        storage.arg_ptrs.resize(1);
        storage.arg_ptrs[0] = stored_params;

        dim3 grid = swizzle.get_grid_shape(grid_shape);
        dim3 block(Kernel::kThreadCount, 1, 1);
        int smem_size = int(sizeof(typename Kernel::SharedStorage));

        if (smem_size >= (48 << 10)) {
            cudaError_t attr_status = cudaFuncSetAttribute(
                cutlass::Kernel<Kernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size);
            if (attr_status != cudaSuccess) {
                return cutlass::Status::kErrorInternal;
            }
        }

        storage.params.func = reinterpret_cast<void *>(cutlass::Kernel<Kernel>);
        storage.params.gridDim = grid;
        storage.params.blockDim = block;
        storage.params.sharedMemBytes = smem_size;
        storage.params.kernelParams = storage.arg_ptrs.data();
        storage.params.extra = nullptr;

        return cutlass::Status::kSuccess;
    }

    template<typename ElementAccumulator>
    inline GemmPlan plan_gemm_strategy(int m,
                                       int n,
                                       int k,
                                       int lda_a,
                                       int lda_b,
                                       ElementInput *activations_base,
                                       ElementInput *packed_weights,
                                       float *d_gates,
                                       int multi_processor_count) {
        using CutlassGemmLarge = CutlassGemmLargeT<ElementAccumulator>;
        using CutlassGemmSmall = CutlassGemmSmallT<ElementAccumulator>;
        using CutlassGemmCudnnLike = CutlassGemmCudnnLikeT<ElementAccumulator>;
        using CutlassGemmCudnnSplitK = CutlassGemmCudnnSplitKT<ElementAccumulator>;

        GemmPlan plan{};
        plan.split_k_slices = 1;
        plan.workspace_bytes = 0;

        if (m == 0 || n == 0 || k == 0) {
            return plan;
        }

        using CudnnLikeThreadblock = typename CutlassGemmCudnnLike::ThreadblockShape;
        const bool aligned_for_cudnn_like =
                (m <= CudnnLikeThreadblock::kM) &&
                (n % CudnnLikeThreadblock::kN == 0) &&
                (k % CudnnLikeThreadblock::kK == 0) &&
                (k % CutlassGemmCudnnLike::kAlignmentA == 0) &&
                (k % CutlassGemmCudnnLike::kAlignmentB == 0) &&
                (n % CutlassGemmCudnnLike::kAlignmentC == 0);

        bool use_cudnn_splitk_kernel = false;
        int split_k_slices = 1;

        if (aligned_for_cudnn_like && multi_processor_count > 0) {
            const int tile_n = CudnnLikeThreadblock::kN;
            const int tile_k = CudnnLikeThreadblock::kK;
            const int base_blocks = (n + tile_n - 1) / tile_n;
            const int max_slices_from_k = std::max(1, k / tile_k);
            const int max_slices = std::min(4, max_slices_from_k);
            if (base_blocks < multi_processor_count && max_slices > 1) {
                const int needed_slices =
                        (multi_processor_count + base_blocks - 1) / base_blocks;
                split_k_slices = std::min(std::max(1, needed_slices), max_slices);
                if (split_k_slices > 1) {
                    use_cudnn_splitk_kernel = true;
                }
            }
        }

        plan.use_splitk = use_cudnn_splitk_kernel;
        plan.use_cudnn_like = aligned_for_cudnn_like && !use_cudnn_splitk_kernel;
        plan.use_small = !aligned_for_cudnn_like &&
                         ((m <= 64) || (n <= 256) || (k <= 256));
        plan.split_k_slices = split_k_slices;

        if (plan.use_splitk) {
            typename CutlassGemmCudnnSplitK::Arguments probe(
                {m, n, k},
                {activations_base, lda_a},
                {packed_weights, lda_b},
                {d_gates, n},
                {d_gates, n},
                {1.0f, 0.0f},
                split_k_slices);
            plan.workspace_bytes = CutlassGemmCudnnSplitK::get_workspace_size(probe);
        } else if (plan.use_cudnn_like) {
            typename CutlassGemmCudnnLike::Arguments probe(
                {m, n, k},
                {activations_base, lda_a},
                {packed_weights, lda_b},
                {d_gates, n},
                {d_gates, n},
                {1.0f, 0.0f},
                1);
            plan.workspace_bytes = CutlassGemmCudnnLike::get_workspace_size(probe);
        } else if (plan.use_small) {
            typename CutlassGemmSmall::Arguments probe(
                {m, n, k},
                {activations_base, lda_a},
                {packed_weights, lda_b},
                {d_gates, n},
                {d_gates, n},
                {1.0f, 0.0f},
                1);
            plan.workspace_bytes = CutlassGemmSmall::get_workspace_size(probe);
        } else {
            typename CutlassGemmLarge::Arguments probe(
                {m, n, k},
                {activations_base, lda_a},
                {packed_weights, lda_b},
                {d_gates, n},
                {d_gates, n},
                {1.0f, 0.0f},
                1);
            plan.workspace_bytes = CutlassGemmLarge::get_workspace_size(probe);
        }

        return plan;
    }
} // namespace

namespace {
    template<typename ElementAccumulator>
    cutlass::Status launch_splitk_gemm_mainloop(
        typename CutlassGemmCudnnSplitKT<ElementAccumulator>::Arguments const &args,
        float *workspace,
        cudaStream_t stream) {
        NVTX_SCOPED_RANGE("FlashLSTM::launch_splitk_gemm_mainloop");
        using SplitKGemm = CutlassGemmCudnnSplitKT<ElementAccumulator>;
        using GemmKernel = typename SplitKGemm::GemmKernel;
        using ThreadblockSwizzle = typename SplitKGemm::ThreadblockSwizzle;
        using ThreadblockShape = typename SplitKGemm::ThreadblockShape;

        if (workspace == nullptr) {
            return cutlass::Status::kErrorWorkspaceNull;
        }

        ThreadblockSwizzle swizzle;
        cutlass::gemm::GemmCoord grid_shape = swizzle.get_tiled_shape(
            args.problem_size,
            {
                ThreadblockShape::kM,
                ThreadblockShape::kN,
                ThreadblockShape::kK
            },
            args.split_k_slices);

        typename GemmKernel::Params params(
            args.problem_size,
            grid_shape,
            args.ref_A.non_const_ref(),
            args.ref_B.non_const_ref(),
            args.ref_D.non_const_ref(),
            args.convert,
            int64_t(args.problem_size.m()) * args.problem_size.n());

        dim3 grid = swizzle.get_grid_shape(grid_shape);
        dim3 block(GemmKernel::kThreadCount, 1, 1);

        int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
        if (smem_size >= (48 << 10)) {
            cudaError_t attr_result = cudaFuncSetAttribute(
                cutlass::Kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size);
            if (attr_result != cudaSuccess) {
                return cutlass::Status::kErrorInternal;
            }
        }

        cutlass::arch::synclog_setup();
        cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params);

        cudaError_t launch_result = cudaGetLastError();
        if (launch_result != cudaSuccess) {
            return cutlass::Status::kErrorInternal;
        }

        return cutlass::Status::kSuccess;
    }
} // namespace

namespace {
    void invalidate_graph(LstmBuffersImpl *impl) {
        if (!impl) {
            return;
        }
        if (impl->graph_exec != nullptr) {
            cudaGraphExecDestroy(impl->graph_exec);
            impl->graph_exec = nullptr;
        }
        if (impl->graph != nullptr) {
            cudaGraphDestroy(impl->graph);
            impl->graph = nullptr;
        }
        impl->graph_seq_len = 0;
        impl->graph_plan_valid = false;
        impl->graph_input_projection_nodes.clear();
        impl->graph_hidden_gemm_nodes.clear();
        impl->graph_pointwise_nodes.clear();
        impl->graph_activations_memset = {};
        impl->graph_populate_node = {};
        impl->graph_combine_bias_node = {};
        impl->graph_hn_memcpy_node = {};
        impl->graph_cn_memcpy_node = {};
    }

    void destroy_impl(LstmBuffersImpl *impl) {
        if (!impl) {
            return;
        }
        invalidate_graph(impl);
        free_if_needed(impl->packed_weights);
        free_if_needed(impl->activations);
        free_if_needed(impl->bias);
        free_if_needed(impl->h_prev);
        free_if_needed(impl->h_next);
        free_if_needed(impl->c_prev);
        free_if_needed(impl->c_next);
        free_if_needed(impl->gates);
        free_if_needed(impl->input_gates);
        free_if_needed(impl->splitk_output);
        free_if_needed(impl->workspace);
        free_if_needed(impl->input_workspace);
        free_if_needed(impl->x_staging);
        free_if_needed(impl->b_ih_staging);
        free_if_needed(impl->b_hh_staging);
        free_if_needed(impl->output_buffer);
        free_if_needed(impl->hn_buffer);
        free_if_needed(impl->cn_buffer);
        if (impl->stream) {
            cudaStreamDestroy(impl->stream);
        }
        delete impl;
    }
} // namespace

extern "C" int lstm_create_buffers(lstm_compute_precision_t precision,
                                   std::size_t seq_len,
                                   std::size_t batch,
                                   std::size_t input_size,
                                   std::size_t hidden_size,
                                   int input_proj_chunk_size,
                                   lstm_buffers *buffers) {
    buffers->impl = nullptr;

    auto *impl = new(std::nothrow) LstmBuffersImpl();
    if (!impl) {
        return cudaErrorMemoryAllocation;
    }

    impl->precision = precision;
    impl->use_fp16_accumulator = (precision == LSTM_COMPUTE_PRECISION_FP16_ACC16);
    impl->seq_len = seq_len;
    impl->seq_len_capacity = seq_len;
    impl->batch = batch;
    impl->input_size = input_size;
    impl->hidden_size = hidden_size;
    impl->input_proj_chunk_size = input_proj_chunk_size;
    impl->input_hidden_size = input_size + hidden_size;
    constexpr std::size_t kTensorCoreAlignment = 8;
    impl->input_hidden_stride = align_up(impl->input_hidden_size, kTensorCoreAlignment);

    impl->weight_elements = 4 * hidden_size * impl->input_hidden_stride;
    impl->activations_elements = seq_len * batch * impl->input_hidden_stride;
    impl->bias_elements = 4 * hidden_size;
    impl->state_elements = batch * hidden_size;
    impl->gates_elements = impl->state_elements * 4;
    impl->input_gates_elements = seq_len * impl->gates_elements;

    impl->packed_weights = nullptr;
    impl->activations = nullptr;
    impl->bias = nullptr;
    impl->h_prev = nullptr;
    impl->h_next = nullptr;
    impl->c_prev = nullptr;
    impl->c_next = nullptr;
        impl->gates = nullptr;
        impl->input_gates = nullptr;
        impl->splitk_output = nullptr;
        impl->workspace = nullptr;
        impl->workspace_size = 0;
        impl->input_workspace = nullptr;
        impl->input_workspace_size = 0;
        impl->splitk_output_elements = 0;
    impl->x_staging = nullptr;
    impl->x_staging_elements = 0;
    impl->b_ih_staging = nullptr;
    impl->b_hh_staging = nullptr;
    impl->output_buffer = nullptr;
    impl->output_buffer_elements = 0;
    impl->hn_buffer = nullptr;
    impl->cn_buffer = nullptr;
    impl->weights_packed = (impl->weight_elements == 0);
    impl->plan = {};
    impl->graph_plan = {};
    impl->graph_plan_valid = false;
    impl->execution_mode = LSTM_EXECUTION_MODE_IMMEDIATE;
    impl->graph = nullptr;
    impl->graph_exec = nullptr;
    impl->graph_seq_len = 0;
    impl->stream = nullptr;
    impl->multi_processor_count = -1;

    int device_idx = 0;
    if (cudaGetDevice(&device_idx) == cudaSuccess) {
        int mp_count = 0;
        if (cudaDeviceGetAttribute(&mp_count,
                                   cudaDevAttrMultiProcessorCount,
                                   device_idx) == cudaSuccess) {
            impl->multi_processor_count = mp_count;
        }
    }

    auto allocate_if_needed = [](auto **ptr, const std::size_t elements) -> cudaError_t {
        if (elements == 0) {
            *ptr = nullptr;
            return cudaSuccess;
        }
        return cudaMalloc(ptr, elements * sizeof(**ptr));
    };

    auto alloc_guard = [&](const cudaError_t err) -> int {
        if (err != cudaSuccess) {
            destroy_impl(impl);
            return err;
        }
        return 0;
    };

    ERR_PROPAGATE(alloc_guard(allocate_if_needed(&impl->packed_weights, impl->weight_elements)));
    ERR_PROPAGATE(alloc_guard(allocate_if_needed(&impl->activations, impl->activations_elements)));
    ERR_PROPAGATE(alloc_guard(allocate_if_needed(&impl->bias, impl->bias_elements)));
    ERR_PROPAGATE(alloc_guard(allocate_if_needed(&impl->h_prev, impl->state_elements)));
    ERR_PROPAGATE(alloc_guard(allocate_if_needed(&impl->h_next, impl->state_elements)));
    ERR_PROPAGATE(alloc_guard(allocate_if_needed(&impl->c_prev, impl->state_elements)));
    ERR_PROPAGATE(alloc_guard(allocate_if_needed(&impl->c_next, impl->state_elements)));
    ERR_PROPAGATE(alloc_guard(allocate_if_needed(&impl->gates, impl->gates_elements)));
    ERR_PROPAGATE(alloc_guard(allocate_if_needed(&impl->input_gates, impl->input_gates_elements)));

    const int batch_i = static_cast<int>(batch);
    const int input_i = static_cast<int>(input_size);
    const int input_hidden_stride_i = static_cast<int>(impl->input_hidden_stride);
    const int hidden_k = input_hidden_stride_i - input_i;
    const int gates_i = static_cast<int>(4 * hidden_size);

    ElementInput *activations_hidden_base =
            impl->activations ? impl->activations + input_i : nullptr;
    ElementInput *packed_hidden_base =
            impl->packed_weights ? impl->packed_weights + input_i : nullptr;

    const GemmPlan plan = impl->use_fp16_accumulator
                              ? plan_gemm_strategy<cutlass::half_t>(batch_i,
                                                                    gates_i,
                                                                    hidden_k,
                                                                    input_hidden_stride_i,
                                                                    input_hidden_stride_i,
                                                                    activations_hidden_base,
                                                                    packed_hidden_base,
                                                                    impl->gates,
                                                                    impl->multi_processor_count)
                              : plan_gemm_strategy<float>(batch_i,
                                                          gates_i,
                                                          hidden_k,
                                                          input_hidden_stride_i,
                                                          input_hidden_stride_i,
                                                          activations_hidden_base,
                                                          packed_hidden_base,
                                                          impl->gates,
                                                          impl->multi_processor_count);
    impl->plan = plan;

    const int input_proj_chunk_timesteps =
            compute_input_proj_chunk_timesteps(impl->input_proj_chunk_size, seq_len);
    const int input_proj_chunk_rows = input_proj_chunk_timesteps * batch_i;
    std::size_t input_workspace_bytes = 0;
    if (input_proj_chunk_rows > 0) {
        input_workspace_bytes = impl->use_fp16_accumulator
                                    ? input_projection_workspace_size<cutlass::half_t>(
                                        input_proj_chunk_rows,
                                        gates_i,
                                        input_i,
                                        impl->activations,
                                        input_hidden_stride_i,
                                        impl->packed_weights,
                                        impl->input_gates)
                                    : input_projection_workspace_size<float>(
                                        input_proj_chunk_rows,
                                        gates_i,
                                        input_i,
                                        impl->activations,
                                        input_hidden_stride_i,
                                        impl->packed_weights,
                                        impl->input_gates);
    }

    impl->workspace_size = plan.workspace_bytes;
    impl->input_workspace_size = input_workspace_bytes;
    if (impl->workspace_size > 0) {
        if (const cudaError_t ws_status = cudaMalloc(&impl->workspace, impl->workspace_size);
            ws_status != cudaSuccess) {
            destroy_impl(impl);
            return ws_status;
        }
    }
    if (impl->input_workspace_size > 0) {
        if (const cudaError_t ws_status = cudaMalloc(&impl->input_workspace, impl->input_workspace_size);
            ws_status != cudaSuccess) {
            destroy_impl(impl);
            return ws_status;
        }
    }

    if (const cudaError_t stream_status = cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking);
        stream_status != cudaSuccess) {
        destroy_impl(impl);
        return stream_status;
    }
    buffers->impl = impl;
    return 0;
}

extern "C" int lstm_destroy_buffers(lstm_buffers *buffers) {
    if (!buffers) {
        return cudaErrorInvalidValue;
    }
    destroy_impl(static_cast<LstmBuffersImpl *>(buffers->impl));
    buffers->impl = nullptr;
    return 0;
}

extern "C" int lstm_set_execution_mode(lstm_buffers *buffers, lstm_execution_mode_t mode) {
    if (!buffers || !buffers->impl) {
        return cudaErrorInvalidValue;
    }
    if (mode != LSTM_EXECUTION_MODE_IMMEDIATE &&
        mode != LSTM_EXECUTION_MODE_GRAPH) {
        return cudaErrorInvalidValue;
    }
    auto *impl = static_cast<LstmBuffersImpl *>(buffers->impl);
    if (impl->execution_mode == mode) {
        return 0;
    }
    invalidate_graph(impl);
    impl->execution_mode = mode;
    return 0;
}

extern "C" int lstm_pack_weights(lstm_compute_precision_t precision,
                                 const float *weight_ih,
                                 const float *weight_hh,
                                 std::size_t input_size,
                                 std::size_t hidden_size,
                                 lstm_buffers *buffers) {
    if (!buffers || !buffers->impl) {
        return cudaErrorInvalidValue;
    }

    auto *impl = static_cast<LstmBuffersImpl *>(buffers->impl);
    if (impl->precision != precision) {
        std::fprintf(stderr, "Precision mismatch in lstm_pack_weights\n");
        return cudaErrorInvalidValue;
    }
    if (impl->input_size != input_size || impl->hidden_size != hidden_size) {
        std::fprintf(stderr, "Dimension mismatch in lstm_pack_weights\n");
        return cudaErrorInvalidValue;
    }

    const int hidden_i = static_cast<int>(hidden_size);
    const int input_i = static_cast<int>(input_size);
    const int input_hidden_i = static_cast<int>(impl->input_hidden_size);
    const int input_hidden_stride_i = static_cast<int>(impl->input_hidden_stride);

    cudaError_t launch_status = launch_pack_weights_kernel(
        weight_ih,
        weight_hh,
        impl->packed_weights,
        hidden_i,
        input_i,
        input_hidden_i,
        input_hidden_stride_i,
        impl->stream);

    if (launch_status != cudaSuccess) {
        return launch_status;
    }

    cudaError_t sync_status = cudaStreamSynchronize(impl->stream);
    if (sync_status != cudaSuccess) {
        return sync_status;
    }
    impl->weights_packed = true;
    return 0;
}

int ensure_sequence_capacity(std::size_t seq_len,
                             std::size_t batch,
                             LstmBuffersImpl *impl) {
    const std::size_t activations_stride = impl->input_hidden_stride;
    if (seq_len > impl->seq_len_capacity) {
        const std::size_t new_capacity = seq_len;
        const std::size_t new_activations_elements =
                new_capacity * batch * activations_stride;
        const std::size_t new_input_gates_elements =
                new_capacity * impl->gates_elements;
        invalidate_graph(impl);
        free_if_needed(impl->activations);
        free_if_needed(impl->input_gates);
        cudaError_t alloc_status = cudaSuccess;
        if (new_activations_elements > 0) {
            alloc_status = cudaMalloc(
                &impl->activations,
                new_activations_elements * sizeof(ElementInput));
        }
        if (alloc_status == cudaSuccess && new_input_gates_elements > 0) {
            alloc_status = cudaMalloc(
                &impl->input_gates,
                new_input_gates_elements * sizeof(float));
        }
        if (alloc_status != cudaSuccess) {
            free_if_needed(impl->activations);
            free_if_needed(impl->input_gates);
            return alloc_status;
        }
        impl->activations_elements = new_activations_elements;
        impl->input_gates_elements = new_input_gates_elements;
        impl->seq_len_capacity = new_capacity;
    }
    if (impl->seq_len != seq_len) {
        impl->seq_len = seq_len;
    }
    return 0;
}

template<typename ElementAccumulator>
int prepare_lstm_execution(std::size_t seq_len,
                           std::size_t batch,
                           std::size_t input_size,
                           std::size_t hidden_size,
                           LstmBuffersImpl *impl,
                           GemmPlan *out_plan,
                           int *out_input_proj_chunk_timesteps) {
    const std::size_t activations_stride = impl->input_hidden_stride;
    const int batch_i = static_cast<int>(batch);
    const int input_i = static_cast<int>(input_size);
    const int input_hidden_stride_i = static_cast<int>(activations_stride);
    const int hidden_k = input_hidden_stride_i - input_i;
    const int gates_i = static_cast<int>(4 * hidden_size);

    ElementInput *activations_base = impl->activations;
    ElementInput *packed_weights = impl->packed_weights;
    ElementInput *packed_hidden_base =
            packed_weights ? packed_weights + input_i : nullptr;
    ElementInput *activations_hidden_base =
            activations_base ? activations_base + input_i : nullptr;

    GemmPlan plan = plan_gemm_strategy<ElementAccumulator>(batch_i,
                                                           gates_i,
                                                           hidden_k,
                                                           input_hidden_stride_i,
                                                           input_hidden_stride_i,
                                                           activations_hidden_base,
                                                           packed_hidden_base,
                                                           impl->gates,
                                                           impl->multi_processor_count);
    impl->plan = plan;
    if (out_plan) {
        *out_plan = plan;
    }

    const int input_proj_chunk_timesteps =
            compute_input_proj_chunk_timesteps(impl->input_proj_chunk_size, seq_len);
    if (out_input_proj_chunk_timesteps) {
        *out_input_proj_chunk_timesteps = input_proj_chunk_timesteps;
    }

    const int input_proj_chunk_rows = input_proj_chunk_timesteps * batch_i;
    std::size_t input_workspace_bytes = 0;
    if (input_proj_chunk_rows > 0) {
        input_workspace_bytes = input_projection_workspace_size<ElementAccumulator>(
            input_proj_chunk_rows,
            gates_i,
            input_i,
            activations_base,
            input_hidden_stride_i,
            packed_weights,
            impl->input_gates);
    }

    const std::size_t plan_workspace_bytes = plan.workspace_bytes;
    if (plan_workspace_bytes > impl->workspace_size) {
        invalidate_graph(impl);
        free_if_needed(impl->workspace);
        cudaError_t ws_status = cudaSuccess;
        if (plan_workspace_bytes > 0) {
            ws_status = cudaMalloc(&impl->workspace, plan_workspace_bytes);
        }
        if (ws_status != cudaSuccess) {
            return ws_status;
        }
        impl->workspace_size = plan_workspace_bytes;
    }

    if (plan.use_splitk) {
        const std::size_t split_k_slices = std::max(1, plan.split_k_slices);
        const std::size_t needed_splitk_output =
            split_k_slices * static_cast<std::size_t>(batch_i) * static_cast<std::size_t>(gates_i);
        if (needed_splitk_output > impl->splitk_output_elements) {
            invalidate_graph(impl);
            free_if_needed(impl->splitk_output);
            cudaError_t splitk_status = cudaSuccess;
            if (needed_splitk_output > 0) {
                splitk_status = cudaMalloc(
                    &impl->splitk_output,
                    needed_splitk_output * sizeof(float));
            }
            if (splitk_status != cudaSuccess) {
                impl->splitk_output_elements = 0;
                return splitk_status;
            }
            impl->splitk_output_elements = needed_splitk_output;
        }
    }

    if (input_workspace_bytes > impl->input_workspace_size) {
        invalidate_graph(impl);
        free_if_needed(impl->input_workspace);
        cudaError_t ws_status = cudaSuccess;
        if (input_workspace_bytes > 0) {
            ws_status = cudaMalloc(&impl->input_workspace, input_workspace_bytes);
        }
        if (ws_status != cudaSuccess) {
            return ws_status;
        }
        impl->input_workspace_size = input_workspace_bytes;
    }

    return 0;
}

template<typename ElementAccumulator>
int lstm_forward_impl(const float *x,
                      const float *b_ih,
                      const float *b_hh,
                      const float *h0,
                      const float *c0,
                      float *output,
                      float *hn,
                      float *cn,
                      const std::size_t seq_len,
                      const std::size_t batch,
                      const std::size_t input_size,
                      const std::size_t hidden_size,
                      LstmBuffersImpl *impl) {
    const std::size_t activations_stride = impl->input_hidden_stride;
    if (seq_len > impl->seq_len_capacity) {
        const std::size_t new_capacity = seq_len;
        const std::size_t new_activations_elements =
                new_capacity * batch * activations_stride;
        const std::size_t new_input_gates_elements =
                new_capacity * impl->gates_elements;
        invalidate_graph(impl);
        free_if_needed(impl->activations);
        free_if_needed(impl->input_gates);
        cudaError_t alloc_status = cudaSuccess;
        if (new_activations_elements > 0) {
            alloc_status = cudaMalloc(
                &impl->activations,
                new_activations_elements * sizeof(ElementInput));
        }
        if (alloc_status == cudaSuccess && new_input_gates_elements > 0) {
            alloc_status = cudaMalloc(
                &impl->input_gates,
                new_input_gates_elements * sizeof(float));
        }
        if (alloc_status != cudaSuccess) {
            free_if_needed(impl->activations);
            free_if_needed(impl->input_gates);
            return alloc_status;
        }
        impl->activations_elements = new_activations_elements;
        impl->input_gates_elements = new_input_gates_elements;
        impl->seq_len_capacity = new_capacity;
    }
    if (impl->seq_len != seq_len) {
        impl->seq_len = seq_len;
    }

    const std::size_t bias_elems = 4 * hidden_size;
    const std::size_t state_elems = batch * hidden_size;
    const std::size_t gates_per_timestep = state_elems * 4;

    const int batch_i = static_cast<int>(batch);
    const int input_i = static_cast<int>(input_size);
    const int hidden_i = static_cast<int>(hidden_size);
    const int input_hidden_stride_i = static_cast<int>(activations_stride);
    const int hidden_k = input_hidden_stride_i - input_i;
    const int gates_i = static_cast<int>(4 * hidden_size);
    const int state_i = static_cast<int>(state_elems);

    cudaStream_t stream = impl->stream;
    cudaStream_t init_stream = stream;
    FLASHLSTM_CHECK_CUDA(launch_populate_activations(x,
        h0,
        impl->activations,
        seq_len,
        batch,
        input_size,
        hidden_size,
        activations_stride,
        init_stream));

    FLASHLSTM_CHECK_CUDA(launch_combine_bias(b_ih,
        b_hh,
        impl->bias,
        bias_elems,
        init_stream));

    float *h_prev_ptr = impl->h_prev;
    float *h_next_ptr = impl->h_next;
    float *c_prev_ptr = impl->c_prev;
    float *c_next_ptr = impl->c_next;
    ElementInput *activations_base = impl->activations;
    ElementInput *packed_weights = impl->packed_weights;
    float *gates_ptr = impl->gates;
    float *bias_ptr = impl->bias;
    float *input_gates_base = impl->input_gates;
    float *splitk_output_ptr = impl->splitk_output;

    using CutlassGemmLarge = CutlassGemmLargeT<ElementAccumulator>;
    using CutlassGemmSmall = CutlassGemmSmallT<ElementAccumulator>;
    using CutlassGemmCudnnLike = CutlassGemmCudnnLikeT<ElementAccumulator>;
    using CutlassGemmCudnnSplitK = CutlassGemmCudnnSplitKT<ElementAccumulator>;
    ElementInput *packed_hidden_base =
            packed_weights ? packed_weights + input_i : nullptr;

    GemmPlan plan{};
    int input_proj_chunk_timesteps = 0;
    int prepare_status = prepare_lstm_execution<ElementAccumulator>(
        seq_len,
        batch,
        input_size,
        hidden_size,
        impl,
        &plan,
        &input_proj_chunk_timesteps);
    if (prepare_status != 0) {
        return prepare_status;
    }

    void *workspace = impl->workspace;
    void *projection_workspace = impl->input_workspace;
    const bool use_cudnn_splitk_kernel = plan.use_splitk;
    const bool use_cudnn_like_kernel = plan.use_cudnn_like;
    const bool use_small_kernel = plan.use_small;
    const int split_k_slices = plan.split_k_slices;

    if (use_cudnn_splitk_kernel && splitk_output_ptr == nullptr) {
        return cudaErrorMemoryAllocation;
    }

    CutlassGemmLarge gemm_large;
    CutlassGemmSmall gemm_small;
    CutlassGemmCudnnLike gemm_cudnn_like;
    CutlassGemmLarge input_projection_gemm;
    const int chunk_timesteps = input_proj_chunk_timesteps;
    auto launch_input_projection = [&](std::size_t chunk_start,
                                       void *workspace_ptr) -> cutlass::Status {
        if (chunk_timesteps <= 0 || chunk_start >= seq_len) {
            return cutlass::Status::kSuccess;
        }
        const std::size_t remaining = seq_len - chunk_start;
        const int current_chunk_timesteps =
                static_cast<int>(std::min<std::size_t>(chunk_timesteps, remaining));
        const int current_rows = current_chunk_timesteps * batch_i;
        if (current_rows <= 0) {
            return cutlass::Status::kSuccess;
        }
        ElementInput *activations_chunk =
                activations_base
                    ? activations_base + chunk_start * static_cast<std::size_t>(batch_i) * input_hidden_stride_i
                    : nullptr;
        float *input_gates_chunk =
                input_gates_base ? input_gates_base + chunk_start * gates_per_timestep : nullptr;

        auto input_projection_args =
                make_input_projection_arguments<ElementAccumulator>(
                    current_rows,
                    gates_i,
                    input_i,
                    activations_chunk,
                    input_hidden_stride_i,
                    packed_weights,
                    input_gates_chunk);

        cutlass::Status input_status =
                input_projection_gemm.initialize(input_projection_args, workspace_ptr, stream);
        if (input_status != cutlass::Status::kSuccess) {
            return input_status;
        }
        return input_projection_gemm(stream);
    };

    auto run_time_step = [&](std::size_t t) -> int {
        ElementInput *activations_step =
                activations_base + t * batch_i * input_hidden_stride_i;
        ElementInput *hidden_activations_step = activations_step + input_i;
        ElementInput *next_hidden_tail =
                (t + 1 < seq_len)
                    ? activations_base + (t + 1) * batch_i * input_hidden_stride_i + input_i
                    : nullptr;
        const float *input_gates_step =
                input_gates_base ? input_gates_base + t * gates_per_timestep : nullptr;

        cutlass::Status gemm_status = cutlass::Status::kSuccess;
        auto *splitk_workspace_ptr = static_cast<float *>(workspace);
        if (use_cudnn_splitk_kernel) {
            typename CutlassGemmCudnnSplitK::Arguments args(
                {batch_i, gates_i, hidden_k},
                {hidden_activations_step, input_hidden_stride_i},
                {packed_hidden_base, input_hidden_stride_i},
                {gates_ptr, gates_i},
                {splitk_output_ptr, gates_i},
                {1.0f, 0.0f},
                split_k_slices);
            gemm_status = launch_splitk_gemm_mainloop<ElementAccumulator>(
                args, splitk_workspace_ptr, stream);
        } else if (use_cudnn_like_kernel) {
            typename CutlassGemmCudnnLike::Arguments args(
                {batch_i, gates_i, hidden_k},
                {hidden_activations_step, input_hidden_stride_i},
                {packed_hidden_base, input_hidden_stride_i},
                {gates_ptr, gates_i},
                {gates_ptr, gates_i},
                {1.0f, 0.0f},
                1);
            gemm_status = gemm_cudnn_like.initialize(args, workspace, stream);
            if (gemm_status == cutlass::Status::kSuccess) {
                gemm_status = gemm_cudnn_like(stream);
            }
        } else if (use_small_kernel) {
            typename CutlassGemmSmall::Arguments args(
                {batch_i, gates_i, hidden_k},
                {hidden_activations_step, input_hidden_stride_i},
                {packed_hidden_base, input_hidden_stride_i},
                {gates_ptr, gates_i},
                {gates_ptr, gates_i},
                {1.0f, 0.0f},
                1);
            gemm_status = gemm_small.initialize(args, workspace, stream);
            if (gemm_status == cutlass::Status::kSuccess) {
                gemm_status = gemm_small(stream);
            }
        } else {
            typename CutlassGemmLarge::Arguments args(
                {batch_i, gates_i, hidden_k},
                {hidden_activations_step, input_hidden_stride_i},
                {packed_hidden_base, input_hidden_stride_i},
                {gates_ptr, gates_i},
                {gates_ptr, gates_i},
                {1.0f, 0.0f},
                1);
            gemm_status = gemm_large.initialize(args, workspace, stream);
            if (gemm_status == cutlass::Status::kSuccess) {
                gemm_status = gemm_large(stream);
            }
        }
        FLASHLSTM_CHECK_CUTLASS(gemm_status);

        constexpr int threads = 256;
        const int blocks = (state_i + threads - 1) / threads;
        const float *gates_src =
                use_cudnn_splitk_kernel ? splitk_output_ptr : gates_ptr;
        const int slice_stride = static_cast<int>(gates_per_timestep);
        const int splitk_for_pointwise =
                use_cudnn_splitk_kernel ? split_k_slices : 1;
        flashlstm::kernels::lstm_pointwise_kernel<<<blocks, threads, 0, stream>>>(
            gates_src,
            splitk_for_pointwise,
            slice_stride,
            input_gates_step,
            bias_ptr,
            c_prev_ptr,
            c_next_ptr,
            h_next_ptr,
            activations_step,
            input_i,
            input_hidden_stride_i,
            next_hidden_tail,
            output + t * state_elems,
            batch_i,
            hidden_i,
            gates_ptr);
        cudaError_t pointwise_err = cudaGetLastError();
        if (pointwise_err != cudaSuccess) {
            return pointwise_err;
        }

        std::swap(h_prev_ptr, h_next_ptr);
        std::swap(c_prev_ptr, c_next_ptr);
        return 0;
    };

    NVTX_SCOPED_RANGE("FlashLSTM::lstm_forward::compute");
    std::size_t next_chunk_start = 0;
    if (chunk_timesteps > 0 && seq_len > 0) {
        cutlass::Status proj_status =
                launch_input_projection(0, projection_workspace);
        FLASHLSTM_CHECK_CUTLASS(proj_status);
        next_chunk_start = static_cast<std::size_t>(chunk_timesteps);
    }
    for (std::size_t t = 0; t < seq_len; ++t) {
        if (chunk_timesteps > 0 && next_chunk_start < seq_len && t == next_chunk_start) {
            cutlass::Status proj_status =
                    launch_input_projection(next_chunk_start, projection_workspace);
            FLASHLSTM_CHECK_CUTLASS(proj_status);
            next_chunk_start += static_cast<std::size_t>(chunk_timesteps);
        }
        ERR_PROPAGATE(run_time_step(t));
    }

    FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(hn,
        h_prev_ptr,
        state_elems * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));
    FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(cn,
        c_prev_ptr,
        state_elems * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));

    return 0;
}

template<typename ElementAccumulator>
int build_lstm_graph_exec(const float *x,
                          const float *b_ih,
                          const float *b_hh,
                          const float *h0,
                          float *output,
                          float *hn,
                          float *cn,
                          std::size_t seq_len,
                          std::size_t batch,
                          std::size_t input_size,
                          std::size_t hidden_size,
                          LstmBuffersImpl *impl) {
    cudaGraph_t graph = nullptr;
    cudaError_t create_status = cudaGraphCreate(&graph, 0);
    if (create_status != cudaSuccess) {
        return create_status;
    }

    auto cleanup_graph = [&](cudaGraph_t g) {
        if (g != nullptr) {
            cudaGraphDestroy(g);
        }
    };

    impl->graph_activations_memset = {};
    impl->graph_populate_node = {};
    impl->graph_combine_bias_node = {};
    impl->graph_input_projection_nodes.clear();
    impl->graph_hidden_gemm_nodes.clear();
    impl->graph_pointwise_nodes.clear();
    impl->graph_hn_memcpy_node = {};
    impl->graph_cn_memcpy_node = {};

    int capacity_status = ensure_sequence_capacity(seq_len, batch, impl);
    if (capacity_status != 0) {
        cleanup_graph(graph);
        return capacity_status;
    }

    GemmPlan plan{};
    int input_proj_chunk_timesteps = 0;
    int prepare_status = prepare_lstm_execution<ElementAccumulator>(
        seq_len,
        batch,
        input_size,
        hidden_size,
        impl,
        &plan,
        &input_proj_chunk_timesteps);
    if (prepare_status != 0) {
        cleanup_graph(graph);
        return prepare_status;
    }

    const std::size_t activations_stride = impl->input_hidden_stride;
    const std::size_t bias_elems = 4 * hidden_size;
    const std::size_t state_elems = batch * hidden_size;
    const std::size_t gates_per_timestep = state_elems * 4;

    const int batch_i = static_cast<int>(batch);
    const int input_i = static_cast<int>(input_size);
    const int hidden_i = static_cast<int>(hidden_size);
    const int input_hidden_stride_i = static_cast<int>(activations_stride);
    const int hidden_k = input_hidden_stride_i - input_i;
    const int gates_i = static_cast<int>(4 * hidden_size);
    const int state_i = static_cast<int>(state_elems);

    ElementInput *activations_base = impl->activations;
    ElementInput *packed_weights = impl->packed_weights;
    ElementInput *packed_hidden_base_local =
            packed_weights ? packed_weights + input_i : nullptr;
    float *gates_ptr = impl->gates;
    float *bias_ptr = impl->bias;
    float *input_gates_base = impl->input_gates;
    float *h_prev_base = impl->h_prev;
    float *h_next_base = impl->h_next;
    float *c_prev_base = impl->c_prev;
    float *c_next_base = impl->c_next;
    void *workspace = impl->workspace;
    float *splitk_workspace_ptr = static_cast<float *>(workspace);
    float *splitk_output_ptr = impl->splitk_output;
    void *projection_workspace = impl->input_workspace;

    if (plan.use_splitk && splitk_output_ptr == nullptr) {
        cleanup_graph(graph);
        return cudaErrorMemoryAllocation;
    }

    // Memset activations buffer if needed
    if (impl->activations_elements > 0) {
        auto &memset_storage = impl->graph_activations_memset;
        memset_storage.enabled = true;
        memset_storage.params.dst = impl->activations;
        memset_storage.params.value = 0;
        memset_storage.params.pitch = 0;
        memset_storage.params.elementSize = 1;
        memset_storage.params.height = 1;
        memset_storage.params.width = impl->activations_elements * sizeof(ElementInput);
        cudaError_t memset_status = cudaGraphAddMemsetNode(
            &memset_storage.node,
            graph,
            nullptr,
            0,
            &memset_storage.params);
        if (memset_status != cudaSuccess) {
            cleanup_graph(graph);
            return memset_status;
        }
    }

    // populate activations kernel
    auto &populate = impl->graph_populate_node;
    populate.x = x;
    populate.h0 = h0;
    populate.activations = impl->activations;
    populate.seq_len = seq_len;
    populate.batch = batch;
    populate.input_size = input_size;
    populate.hidden_size = hidden_size;
    populate.activations_stride = activations_stride;
    const std::size_t total_activations = seq_len * batch * input_size +
                                          ((h0 != nullptr) ? batch * hidden_size : 0);
    cudaGraphNode_t populate_deps[1];
    int populate_dep_count = 0;
    if (impl->graph_activations_memset.enabled) {
        populate_deps[populate_dep_count++] = impl->graph_activations_memset.node;
    }
    if (total_activations > 0) {
        constexpr int threads = 256;
        dim3 block(threads, 1, 1);
        dim3 grid = make_grid_dim(total_activations, threads);
        populate.params.func = reinterpret_cast<void *>(flashlstm::kernels::populate_activations_kernel);
        populate.params.blockDim = block;
        populate.params.gridDim = grid;
        populate.params.sharedMemBytes = 0;
        populate.args[0] = const_cast<float **>(&populate.x);
        populate.args[1] = const_cast<float **>(&populate.h0);
        populate.args[2] = &populate.activations;
        populate.args[3] = &populate.seq_len;
        populate.args[4] = &populate.batch;
        populate.args[5] = &populate.input_size;
        populate.args[6] = &populate.hidden_size;
        populate.args[7] = &populate.activations_stride;
        populate.params.kernelParams = populate.args;
        populate.params.extra = nullptr;
        cudaError_t populate_status = cudaGraphAddKernelNode(
            &populate.node,
            graph,
            populate_dep_count > 0 ? populate_deps : nullptr,
            populate_dep_count,
            &populate.params);
        if (populate_status != cudaSuccess) {
            cleanup_graph(graph);
            return populate_status;
        }
    }

    // combine bias kernel
    auto &combine = impl->graph_combine_bias_node;
    combine.b_ih = b_ih;
    combine.b_hh = b_hh;
    combine.bias_out = impl->bias;
    combine.elements = bias_elems;
    cudaGraphNode_t combine_deps[1];
    int combine_dep_count = 0;
    if (total_activations > 0) {
        combine_deps[combine_dep_count++] = populate.node;
    } else if (impl->graph_activations_memset.enabled) {
        combine_deps[combine_dep_count++] = impl->graph_activations_memset.node;
    }
    if (combine.elements > 0) {
        constexpr int threads = 256;
        dim3 block(threads, 1, 1);
        dim3 grid = make_grid_dim(combine.elements, threads);
        combine.params.func = reinterpret_cast<void *>(flashlstm::kernels::combine_bias_kernel);
        combine.params.blockDim = block;
        combine.params.gridDim = grid;
        combine.params.sharedMemBytes = 0;
        combine.args[0] = const_cast<float **>(&combine.b_ih);
        combine.args[1] = const_cast<float **>(&combine.b_hh);
        combine.args[2] = &combine.bias_out;
        combine.args[3] = &combine.elements;
        combine.params.kernelParams = combine.args;
        combine.params.extra = nullptr;
        cudaError_t combine_status = cudaGraphAddKernelNode(
            &combine.node,
            graph,
            combine_dep_count > 0 ? combine_deps : nullptr,
            combine_dep_count,
            &combine.params);
        if (combine_status != cudaSuccess) {
            cleanup_graph(graph);
            return combine_status;
        }
    }

    // Input projection chunks
    std::size_t chunk_timesteps = static_cast<std::size_t>(input_proj_chunk_timesteps > 0 ? input_proj_chunk_timesteps : 0);
    std::size_t chunk_count = (chunk_timesteps == 0)
                                  ? 0
                                  : (seq_len + chunk_timesteps - 1) / chunk_timesteps;
    impl->graph_input_projection_nodes.resize(chunk_count);
    std::vector<cudaGraphNode_t> chunk_nodes(chunk_count, nullptr);

    using CutlassGemmLarge = CutlassGemmLargeT<ElementAccumulator>;

    for (std::size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        std::size_t chunk_start = chunk_idx * chunk_timesteps;
        const std::size_t remaining = seq_len - chunk_start;
        const int current_chunk_timesteps =
                static_cast<int>(std::min<std::size_t>(chunk_timesteps, remaining));
        const int current_rows = current_chunk_timesteps * batch_i;
        ElementInput *activations_chunk =
                activations_base
                    ? activations_base + chunk_start * static_cast<std::size_t>(batch_i) * input_hidden_stride_i
                    : nullptr;
        float *input_gates_chunk =
                input_gates_base ? input_gates_base + chunk_start * gates_per_timestep : nullptr;

        typename CutlassGemmLarge::Arguments projection_args(
            {current_rows, gates_i, input_i},
            {activations_chunk, input_hidden_stride_i},
            {packed_weights, input_hidden_stride_i},
            {input_gates_chunk, gates_i},
            {input_gates_chunk, gates_i},
            {1.0f, 0.0f},
            1);

        auto &storage = impl->graph_input_projection_nodes[chunk_idx];
        cutlass::Status status =
                prepare_cutlass_gemm_kernel_node<CutlassGemmLarge>(projection_args,
                                                                   projection_workspace,
                                                                   storage);
        if (status != cutlass::Status::kSuccess) {
            cleanup_graph(graph);
            return static_cast<int>(cudaErrorUnknown);
        }

        cudaGraphNode_t deps[2];
        int dep_count = 0;
        if (total_activations > 0) {
            deps[dep_count++] = populate.node;
        } else if (impl->graph_activations_memset.enabled) {
            deps[dep_count++] = impl->graph_activations_memset.node;
        }
        if (chunk_idx > 0) {
            deps[dep_count++] = chunk_nodes[chunk_idx - 1];
        }

        cudaError_t add_status = cudaGraphAddKernelNode(
            &storage.node,
            graph,
            dep_count > 0 ? deps : nullptr,
            dep_count,
            &storage.params);
        if (add_status != cudaSuccess) {
            cleanup_graph(graph);
            return add_status;
        }
        chunk_nodes[chunk_idx] = storage.node;
    }

    // Hidden GEMM nodes
    impl->graph_hidden_gemm_nodes.resize(seq_len);
    impl->graph_pointwise_nodes.resize(seq_len);
    std::vector<cudaGraphNode_t> hidden_nodes(seq_len, nullptr);
    std::vector<cudaGraphNode_t> pointwise_nodes(seq_len, nullptr);

    const bool use_cudnn_splitk_kernel = plan.use_splitk;
    const bool use_cudnn_like_kernel = plan.use_cudnn_like;
    const bool use_small_kernel = plan.use_small;
    const int split_k_slices = plan.split_k_slices;

    cudaGraphNode_t last_pointwise_node = nullptr;

    for (std::size_t t = 0; t < seq_len; ++t) {
        ElementInput *activations_step =
                activations_base + t * batch_i * input_hidden_stride_i;
        ElementInput *hidden_activations_step = activations_step + input_i;
        ElementInput *next_hidden_tail =
                (t + 1 < seq_len)
                    ? activations_base + (t + 1) * batch_i * input_hidden_stride_i + input_i
                    : nullptr;
        const float *input_gates_step =
                input_gates_base ? input_gates_base + t * gates_per_timestep : nullptr;

        auto &gemm_storage = impl->graph_hidden_gemm_nodes[t];

        cudaGraphNode_t gemm_deps[3];
        int gemm_dep_count = 0;
        if (chunk_count > 0) {
            std::size_t chunk_idx = input_proj_chunk_timesteps > 0
                                        ? (t / input_proj_chunk_timesteps)
                                        : 0;
            gemm_deps[gemm_dep_count++] = chunk_nodes[chunk_idx];
        } else if (total_activations > 0) {
            gemm_deps[gemm_dep_count++] = populate.node;
        }
        if (last_pointwise_node != nullptr) {
            gemm_deps[gemm_dep_count++] = last_pointwise_node;
        }

        cutlass::Status gemm_status = cutlass::Status::kSuccess;
        if (use_cudnn_splitk_kernel) {
            using SplitKGemm = CutlassGemmCudnnSplitKT<ElementAccumulator>;
            typename SplitKGemm::Arguments args(
                {batch_i, gates_i, hidden_k},
                {hidden_activations_step, input_hidden_stride_i},
                {packed_hidden_base_local, input_hidden_stride_i},
                {gates_ptr, gates_i},
                {splitk_output_ptr, gates_i},
                {1.0f, 0.0f},
                split_k_slices);
            gemm_status = prepare_splitk_gemm_kernel_node<SplitKGemm>(
                args,
                splitk_workspace_ptr,
                gemm_storage);
        } else if (use_cudnn_like_kernel) {
            using Gemm = CutlassGemmCudnnLikeT<ElementAccumulator>;
            typename Gemm::Arguments args(
                {batch_i, gates_i, hidden_k},
                {hidden_activations_step, input_hidden_stride_i},
                {packed_hidden_base_local, input_hidden_stride_i},
                {gates_ptr, gates_i},
                {gates_ptr, gates_i},
                {1.0f, 0.0f},
                1);
            gemm_status = prepare_cutlass_gemm_kernel_node<Gemm>(
                args,
                workspace,
                gemm_storage);
        } else if (use_small_kernel) {
            using Gemm = CutlassGemmSmallT<ElementAccumulator>;
            typename Gemm::Arguments args(
                {batch_i, gates_i, hidden_k},
                {hidden_activations_step, input_hidden_stride_i},
                {packed_hidden_base_local, input_hidden_stride_i},
                {gates_ptr, gates_i},
                {gates_ptr, gates_i},
                {1.0f, 0.0f},
                1);
            gemm_status = prepare_cutlass_gemm_kernel_node<Gemm>(
                args,
                workspace,
                gemm_storage);
        } else {
            using Gemm = CutlassGemmLargeT<ElementAccumulator>;
            typename Gemm::Arguments args(
                {batch_i, gates_i, hidden_k},
                {hidden_activations_step, input_hidden_stride_i},
                {packed_hidden_base_local, input_hidden_stride_i},
                {gates_ptr, gates_i},
                {gates_ptr, gates_i},
                {1.0f, 0.0f},
                1);
            gemm_status = prepare_cutlass_gemm_kernel_node<Gemm>(
                args,
                workspace,
                gemm_storage);
        }

        if (gemm_status != cutlass::Status::kSuccess) {
            cleanup_graph(graph);
            return static_cast<int>(cudaErrorUnknown);
        }

        cudaError_t gemm_add_status = cudaGraphAddKernelNode(
            &gemm_storage.node,
            graph,
            gemm_dep_count > 0 ? gemm_deps : nullptr,
            gemm_dep_count,
            &gemm_storage.params);
        if (gemm_add_status != cudaSuccess) {
            cleanup_graph(graph);
            return gemm_add_status;
        }
        hidden_nodes[t] = gemm_storage.node;

        // Pointwise node
        auto &pointwise_storage = impl->graph_pointwise_nodes[t];
        const bool use_splitk_workspace = use_cudnn_splitk_kernel;
        pointwise_storage.gates = use_splitk_workspace ? splitk_output_ptr : gates_ptr;
        pointwise_storage.split_k_slices = use_splitk_workspace ? split_k_slices : 1;
        pointwise_storage.slice_stride = static_cast<int>(gates_per_timestep);
        pointwise_storage.input_gates = input_gates_step;
        pointwise_storage.bias = bias_ptr;
        pointwise_storage.c_prev = (t % 2 == 0) ? c_prev_base : c_next_base;
        pointwise_storage.c_next = (t % 2 == 0) ? c_next_base : c_prev_base;
        pointwise_storage.h_next = (t % 2 == 0) ? h_next_base : h_prev_base;
        pointwise_storage.activations = activations_step;
        pointwise_storage.input_size = input_i;
        pointwise_storage.activations_stride = input_hidden_stride_i;
        pointwise_storage.next_hidden_tail = next_hidden_tail;
        pointwise_storage.output_t = output ? (output + t * state_elems) : nullptr;
        pointwise_storage.batch_size = batch_i;
        pointwise_storage.hidden_size = hidden_i;
        pointwise_storage.gates_out = gates_ptr;

        pointwise_storage.params.func = reinterpret_cast<void *>(flashlstm::kernels::lstm_pointwise_kernel);
        pointwise_storage.params.sharedMemBytes = 0;
        constexpr int pointwise_threads = 256;
        const int blocks = (state_i + pointwise_threads - 1) / pointwise_threads;
        pointwise_storage.params.blockDim = dim3(pointwise_threads, 1, 1);
        pointwise_storage.params.gridDim = dim3(blocks, 1, 1);
        pointwise_storage.args[0] = const_cast<float **>(&pointwise_storage.gates);
        pointwise_storage.args[1] = &pointwise_storage.split_k_slices;
        pointwise_storage.args[2] = &pointwise_storage.slice_stride;
        pointwise_storage.args[3] = const_cast<float **>(&pointwise_storage.input_gates);
        pointwise_storage.args[4] = const_cast<float **>(&pointwise_storage.bias);
        pointwise_storage.args[5] = const_cast<float **>(&pointwise_storage.c_prev);
        pointwise_storage.args[6] = &pointwise_storage.c_next;
        pointwise_storage.args[7] = &pointwise_storage.h_next;
        pointwise_storage.args[8] = &pointwise_storage.activations;
        pointwise_storage.args[9] = &pointwise_storage.input_size;
        pointwise_storage.args[10] = &pointwise_storage.activations_stride;
        pointwise_storage.args[11] = &pointwise_storage.next_hidden_tail;
        pointwise_storage.args[12] = &pointwise_storage.output_t;
        pointwise_storage.args[13] = &pointwise_storage.batch_size;
        pointwise_storage.args[14] = &pointwise_storage.hidden_size;
        pointwise_storage.args[15] = &pointwise_storage.gates_out;
        pointwise_storage.params.kernelParams = pointwise_storage.args;
        pointwise_storage.params.extra = nullptr;

        cudaGraphNode_t pointwise_deps[3];
        int pointwise_dep_count = 0;
        pointwise_deps[pointwise_dep_count++] = gemm_storage.node;
        if (combine.elements > 0) {
            pointwise_deps[pointwise_dep_count++] = combine.node;
        }

        cudaError_t pointwise_status = cudaGraphAddKernelNode(
            &pointwise_storage.node,
            graph,
            pointwise_deps,
            pointwise_dep_count,
            &pointwise_storage.params);
        if (pointwise_status != cudaSuccess) {
            cleanup_graph(graph);
            return pointwise_status;
        }

        pointwise_nodes[t] = pointwise_storage.node;
        last_pointwise_node = pointwise_storage.node;
    }

    // Final copies for hn and cn
    if (state_elems > 0) {
        float *final_h_source = (seq_len % 2 == 0) ? h_prev_base : h_next_base;
        float *final_c_source = (seq_len % 2 == 0) ? c_prev_base : c_next_base;
        cudaGraphNode_t tail_dependency = nullptr;
        if (seq_len > 0) {
            tail_dependency = last_pointwise_node;
        } else if (combine.elements > 0) {
            tail_dependency = combine.node;
        } else if (total_activations > 0) {
            tail_dependency = populate.node;
        } else if (impl->graph_activations_memset.enabled) {
            tail_dependency = impl->graph_activations_memset.node;
        }

        auto prepare_memcpy_node = [&](float *src, float *dst, LstmBuffersImpl::GraphMemcpyNodeStorage &storage) -> int {
            storage.enabled = true;
            storage.params = {};
            storage.params.kind = cudaMemcpyDeviceToDevice;
            storage.params.srcPtr = make_cudaPitchedPtr(src,
                                                        state_elems * sizeof(float),
                                                        state_elems * sizeof(float),
                                                        1);
            storage.params.dstPtr = make_cudaPitchedPtr(dst,
                                                        state_elems * sizeof(float),
                                                        state_elems * sizeof(float),
                                                        1);
            storage.params.extent = make_cudaExtent(state_elems * sizeof(float), 1, 1);
            cudaGraphNode_t dep = tail_dependency;
            cudaError_t status = cudaGraphAddMemcpyNode(
                &storage.node,
                graph,
                dep ? &dep : nullptr,
                dep ? 1 : 0,
                &storage.params);
            return static_cast<int>(status);
        };

        if (hn != nullptr) {
            int status = prepare_memcpy_node(final_h_source, hn, impl->graph_hn_memcpy_node);
            if (status != 0) {
                cleanup_graph(graph);
                return status;
            }
        }
        if (cn != nullptr) {
            int status = prepare_memcpy_node(final_c_source, cn, impl->graph_cn_memcpy_node);
            if (status != 0) {
                cleanup_graph(graph);
                return status;
            }
        }
    }

    cudaGraphExec_t exec = nullptr;
    cudaError_t instantiate_status = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (instantiate_status != cudaSuccess) {
        cleanup_graph(graph);
        return instantiate_status;
    }

    if (impl->graph_exec != nullptr) {
        cudaGraphExecDestroy(impl->graph_exec);
        impl->graph_exec = nullptr;
    }
    if (impl->graph != nullptr) {
        cudaGraphDestroy(impl->graph);
        impl->graph = nullptr;
    }

    impl->graph = graph;
    impl->graph_exec = exec;
    impl->graph_seq_len = seq_len;
    impl->graph_plan = plan;
    impl->graph_plan_valid = true;
    return 0;
}

extern "C" int lstm_forward(const float *x,
                            const float *b_ih,
                            const float *b_hh,
                            const float *h0,
                            const float *c0,
                            float *output,
                            float *hn,
                            float *cn,
                            const std::size_t seq_len,
                            const std::size_t batch,
                            const std::size_t input_size,
                            const std::size_t hidden_size,
                            const lstm_buffers *buffers,
                            const lstm_compute_precision_t precision) {
    NVTX_SCOPED_RANGE("FlashLSTM::lstm_forward");
    FLASHLSTM_VALIDATE(buffers != nullptr && buffers->impl != nullptr, cudaErrorInvalidValue);

    auto *impl = static_cast<LstmBuffersImpl *>(buffers->impl);
    FLASHLSTM_VALIDATE(impl->precision == precision, cudaErrorInvalidValue);
    FLASHLSTM_VALIDATE(impl->batch == batch && impl->input_size == input_size && impl->hidden_size == hidden_size,
                       cudaErrorInvalidValue);
    FLASHLSTM_VALIDATE(impl->weights_packed && impl->weight_elements > 0, cudaErrorInvalidValue);
    cudaStream_t stream = impl->stream;
    FLASHLSTM_VALIDATE(stream != nullptr, cudaErrorInvalidValue);

    const bool use_graph =
            (impl->execution_mode == LSTM_EXECUTION_MODE_GRAPH) ||
            g_flashlstm_force_graph_mode;
    const std::size_t state_elems = batch * hidden_size;
    const std::size_t x_elements = seq_len * batch * input_size;
    const std::size_t output_elems = seq_len * batch * hidden_size;
    const std::size_t bias_elems = impl->bias_elements;

    if (state_elems > 0) {
        FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(impl->h_prev,
            h0,
            state_elems * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));
        FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(impl->c_prev,
            c0,
            state_elems * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));
    }

    const float *x_compute = x;
    const float *b_ih_compute = b_ih;
    const float *b_hh_compute = b_hh;
    float *output_compute = output;
    float *hn_compute = hn;
    float *cn_compute = cn;

    if (use_graph) {
        if (x_elements > impl->x_staging_elements) {
            invalidate_graph(impl);
            free_if_needed(impl->x_staging);
            impl->x_staging = nullptr;
            if (x_elements > 0) {
                cudaError_t alloc_status =
                        cudaMalloc(&impl->x_staging, x_elements * sizeof(float));
                if (alloc_status != cudaSuccess) {
                    impl->x_staging_elements = 0;
                    return alloc_status;
                }
            }
            impl->x_staging_elements = x_elements;
        }
        if (x_elements > 0) {
            FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(impl->x_staging,
                x,
                x_elements * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
        }
        x_compute = impl->x_staging;

        if (bias_elems > 0) {
            if (impl->b_ih_staging == nullptr) {
                cudaError_t alloc_status =
                        cudaMalloc(&impl->b_ih_staging, bias_elems * sizeof(float));
                if (alloc_status != cudaSuccess) {
                    return alloc_status;
                }
            }
            if (impl->b_hh_staging == nullptr) {
                cudaError_t alloc_status =
                        cudaMalloc(&impl->b_hh_staging, bias_elems * sizeof(float));
                if (alloc_status != cudaSuccess) {
                    return alloc_status;
                }
            }
            FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(impl->b_ih_staging,
                b_ih,
                bias_elems * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
            FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(impl->b_hh_staging,
                b_hh,
                bias_elems * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
            b_ih_compute = impl->b_ih_staging;
            b_hh_compute = impl->b_hh_staging;
        }

        if (output_elems > impl->output_buffer_elements) {
            invalidate_graph(impl);
            free_if_needed(impl->output_buffer);
            impl->output_buffer = nullptr;
            if (output_elems > 0) {
                cudaError_t alloc_status =
                        cudaMalloc(&impl->output_buffer, output_elems * sizeof(float));
                if (alloc_status != cudaSuccess) {
                    impl->output_buffer_elements = 0;
                    return alloc_status;
                }
            }
            impl->output_buffer_elements = output_elems;
        }
        if (output_elems > 0) {
            output_compute = impl->output_buffer;
        } else {
            output_compute = impl->output_buffer;
        }

        if (state_elems > 0) {
            if (impl->hn_buffer == nullptr) {
                cudaError_t alloc_status =
                        cudaMalloc(&impl->hn_buffer, state_elems * sizeof(float));
                if (alloc_status != cudaSuccess) {
                    return alloc_status;
                }
            }
            if (impl->cn_buffer == nullptr) {
                cudaError_t alloc_status =
                        cudaMalloc(&impl->cn_buffer, state_elems * sizeof(float));
                if (alloc_status != cudaSuccess) {
                    return alloc_status;
                }
            }
            hn_compute = impl->hn_buffer;
            cn_compute = impl->cn_buffer;
        }
    }

    const float *h0_compute = impl->h_prev;
    const float *c0_compute = impl->c_prev;

    auto run_forward = [&]() -> int {
        if (impl->use_fp16_accumulator) {
            return lstm_forward_impl<cutlass::half_t>(x_compute,
                                                      b_ih_compute,
                                                      b_hh_compute,
                                                      h0_compute,
                                                      c0_compute,
                                                      output_compute,
                                                      hn_compute,
                                                      cn_compute,
                                                      seq_len,
                                                      batch,
                                                      input_size,
                                                      hidden_size,
                                                      impl);
        }
        return lstm_forward_impl<float>(x_compute,
                                        b_ih_compute,
                                        b_hh_compute,
                                        h0_compute,
                                        c0_compute,
                                        output_compute,
                                        hn_compute,
                                        cn_compute,
                                        seq_len,
                                        batch,
                                        input_size,
                                        hidden_size,
                                        impl);
    };

    if (!use_graph) {
        if (const int status = run_forward(); status != 0) {
            return status;
        }
        FLASHLSTM_CHECK_CUDA(cudaStreamSynchronize(stream));
        return 0;
    }

    const bool need_rebuild = (impl->graph_exec == nullptr) ||
                              (impl->graph_seq_len != seq_len) ||
                              !impl->graph_plan_valid;
    if (need_rebuild) {
        int build_status = 0;
        if (impl->use_fp16_accumulator) {
            build_status = build_lstm_graph_exec<cutlass::half_t>(x_compute,
                                                                  b_ih_compute,
                                                                  b_hh_compute,
                                                                  h0_compute,
                                                                  output_compute,
                                                                  hn_compute,
                                                                  cn_compute,
                                                                  seq_len,
                                                                  batch,
                                                                  input_size,
                                                                  hidden_size,
                                                                  impl);
        } else {
            build_status = build_lstm_graph_exec<float>(x_compute,
                                                        b_ih_compute,
                                                        b_hh_compute,
                                                        h0_compute,
                                                        output_compute,
                                                        hn_compute,
                                                        cn_compute,
                                                        seq_len,
                                                        batch,
                                                        input_size,
                                                        hidden_size,
                                                        impl);
        }
        if (build_status != 0) {
            return build_status;
        }
    }

    FLASHLSTM_CHECK_CUDA(cudaGraphLaunch(impl->graph_exec, stream));

    if (output_compute != output && output_elems > 0) {
        FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(output,
            output_compute,
            output_elems * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));
    }
    if (hn_compute != hn && state_elems > 0) {
        FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(hn,
            hn_compute,
            state_elems * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));
    }
    if (cn_compute != cn && state_elems > 0) {
        FLASHLSTM_CHECK_CUDA(cudaMemcpyAsync(cn,
            cn_compute,
            state_elems * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));
    }

    FLASHLSTM_CHECK_CUDA(cudaStreamSynchronize(stream));
    return 0;
}
