#include "gemm.h"
#include "gputx.h"
#include "lstm.hpp"
#include "mfu_profiler.hpp"
#include "numeric_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

inline void CheckCuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

// Helper for consistent 1D launch configuration.
inline int BlocksFor(size_t count, int threads = 256) {
    return static_cast<int>((count + threads - 1) / threads);
}

inline const char *GateCacheDTypeName(flstm::GateCacheDType dtype) {
    switch (dtype) {
        case flstm::GateCacheDType::kFloat32: return "float32";
        case flstm::GateCacheDType::kFloat16: return "float16";
    }
    return "unknown";
}

inline size_t GateCacheDTypeSize(flstm::GateCacheDType dtype) {
    switch (dtype) {
        case flstm::GateCacheDType::kFloat32: return sizeof(float);
        case flstm::GateCacheDType::kFloat16: return sizeof(__half);
    }
    throw std::runtime_error("Unsupported gate cache dtype");
}

inline void ValidateGateCacheOptions(const flstm::StreamingLstmOptions &options) {
    auto validate = [](flstm::GateCacheDType dtype, const char *label) {
        switch (dtype) {
            case flstm::GateCacheDType::kFloat32:
            case flstm::GateCacheDType::kFloat16:
                return;
            default:
                throw std::runtime_error(std::string("Unsupported gate cache dtype for ") + label
                                         + ": " + GateCacheDTypeName(dtype));
        }
    };
    validate(options.h_dtype, "h");
    validate(options.c_dtype, "c");
}

bool RegisterHostMemoryOrThrow(const void *ptr, size_t bytes, const char *what) {
    if (ptr == nullptr || bytes == 0) {
        return false;
    }
    const cudaError_t status = cudaHostRegister(const_cast<void *>(ptr), bytes, cudaHostRegisterDefault);
    if (status == cudaSuccess) {
        return true;
    }
    if (status == cudaErrorHostMemoryAlreadyRegistered) {
        cudaGetLastError();
        return false;
    }
    if (status == cudaErrorInvalidValue) {
        cudaPointerAttributes attrs{};
#if CUDART_VERSION >= 10000
        const cudaError_t attr_status = cudaPointerGetAttributes(&attrs, ptr);
        if (attr_status == cudaSuccess && attrs.type == cudaMemoryTypeHost) {
#else
        const cudaError_t attr_status = cudaPointerGetAttributes(&attrs, ptr);
        if (attr_status == cudaSuccess && attrs.memoryType == cudaMemoryTypeHost) {
#endif
            cudaGetLastError();
            return false;
        }
        if (attr_status != cudaSuccess) {
            throw std::runtime_error(std::string(what) + " (cudaPointerGetAttributes failed): "
                                     + cudaGetErrorString(attr_status));
        }
    }
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template<typename T>
struct DeviceBuffer {
    T *ptr{nullptr};
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

// Utility to allocate device memory while keeping the RAII wrapper simple.
template<typename T>
void AllocateDeviceBuffer(DeviceBuffer<T> &buffer, size_t elements, const char *what) {
    if (elements == 0) {
        buffer.ptr = nullptr;
        return;
    }
    CheckCuda(cudaMalloc(&buffer.ptr, elements * sizeof(T)), what);
}

template<typename T, size_t N>
void AllocateDeviceBufferArray(DeviceBuffer<T> (&buffers)[N], size_t elements, const char *what_prefix) {
    for (size_t i = 0; i < N; ++i) {
        std::string label = std::string(what_prefix) + "[" + std::to_string(i) + "]";
        AllocateDeviceBuffer(buffers[i], elements, label.c_str());
    }
}

// Async zero fill that honours optional buffers and element counts.
template<typename T>
void ZeroDeviceMemory(T *ptr, size_t elements, cudaStream_t stream, const char *what) {
    if (elements == 0 || ptr == nullptr) {
        return;
    }
    CheckCuda(cudaMemsetAsync(ptr, 0, elements * sizeof(T), stream), what);
}

struct CudaEvent {
    cudaEvent_t evt{nullptr};
    CudaEvent() {
        CheckCuda(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    }
    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;
    ~CudaEvent() {
        if (evt != nullptr) {
            cudaEventDestroy(evt);
            evt = nullptr;
        }
    }
};

struct HostRegistration {
    const void *ptr{nullptr};
    void reset(const void *p, size_t bytes, const char *what) {
        if (ptr != nullptr) {
            cudaHostUnregister(const_cast<void *>(ptr));
            ptr = nullptr;
        }
        const bool registered = RegisterHostMemoryOrThrow(p, bytes, what);
        if (registered) {
            ptr = p;
        }
    }
    ~HostRegistration() {
        if (ptr != nullptr) {
            cudaHostUnregister(const_cast<void *>(ptr));
        }
    }
};

__global__ void HalfToFloatKernel(const __half *src, float *dst, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    dst[idx] = __half2float(src[idx]);
}

__global__ void FloatToHalfKernel(const float *src, __half *dst, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    dst[idx] = __float2half(src[idx]);
}

__global__ void FuseWeightsKernel(
    const float *weight_ih,  // (4H, I) row-major
    const float *weight_hh,  // (4H, H) row-major
    __half *weight_cat,      // (I+H, 4H) column-major
    size_t input_size,
    size_t hidden_size
) {
    const size_t gate_dim = 4 * hidden_size;
    const size_t rows = input_size + hidden_size;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * gate_dim) {
        return;
    }
    const size_t col = idx / rows;
    const size_t row = idx % rows;
    float value;
    if (row < input_size) {
        value = weight_ih[col * input_size + row];
    } else {
        const size_t h_row = row - input_size;
        value = weight_hh[col * hidden_size + h_row];
    }
    weight_cat[row + col * rows] = __float2half(value);
}

__global__ void FuseBiasKernel(
    const float *bias_ih,
    const float *bias_hh,
    float *bias_out,
    size_t hidden_size
) {
    const size_t gate_dim = 4 * hidden_size;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gate_dim) {
        return;
    }
    bias_out[idx] = bias_ih[idx] + bias_hh[idx];
}

__device__ __forceinline__ float Sigmoid(float x) {
    return flstm::numeric::StableSigmoid(x);
}

__device__ __forceinline__ float WarpReduceMax(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__global__ void ConvertInputToZCacheKernel(
    const __half *x_src,           // (chunk_steps, B, I) half
    float *z_cache_col,            // (I+H, T*B) column-major float staging
    size_t time_offset,
    size_t chunk_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size
) {
    const size_t column_count = chunk_steps * batch_size;
    const size_t column = blockIdx.x;
    if (column >= column_count) {
        return;
    }
    const size_t step_idx = column / batch_size;
    const size_t batch_idx = column % batch_size;
    const size_t global_column = (time_offset + step_idx) * batch_size + batch_idx;

    const __half *src = x_src + (step_idx * batch_size + batch_idx) * input_size;
    float *dst = z_cache_col + global_column * (input_size + hidden_size);

    for (size_t row = threadIdx.x; row < input_size; row += blockDim.x) {
        dst[row] = __half2float(src[row]);
    }
}

__global__ void SeedHiddenColumnKernel(
    const float *h0,          // (B, H) row-major
    float *z_cache_col,       // (I+H, T*B) column-major float staging
    size_t batch_size,
    size_t input_size,
    size_t hidden_size
) {
    const size_t total = batch_size * hidden_size;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const size_t batch_idx = idx / hidden_size;
    const size_t hidden_idx = idx % hidden_size;
    const size_t z_rows = input_size + hidden_size;
    const float value = h0[batch_idx * hidden_size + hidden_idx];
    z_cache_col[input_size + hidden_idx + batch_idx * z_rows] = value;
}

__global__ void ScaleAndPackColumnsKernel(
    const float *z_cols_float,   // (I+H, B) column-major float
    __half *z_cols_half,         // (I+H, B) column-major half
    float *column_scale,         // (B,) scale factors to undo quantisation
    size_t z_rows,
    size_t batch_size
) {
    const size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }
    __shared__ float warp_max_shared[32];
    __shared__ float inv_scale_shared;
    const float *src = z_cols_float + batch_idx * z_rows;
    __half *dst = z_cols_half + batch_idx * z_rows;

    float local_max = 0.0f;
    for (size_t row = threadIdx.x; row < z_rows; row += blockDim.x) {
        const float value = flstm::numeric::FiniteOrZero(src[row]);
        local_max = fmaxf(local_max, fabsf(value));
    }
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x / warpSize;
    float warp_max = WarpReduceMax(local_max);
    if (lane == 0) {
        warp_max_shared[warp_id] = warp_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        float val = (lane < warp_count) ? warp_max_shared[lane] : 0.0f;
        float reduced = WarpReduceMax(val);
        if (lane == 0) {
            float inv_scale = 1.0f;
            float column_scale_value = 1.0f;
            if (reduced > 0.0f && isfinite(reduced)) {
                float scale = reduced / flstm::numeric::kFp16SafeMax;
                if (!(scale > 0.0f) || !isfinite(scale)) {
                    scale = 1.0f;
                }
                inv_scale = 1.0f / scale;
                column_scale_value = scale;
            }
            inv_scale_shared = inv_scale;
            column_scale[batch_idx] = column_scale_value;
        }
    }
    __syncthreads();
    const float inv_scale = inv_scale_shared;

    const size_t pair_count = z_rows / 2;
    const size_t pair_stride = blockDim.x;
    for (size_t pair_idx = threadIdx.x; pair_idx < pair_count; pair_idx += pair_stride) {
        const size_t offset = pair_idx * 2;
        const float value0 = flstm::numeric::FiniteOrZero(src[offset]);
        const float value1 = flstm::numeric::FiniteOrZero(src[offset + 1]);
        const float scaled0 = flstm::numeric::ClampToHalfRange(value0 * inv_scale);
        const float scaled1 = flstm::numeric::ClampToHalfRange(value1 * inv_scale);
        dst[offset] = __float2half(scaled0);
        dst[offset + 1] = __float2half(scaled1);
    }

    if ((z_rows & 1u) && threadIdx.x == 0) {
        const float tail_value = flstm::numeric::FiniteOrZero(src[z_rows - 1]);
        const float tail = flstm::numeric::ClampToHalfRange(tail_value * inv_scale);
        dst[z_rows - 1] = __float2half(tail);
    }
}

struct PointwiseContext {
    float *c_prev;
    float *h_prev;
    float *h_next;
    float *c_next;
    __half *y_half_out;
    __half *gate_cache_step;
    __half *h_cache;
    __half *c_cache;
    float *z_cache_col;
    float *checkpoint_dst_h;
    float *checkpoint_dst_c;
};

__device__ __forceinline__ void StorePointwiseOutputs(
    size_t state_index,
    size_t batch_idx,
    size_t hidden_idx,
    size_t gate_dim,
    size_t hidden_size,
    size_t batch_size,
    size_t cache_index,
    int has_next_column,
    size_t next_column_offset,
    size_t input_size,
    size_t z_rows,
    const PointwiseContext &ctx,
    float i_gate,
    float f_gate,
    float g_gate,
    float o_gate,
    float c_prev_val,
    float c_val,
    float h_val
) {
    if (ctx.checkpoint_dst_c != nullptr) {
        ctx.checkpoint_dst_c[state_index] = c_prev_val;
    }
    if (ctx.checkpoint_dst_h != nullptr && ctx.h_prev != nullptr) {
        ctx.checkpoint_dst_h[state_index] = ctx.h_prev[state_index];
    }
    const float h_storable = flstm::numeric::ClampToHalfRange(h_val);
    const float c_storable = flstm::numeric::ClampToHalfRange(c_val);
    if (ctx.h_next != nullptr) {
        ctx.h_next[state_index] = h_val;
    }
    if (ctx.c_next != nullptr) {
        ctx.c_next[state_index] = c_val;
    }
    if (ctx.y_half_out != nullptr) {
        ctx.y_half_out[state_index] = __float2half(h_storable);
    }
    if (ctx.gate_cache_step != nullptr) {
        __half *gate_ptr = ctx.gate_cache_step + batch_idx * gate_dim;
        gate_ptr[hidden_idx + 0 * hidden_size] = __float2half(i_gate);
        gate_ptr[hidden_idx + 1 * hidden_size] = __float2half(f_gate);
        gate_ptr[hidden_idx + 2 * hidden_size] = __float2half(g_gate);
        gate_ptr[hidden_idx + 3 * hidden_size] = __float2half(o_gate);
    }
    if (ctx.h_cache != nullptr) {
        __half *dst = ctx.h_cache + cache_index * (batch_size * hidden_size);
        dst[batch_idx * hidden_size + hidden_idx] = __float2half(h_storable);
    }
    if (ctx.c_cache != nullptr) {
        __half *dst = ctx.c_cache + cache_index * (batch_size * hidden_size);
        dst[batch_idx * hidden_size + hidden_idx] = __float2half(c_storable);
    }
    if (has_next_column && ctx.z_cache_col != nullptr) {
        const size_t column = next_column_offset + batch_idx;
        ctx.z_cache_col[input_size + hidden_idx + column * z_rows] = flstm::numeric::FiniteOrZero(h_val);
    }
}

__global__ void ForwardPointwiseKernel(
    const float *gate_col,         // (4H, B) column-major
    const float *bias,             // (4H,)
    PointwiseContext ctx,
    const float *column_scale,     // (B,) scaling factors for this step
    size_t z_rows,
    size_t input_size,
    int has_next_column,
    size_t next_column_offset,
    size_t cache_index,
    size_t batch_size,
    size_t hidden_size
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = batch_size * hidden_size;
    if (idx >= total) {
        return;
    }
    const size_t batch_idx = idx / hidden_size;
    const size_t hidden_idx = idx % hidden_size;
    const size_t gate_dim = 4 * hidden_size;
    const size_t col = batch_idx;
    const size_t row_base = hidden_idx;
    const size_t state_index = batch_idx * hidden_size + hidden_idx;

    float scale = flstm::numeric::FiniteOrDefault(column_scale[col], 1.0f);
    if (scale == 0.0f) {
        scale = 1.0f;
    }
    const size_t col_offset = col * gate_dim + row_base;
    const float gi = fmaf(gate_col[col_offset + 0 * hidden_size], scale, bias[row_base + 0 * hidden_size]);
    const float gf = fmaf(gate_col[col_offset + 1 * hidden_size], scale, bias[row_base + 1 * hidden_size]);
    const float gg = fmaf(gate_col[col_offset + 2 * hidden_size], scale, bias[row_base + 2 * hidden_size]);
    const float go = fmaf(gate_col[col_offset + 3 * hidden_size], scale, bias[row_base + 3 * hidden_size]);

    const float i_gate = Sigmoid(gi);
    const float f_gate = Sigmoid(gf);
    const float g_gate = tanhf(gg);
    const float c_prev_val = ctx.c_prev[state_index];
    const float c_val = f_gate * c_prev_val + i_gate * g_gate;
    const float o_gate = Sigmoid(go);
    const float h_val = o_gate * tanhf(c_val);

    StorePointwiseOutputs(
        state_index,
        batch_idx,
        hidden_idx,
        gate_dim,
        hidden_size,
        batch_size,
        cache_index,
        has_next_column,
        next_column_offset,
        input_size,
        z_rows,
        ctx,
        i_gate,
        f_gate,
        g_gate,
        o_gate,
        c_prev_val,
        c_val,
        h_val
    );
}

__global__ void ScalePackAndPointwiseKernel(
    const float *gate_col,
    const float *bias,
    PointwiseContext ctx,
    const float *column_scale_cur,
    float *column_scale_next,
    const float *next_z_cols_float,
    __half *next_z_cols_half,
    size_t z_rows,
    size_t batch_size,
    size_t hidden_size,
    size_t input_size,
    size_t cache_index,
    int scale_next,
    int has_next_column,
    size_t next_column_offset
) {
    const size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }
    const size_t gate_dim = 4 * hidden_size;
    const size_t state_base = batch_idx * hidden_size;

    float scale = flstm::numeric::FiniteOrDefault(column_scale_cur[batch_idx], 1.0f);
    if (scale == 0.0f) {
        scale = 1.0f;
    }
    const size_t col_offset = batch_idx * gate_dim;
    for (size_t hidden_idx = threadIdx.x; hidden_idx < hidden_size; hidden_idx += blockDim.x) {
        const size_t row_base = hidden_idx;
        const float gi = fmaf(gate_col[col_offset + row_base + 0 * hidden_size], scale, bias[row_base + 0 * hidden_size]);
        const float gf = fmaf(gate_col[col_offset + row_base + 1 * hidden_size], scale, bias[row_base + 1 * hidden_size]);
        const float gg = fmaf(gate_col[col_offset + row_base + 2 * hidden_size], scale, bias[row_base + 2 * hidden_size]);
        const float go = fmaf(gate_col[col_offset + row_base + 3 * hidden_size], scale, bias[row_base + 3 * hidden_size]);
        const float i_gate = Sigmoid(gi);
        const float f_gate = Sigmoid(gf);
        const float g_gate = tanhf(gg);
        const size_t state_index = state_base + hidden_idx;
        const float c_prev_val = ctx.c_prev[state_index];
        const float c_val = f_gate * c_prev_val + i_gate * g_gate;
        const float o_gate = Sigmoid(go);
        const float h_val = o_gate * tanhf(c_val);
        StorePointwiseOutputs(
            state_index,
            batch_idx,
            hidden_idx,
            gate_dim,
            hidden_size,
            batch_size,
            cache_index,
            has_next_column,
            next_column_offset,
            input_size,
            z_rows,
            ctx,
            i_gate,
            f_gate,
            g_gate,
            o_gate,
            c_prev_val,
            c_val,
            h_val
        );
    }

    if (scale_next && next_z_cols_float != nullptr && next_z_cols_half != nullptr && column_scale_next != nullptr) {
        __syncthreads();
        __shared__ float warp_max_shared[32];
        __shared__ float inv_scale_shared;
        const float *src = next_z_cols_float + batch_idx * z_rows;
        float local_max = 0.0f;
        for (size_t row = threadIdx.x; row < z_rows; row += blockDim.x) {
            const float value = flstm::numeric::FiniteOrZero(src[row]);
            local_max = fmaxf(local_max, fabsf(value));
        }
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x / warpSize;
        float warp_max = WarpReduceMax(local_max);
        if (lane == 0) {
            warp_max_shared[warp_id] = warp_max;
        }
        __syncthreads();
        if (warp_id == 0) {
            const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
            float value = (lane < warp_count) ? warp_max_shared[lane] : 0.0f;
            float reduced = WarpReduceMax(value);
            if (lane == 0) {
                float inv_scale = 1.0f;
                float column_scale_value = 1.0f;
                if (reduced > 0.0f && isfinite(reduced)) {
                    float next_scale = reduced / flstm::numeric::kFp16SafeMax;
                    if (!(next_scale > 0.0f) || !isfinite(next_scale)) {
                        next_scale = 1.0f;
                    }
                    inv_scale = 1.0f / next_scale;
                    column_scale_value = next_scale;
                }
                inv_scale_shared = inv_scale;
                column_scale_next[batch_idx] = column_scale_value;
            }
        }
        __syncthreads();
        const float inv_scale = inv_scale_shared;
        __half *dst = next_z_cols_half + batch_idx * z_rows;
        for (size_t row = threadIdx.x; row < z_rows; row += blockDim.x) {
            const float value = flstm::numeric::FiniteOrZero(src[row]);
            const float scaled = flstm::numeric::ClampToHalfRange(value * inv_scale);
            dst[row] = __float2half(scaled);
        }
    }
}

// Aggregates references used while double-buffering host-to-device input loads.
struct ForwardChunkCopyParams {
    size_t total_chunks;
    size_t time_steps;
    size_t chunk_capacity;
    size_t batch_size;
    size_t input_size;
    const __half *x_tensor_host;
    DeviceBuffer<__half> *x_chunk_buffers;
    CudaEvent *x_ready;
    cudaStream_t h2d_stream;
    bool *compute_done_valid;
    CudaEvent *compute_done;
};

// Schedules input chunk copies, ensuring reuse waits for outstanding compute.
static size_t IssueChunkCopy(size_t chunk_index, int slot, const ForwardChunkCopyParams &params) {
    if (chunk_index >= params.total_chunks) {
        return 0;
    }
    const size_t chunk_start = chunk_index * params.chunk_capacity;
    if (chunk_start >= params.time_steps) {
        return 0;
    }
    if (params.x_chunk_buffers[slot].ptr == nullptr) {
        return 0;
    }
    if (params.compute_done_valid[slot]) {
        CheckCuda(cudaStreamWaitEvent(params.h2d_stream, params.compute_done[slot].evt, 0),
                  "wait compute_done before reuse");
        params.compute_done_valid[slot] = false;
    }
    const size_t remaining = params.time_steps - chunk_start;
    const size_t steps = std::min(params.chunk_capacity, remaining);
    if (steps == 0) {
        return 0;
    }
    const size_t bytes = steps * params.batch_size * params.input_size * sizeof(__half);
    if (bytes == 0) {
        return 0;
    }
    const __half *src = params.x_tensor_host + chunk_start * params.batch_size * params.input_size;
    CheckCuda(cudaMemcpyAsync(
                  params.x_chunk_buffers[slot].ptr,
                  src,
                  bytes,
                  cudaMemcpyHostToDevice,
                  params.h2d_stream),
              "async copy x chunk");
    CheckCuda(cudaEventRecord(params.x_ready[slot].evt, params.h2d_stream), "record x_ready");
    return steps;
}

} // namespace

namespace flstm {
namespace testing {
void LaunchScalePackPointwiseKernel(
    const float *gate_col,
    const float *bias,
    const float *c_prev,
    const float *h_prev,
    float *h_next,
    float *c_next,
    __half *y_half_out,
    __half *gate_cache_step,
    __half *h_cache,
    __half *c_cache,
    float *z_cache_col,
    const float *column_scale_cur,
    float *column_scale_next,
    const float *next_z_cols_float,
    __half *next_z_cols_half,
    float *checkpoint_dst_h,
    float *checkpoint_dst_c,
    size_t z_rows,
    size_t batch_size,
    size_t hidden_size,
    size_t input_size,
    size_t cache_index,
    int scale_next,
    int has_next_column,
    size_t next_column_offset,
    cudaStream_t stream
) {
    constexpr int kThreads = 256;
    dim3 grid(static_cast<unsigned int>(batch_size));
    PointwiseContext ctx{};
    ctx.c_prev = const_cast<float *>(c_prev);
    ctx.h_prev = const_cast<float *>(h_prev);
    ctx.h_next = h_next;
    ctx.c_next = c_next;
    ctx.y_half_out = y_half_out;
    ctx.gate_cache_step = gate_cache_step;
    ctx.h_cache = h_cache;
    ctx.c_cache = c_cache;
    ctx.z_cache_col = z_cache_col;
    ctx.checkpoint_dst_h = checkpoint_dst_h;
    ctx.checkpoint_dst_c = checkpoint_dst_c;
    ScalePackAndPointwiseKernel<<<grid, kThreads, 0, stream>>>(
        gate_col,
        bias,
        ctx,
        column_scale_cur,
        column_scale_next,
        next_z_cols_float,
        next_z_cols_half,
        z_rows,
        batch_size,
        hidden_size,
        input_size,
        cache_index,
        scale_next,
        has_next_column,
        next_column_offset
    );
    if (cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
        throw std::runtime_error(std::string("ScalePackAndPointwiseKernel: ") + cudaGetErrorString(status));
    }
}

void LaunchConvertInputToZCacheKernel(
    const __half *x_src,
    float *z_cache_col,
    size_t time_offset,
    size_t chunk_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,
    cudaStream_t stream
) {
    constexpr int kThreads = 256;
    const size_t columns = chunk_steps * batch_size;
    const dim3 grid(columns == 0 ? 1u : static_cast<unsigned int>(columns));
    ConvertInputToZCacheKernel<<<grid, kThreads, 0, stream>>>(
        x_src,
        z_cache_col,
        time_offset,
        chunk_steps,
        batch_size,
        input_size,
        hidden_size
    );
    if (cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
        throw std::runtime_error(std::string("ConvertInputToZCacheKernel: ") + cudaGetErrorString(status));
    }
}

void LaunchScaleAndPackColumnsKernel(
    const float *z_cols_float,
    __half *z_cols_half,
    float *column_scale,
    size_t z_rows,
    size_t batch_size,
    cudaStream_t stream
) {
    constexpr int kThreads = 256;
    dim3 grid(static_cast<unsigned int>(batch_size));
    ScaleAndPackColumnsKernel<<<grid, kThreads, 0, stream>>>(
        z_cols_float,
        z_cols_half,
        column_scale,
        z_rows,
        batch_size
    );
    if (cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
        throw std::runtime_error(std::string("ScaleAndPackColumnsKernel: ") + cudaGetErrorString(status));
    }
}

void LaunchForwardPointwiseKernel(
    const float *gate_col,
    const float *bias,
    const float *c_prev,
    const float *h_prev,
    float *h_next,
    float *c_next,
    __half *y_half_out,
    __half *gate_cache_step,
    __half *h_cache,
    __half *c_cache,
    float *z_cache_col,
    const float *column_scale,
    float *checkpoint_dst_h,
    float *checkpoint_dst_c,
    size_t z_rows,
    size_t input_size,
    int has_next_column,
    size_t next_column_offset,
    size_t cache_index,
    size_t batch_size,
    size_t hidden_size,
    cudaStream_t stream
) {
    constexpr int kThreads = 256;
    const size_t total = batch_size * hidden_size;
    const int blocks = total == 0 ? 1 : static_cast<int>((total + kThreads - 1) / kThreads);
    PointwiseContext ctx{};
    ctx.c_prev = const_cast<float *>(c_prev);
    ctx.h_prev = const_cast<float *>(h_prev);
    ctx.h_next = h_next;
    ctx.c_next = c_next;
    ctx.y_half_out = y_half_out;
    ctx.gate_cache_step = gate_cache_step;
    ctx.h_cache = h_cache;
    ctx.c_cache = c_cache;
    ctx.z_cache_col = z_cache_col;
    ctx.checkpoint_dst_h = checkpoint_dst_h;
    ctx.checkpoint_dst_c = checkpoint_dst_c;
    ForwardPointwiseKernel<<<blocks, kThreads, 0, stream>>>(
        gate_col,
        bias,
        ctx,
        column_scale,
        z_rows,
        input_size,
        has_next_column,
        next_column_offset,
        cache_index,
        batch_size,
        hidden_size
    );
    if (cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
        throw std::runtime_error(std::string("ForwardPointwiseKernel: ") + cudaGetErrorString(status));
    }
}

} // namespace testing
} // namespace flstm

namespace flstm {

void StreamingLstmForward(
    size_t time_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,
    size_t recompute_interval,

    const __half *x_tensor_host,
    const __half *h_0_device,
    const __half *c_0_device,

    const float *weights_ih,
    const float *weights_hh,
    const float *bias_ih,
    const float *bias_hh,

    __half *y_tensor_host,

    GateCacheHost gate_cache_host,
    StreamingLstmOptions options,
    __half *hy_device,
    __half *cy_device,

    cudaStream_t compute_stream,
    cudaStream_t h2d_stream,
    cudaStream_t d2h_stream
) {
    GPUTX_RANGE("StreamingLstmForward");
    mfu::Profiler profiler("forward");
    if (time_steps == 0 || batch_size == 0 || input_size == 0 || hidden_size == 0) {
        return;
    }
    ValidateGateCacheOptions(options);
    if (recompute_interval == 0) {
        throw std::runtime_error("StreamingLstmForward requires recompute_interval >= 1");
    }
    if (compute_stream == h2d_stream || compute_stream == d2h_stream || h2d_stream == d2h_stream) {
        throw std::runtime_error("StreamingLstmForward requires distinct compute/h2d/d2h streams");
    }

    const size_t gate_dim = 4 * hidden_size;
    const size_t z_rows = input_size + hidden_size;
    const size_t bh_elements = batch_size * hidden_size;
    const size_t checkpoint_stride = recompute_interval;
    const size_t checkpoint_count = (time_steps + checkpoint_stride - 1) / checkpoint_stride;
    const size_t checkpoint_h_elements = checkpoint_count * bh_elements;
    const size_t checkpoint_c_elements = checkpoint_count * bh_elements;
    const size_t checkpoint_h_bytes = checkpoint_h_elements * GateCacheDTypeSize(options.h_dtype);
    const size_t checkpoint_c_bytes = checkpoint_c_elements * GateCacheDTypeSize(options.c_dtype);
    const size_t x_step_bytes = batch_size * input_size * sizeof(__half);
    const size_t y_step_bytes = bh_elements * sizeof(__half);
    constexpr size_t kChunkSteps = 32;
    const size_t chunk_capacity = kChunkSteps;
    const size_t chunk_input_capacity = chunk_capacity * batch_size * input_size;
    const size_t chunk_output_capacity = chunk_capacity * bh_elements;
    const bool store_checkpoints = (gate_cache_host.h_ptr != nullptr && gate_cache_host.c_ptr != nullptr);
    const bool needs_y_fallback = (y_tensor_host != nullptr);

    const int threads = 256;
    const int bh_blocks = BlocksFor(bh_elements, threads);

    HostRegistration x_host_registration;
    HostRegistration y_host_registration;
    HostRegistration gate_h_host_registration;
    HostRegistration gate_c_host_registration;

    x_host_registration.reset(
        x_tensor_host,
        time_steps * x_step_bytes,
        "cudaHostRegister x_tensor_host"
    );
    if (y_tensor_host != nullptr) {
        y_host_registration.reset(
            y_tensor_host,
            time_steps * y_step_bytes,
            "cudaHostRegister y_tensor_host"
        );
    }
    if (store_checkpoints) {
        gate_h_host_registration.reset(
            gate_cache_host.h_ptr,
            checkpoint_h_bytes,
            "cudaHostRegister checkpoint_cache_h_host"
        );
        gate_c_host_registration.reset(
            gate_cache_host.c_ptr,
            checkpoint_c_bytes,
            "cudaHostRegister checkpoint_cache_c_host"
        );
    }

    const size_t z_chunk_elements = z_rows * chunk_capacity * batch_size;
    DeviceBuffer<float> z_chunk_float_buffer;
    AllocateDeviceBuffer(z_chunk_float_buffer, z_chunk_elements, "cudaMalloc z_chunk_float");
    float *z_chunk_float = z_chunk_float_buffer.ptr;
    DeviceBuffer<__half> z_step_half_buffer;
    AllocateDeviceBuffer(z_step_half_buffer, z_rows * batch_size, "cudaMalloc z_step_half");
    DeviceBuffer<float> column_scale_buffer;
    DeviceBuffer<float> column_scale_buffer_alt;
    AllocateDeviceBuffer(column_scale_buffer, batch_size, "cudaMalloc column_scale");
    AllocateDeviceBuffer(column_scale_buffer_alt, batch_size, "cudaMalloc column_scale_alt");

    DeviceBuffer<float> h0_float;
    DeviceBuffer<float> c0_float;
    DeviceBuffer<float> h_prev;
    DeviceBuffer<float> c_prev;
    DeviceBuffer<float> h_next;
    DeviceBuffer<float> c_next;
    DeviceBuffer<float> gate_pre_col;
    DeviceBuffer<__half> weight_cat_half;
    DeviceBuffer<float> bias_fused;

    AllocateDeviceBuffer(h0_float, bh_elements, "cudaMalloc h0_float");
    AllocateDeviceBuffer(c0_float, bh_elements, "cudaMalloc c0_float");
    HalfToFloatKernel<<<bh_blocks, threads, 0, compute_stream>>>(h_0_device, h0_float.ptr, bh_elements);
    CheckCuda(cudaGetLastError(), "HalfToFloatKernel h0");
    HalfToFloatKernel<<<bh_blocks, threads, 0, compute_stream>>>(c_0_device, c0_float.ptr, bh_elements);
    CheckCuda(cudaGetLastError(), "HalfToFloatKernel c0");

    AllocateDeviceBuffer(h_prev, bh_elements, "cudaMalloc h_prev");
    AllocateDeviceBuffer(c_prev, bh_elements, "cudaMalloc c_prev");
    AllocateDeviceBuffer(h_next, bh_elements, "cudaMalloc h_next");
    AllocateDeviceBuffer(c_next, bh_elements, "cudaMalloc c_next");
    CheckCuda(cudaMemcpyAsync(
                  h_prev.ptr,
                  h0_float.ptr,
                  bh_elements * sizeof(float),
                  cudaMemcpyDeviceToDevice,
                  compute_stream),
              "copy h0 -> h_prev");
    CheckCuda(cudaMemcpyAsync(
                  c_prev.ptr,
                  c0_float.ptr,
                  bh_elements * sizeof(float),
                  cudaMemcpyDeviceToDevice,
                  compute_stream),
              "copy c0 -> c_prev");

    AllocateDeviceBuffer(gate_pre_col, gate_dim * batch_size, "cudaMalloc gate_pre_col");
    AllocateDeviceBuffer(weight_cat_half, z_rows * gate_dim, "cudaMalloc weight_cat_half");
    AllocateDeviceBuffer(bias_fused, gate_dim, "cudaMalloc bias_fused");

    const int weight_blocks = BlocksFor(z_rows * gate_dim, threads);
    FuseWeightsKernel<<<weight_blocks, threads, 0, compute_stream>>>(
        weights_ih,
        weights_hh,
        weight_cat_half.ptr,
        input_size,
        hidden_size
    );
    CheckCuda(cudaGetLastError(), "FuseWeightsKernel");

    const int bias_blocks = BlocksFor(gate_dim, threads);
    FuseBiasKernel<<<bias_blocks, threads, 0, compute_stream>>>(bias_ih, bias_hh, bias_fused.ptr, hidden_size);
    CheckCuda(cudaGetLastError(), "FuseBiasKernel");

    const float alpha_f = 1.0f;
    const float beta_zero_f = 0.0f;
    const int seed_blocks = BlocksFor(bh_elements, threads);

    DeviceBuffer<__half> x_chunk_buffers[2];
    DeviceBuffer<__half> y_chunk_half[2];
    AllocateDeviceBufferArray(x_chunk_buffers, chunk_input_capacity, "cudaMalloc x_chunk_buffer");
    if (needs_y_fallback) {
        AllocateDeviceBufferArray(y_chunk_half, chunk_output_capacity, "cudaMalloc y_chunk_half");
    }

    const size_t max_checkpoints_per_chunk = std::max<size_t>(1, (chunk_capacity + checkpoint_stride - 1) / checkpoint_stride + 1);
    DeviceBuffer<float> checkpoint_chunks_h[2];
    DeviceBuffer<float> checkpoint_chunks_c[2];
    DeviceBuffer<__half> checkpoint_chunks_h_half[2];
    DeviceBuffer<__half> checkpoint_chunks_c_half[2];
    const bool h_cache_uses_half = (options.h_dtype == GateCacheDType::kFloat16);
    const bool c_cache_uses_half = (options.c_dtype == GateCacheDType::kFloat16);
    if (store_checkpoints) {
        AllocateDeviceBufferArray(checkpoint_chunks_h, max_checkpoints_per_chunk * bh_elements, "cudaMalloc checkpoint_chunk_h");
        AllocateDeviceBufferArray(checkpoint_chunks_c, max_checkpoints_per_chunk * bh_elements, "cudaMalloc checkpoint_chunk_c");
        if (h_cache_uses_half) {
            AllocateDeviceBufferArray(checkpoint_chunks_h_half, max_checkpoints_per_chunk * bh_elements, "cudaMalloc checkpoint_chunk_h_half");
        }
        if (c_cache_uses_half) {
            AllocateDeviceBufferArray(checkpoint_chunks_c_half, max_checkpoints_per_chunk * bh_elements, "cudaMalloc checkpoint_chunk_c_half");
        }
    }
    CudaEvent checkpoint_copy_done[2];
    bool checkpoint_copy_inflight[2] = {false, false};
    size_t checkpoint_counts[2] = {0, 0};
    size_t checkpoint_host_offsets[2] = {0, 0};
    CudaEvent x_ready[2];
    CudaEvent compute_done[2];
    CudaEvent y_copy_done[2];
    bool compute_done_valid[2] = {false, false};
    bool y_copy_inflight[2] = {false, false};
    size_t chunk_steps[2] = {0, 0};

    const size_t total_chunks = (time_steps + chunk_capacity - 1) / chunk_capacity;

    // Shared metadata for the double-buffered input copy helper.
    ForwardChunkCopyParams chunk_params{};
    chunk_params.total_chunks = total_chunks;
    chunk_params.time_steps = time_steps;
    chunk_params.chunk_capacity = chunk_capacity;
    chunk_params.batch_size = batch_size;
    chunk_params.input_size = input_size;
    chunk_params.x_tensor_host = x_tensor_host;
    chunk_params.x_chunk_buffers = x_chunk_buffers;
    chunk_params.x_ready = x_ready;
    chunk_params.h2d_stream = h2d_stream;
    chunk_params.compute_done_valid = compute_done_valid;
    chunk_params.compute_done = compute_done;

    size_t prefetched = 0;
    for (; prefetched < std::min<size_t>(total_chunks, 2); ++prefetched) {
        const int slot = static_cast<int>(prefetched % 2);
        chunk_steps[slot] = IssueChunkCopy(prefetched, slot, chunk_params);
    }

    const int point_blocks = BlocksFor(bh_elements, threads);

    size_t chunk_idx = 0;
    while (chunk_idx < total_chunks) {
        const int slot = static_cast<int>(chunk_idx % 2);
        const size_t steps_in_chunk = chunk_steps[slot];
        if (steps_in_chunk == 0) {
            break;
        }

        if (needs_y_fallback && y_copy_inflight[slot]) {
            CheckCuda(cudaStreamWaitEvent(compute_stream, y_copy_done[slot].evt, 0),
                      "wait y copy before compute");
            y_copy_inflight[slot] = false;
        }
        if (store_checkpoints && checkpoint_copy_inflight[slot]) {
            CheckCuda(cudaStreamWaitEvent(compute_stream, checkpoint_copy_done[slot].evt, 0),
                      "wait checkpoint copy before compute");
            checkpoint_copy_inflight[slot] = false;
        }
        CheckCuda(cudaStreamWaitEvent(compute_stream, x_ready[slot].evt, 0), "wait x chunk ready");

        const size_t chunk_start_step = chunk_idx * chunk_capacity;
        const size_t column_count = steps_in_chunk * batch_size;
        if (column_count > 0) {
            const dim3 convert_grid(static_cast<unsigned int>(column_count));
            ConvertInputToZCacheKernel<<<convert_grid, threads, 0, compute_stream>>>(
                x_chunk_buffers[slot].ptr,
                z_chunk_float,
                0,
                steps_in_chunk,
                batch_size,
                input_size,
                hidden_size
            );
            CheckCuda(cudaGetLastError(), "ConvertInputToZCacheKernel chunk");
        }
        if (steps_in_chunk > 0) {
            SeedHiddenColumnKernel<<<seed_blocks, threads, 0, compute_stream>>>(
                h_prev.ptr,
                z_chunk_float,
                batch_size,
                input_size,
                hidden_size
            );
            CheckCuda(cudaGetLastError(), "SeedHiddenColumnKernel chunk");
        }

        float *column_scale_cur = column_scale_buffer.ptr;
        float *column_scale_next = column_scale_buffer_alt.ptr;

        if (steps_in_chunk > 0) {
            const float *z_step_float0 = z_chunk_float;
            const int scale_blocks_init = static_cast<int>(batch_size);
            ScaleAndPackColumnsKernel<<<scale_blocks_init, threads, 0, compute_stream>>>(
                z_step_float0,
                z_step_half_buffer.ptr,
                column_scale_cur,
                z_rows,
                batch_size
            );
            CheckCuda(cudaGetLastError(), "ScaleAndPackColumnsKernel init");
        }

    size_t next_checkpoint_step = static_cast<size_t>(-1);
    size_t checkpoint_global_index = 0;
        if (store_checkpoints && checkpoint_stride > 0) {
            const size_t chunk_end_step = chunk_start_step + steps_in_chunk;
            size_t aligned = (chunk_start_step + checkpoint_stride - 1) / checkpoint_stride;
            next_checkpoint_step = aligned * checkpoint_stride;
            if (next_checkpoint_step < chunk_start_step) {
                next_checkpoint_step += checkpoint_stride;
            }
            if (next_checkpoint_step >= chunk_end_step) {
                next_checkpoint_step = static_cast<size_t>(-1);
            } else {
                checkpoint_global_index = next_checkpoint_step / checkpoint_stride;
            }
            checkpoint_counts[slot] = 0;
        }

        for (size_t step = 0; step < steps_in_chunk; ++step) {
            const size_t global_step = chunk_start_step + step;
            __half *y_step = nullptr;
            if (needs_y_fallback && y_chunk_half[slot].ptr != nullptr) {
                y_step = y_chunk_half[slot].ptr + step * bh_elements;
            }
            __half *gate_cache_step = nullptr;
            float *checkpoint_dst_h = nullptr;
            float *checkpoint_dst_c = nullptr;

            if (store_checkpoints && next_checkpoint_step != static_cast<size_t>(-1) &&
                global_step == next_checkpoint_step) {
                if (checkpoint_global_index < checkpoint_count) {
                    checkpoint_dst_h = checkpoint_chunks_h[slot].ptr + checkpoint_counts[slot] * bh_elements;
                    checkpoint_dst_c = checkpoint_chunks_c[slot].ptr + checkpoint_counts[slot] * bh_elements;
                    if (checkpoint_counts[slot] == 0) {
                        checkpoint_host_offsets[slot] = checkpoint_global_index;
                    }
                    ++checkpoint_counts[slot];
                    next_checkpoint_step += checkpoint_stride;
                    ++checkpoint_global_index;
                    if (next_checkpoint_step >= chunk_start_step + steps_in_chunk) {
                        next_checkpoint_step = static_cast<size_t>(-1);
                    }
                } else {
                    next_checkpoint_step = static_cast<size_t>(-1);
                }
            }

        flstm::GemmTN(
            static_cast<int>(gate_dim),
            static_cast<int>(batch_size),
            static_cast<int>(z_rows),
            weight_cat_half.ptr,
            static_cast<int>(z_rows),
            z_step_half_buffer.ptr,
            static_cast<int>(z_rows),
            gate_pre_col.ptr,
            static_cast<int>(gate_dim),
            alpha_f,
            beta_zero_f,
            compute_stream
        );
        profiler.AddUseful(mfu::GemmFlops(gate_dim, batch_size, z_rows));

        const bool has_next_column = (step + 1 < steps_in_chunk);
        const size_t next_column_offset = has_next_column ? ((step + 1) * batch_size) : 0;

        const int scale_blocks = static_cast<int>(batch_size);
        PointwiseContext ctx{};
        ctx.c_prev = c_prev.ptr;
        ctx.h_prev = h_prev.ptr;
        ctx.h_next = h_next.ptr;
        ctx.c_next = c_next.ptr;
        ctx.y_half_out = y_step;
        ctx.gate_cache_step = gate_cache_step;
        ctx.h_cache = nullptr;
        ctx.c_cache = nullptr;
        ctx.z_cache_col = z_chunk_float;
        ctx.checkpoint_dst_h = checkpoint_dst_h;
        ctx.checkpoint_dst_c = checkpoint_dst_c;

        const bool has_next_step = (step + 1 < steps_in_chunk);
        const float *next_z_cols = has_next_step ? (z_chunk_float + (step + 1) * batch_size * z_rows) : nullptr;
        float *column_scale_out = has_next_step ? column_scale_next : nullptr;
        __half *next_z_half = has_next_step ? z_step_half_buffer.ptr : nullptr;

        ScalePackAndPointwiseKernel<<<scale_blocks, threads, 0, compute_stream>>>(
            gate_pre_col.ptr,
            bias_fused.ptr,
            ctx,
            column_scale_cur,
            column_scale_out,
            next_z_cols,
            next_z_half,
            z_rows,
            batch_size,
            hidden_size,
            input_size,
            step + 1,
            has_next_step ? 1 : 0,
            has_next_column ? 1 : 0,
            next_column_offset
        );
        CheckCuda(cudaGetLastError(), "ScalePackAndPointwiseKernel");
        if (has_next_step) {
            std::swap(column_scale_cur, column_scale_next);
        }

        std::swap(h_prev.ptr, h_next.ptr);
        std::swap(c_prev.ptr, c_next.ptr);
        }

        if (store_checkpoints) {
            const size_t checkpoint_count_chunk = checkpoint_counts[slot];
            if (checkpoint_count_chunk > 0) {
                const size_t checkpoint_elements_chunk = checkpoint_count_chunk * bh_elements;
                if (h_cache_uses_half) {
                    const int convert_blocks = BlocksFor(checkpoint_elements_chunk, threads);
                    FloatToHalfKernel<<<convert_blocks, threads, 0, compute_stream>>>(
                        checkpoint_chunks_h[slot].ptr,
                        checkpoint_chunks_h_half[slot].ptr,
                        checkpoint_elements_chunk
                    );
                    CheckCuda(cudaGetLastError(), "FloatToHalfKernel checkpoint h");
                }
                if (c_cache_uses_half) {
                    const int convert_blocks = BlocksFor(checkpoint_elements_chunk, threads);
                    FloatToHalfKernel<<<convert_blocks, threads, 0, compute_stream>>>(
                        checkpoint_chunks_c[slot].ptr,
                        checkpoint_chunks_c_half[slot].ptr,
                        checkpoint_elements_chunk
                    );
                    CheckCuda(cudaGetLastError(), "FloatToHalfKernel checkpoint c");
                }
            }
        }

        CheckCuda(cudaEventRecord(compute_done[slot].evt, compute_stream), "record compute_done");
        compute_done_valid[slot] = true;

        CheckCuda(cudaStreamWaitEvent(d2h_stream, compute_done[slot].evt, 0), "wait compute_done on d2h");

        if (store_checkpoints) {
            const size_t checkpoint_count_chunk = checkpoint_counts[slot];
            if (checkpoint_count_chunk > 0) {
                const size_t dst_offset = checkpoint_host_offsets[slot] * bh_elements;
                const size_t chunk_elements = checkpoint_count_chunk * bh_elements;
                if (!h_cache_uses_half) {
                    float *dst = reinterpret_cast<float *>(gate_cache_host.h_ptr) + dst_offset;
                    const float *src = checkpoint_chunks_h[slot].ptr;
                    CheckCuda(cudaMemcpyAsync(
                                  dst,
                                  src,
                                  chunk_elements * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  d2h_stream),
                              "copy checkpoint h chunk");
                } else {
                    __half *dst = reinterpret_cast<__half *>(gate_cache_host.h_ptr) + dst_offset;
                    const __half *src = checkpoint_chunks_h_half[slot].ptr;
                    CheckCuda(cudaMemcpyAsync(
                                  dst,
                                  src,
                                  chunk_elements * sizeof(__half),
                                  cudaMemcpyDeviceToHost,
                                  d2h_stream),
                              "copy checkpoint h chunk half");
                }

                if (!c_cache_uses_half) {
                    float *dst = reinterpret_cast<float *>(gate_cache_host.c_ptr) + dst_offset;
                    const float *src = checkpoint_chunks_c[slot].ptr;
                    CheckCuda(cudaMemcpyAsync(
                                  dst,
                                  src,
                                  chunk_elements * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  d2h_stream),
                              "copy checkpoint c chunk");
                } else {
                    __half *dst = reinterpret_cast<__half *>(gate_cache_host.c_ptr) + dst_offset;
                    const __half *src = checkpoint_chunks_c_half[slot].ptr;
                    CheckCuda(cudaMemcpyAsync(
                                  dst,
                                  src,
                                  chunk_elements * sizeof(__half),
                                  cudaMemcpyDeviceToHost,
                                  d2h_stream),
                              "copy checkpoint c chunk half");
                }

                CheckCuda(cudaEventRecord(checkpoint_copy_done[slot].evt, d2h_stream), "record checkpoint_copy_done");
                checkpoint_copy_inflight[slot] = true;
            }
        }

        if (y_tensor_host != nullptr && y_chunk_half[slot].ptr != nullptr) {
            const size_t y_elements_chunk = steps_in_chunk * bh_elements;
            if (y_elements_chunk > 0) {
                __half *dst = y_tensor_host + chunk_start_step * bh_elements;
                const __half *src = y_chunk_half[slot].ptr;
                CheckCuda(cudaMemcpyAsync(
                              dst,
                              src,
                              y_elements_chunk * sizeof(__half),
                              cudaMemcpyDeviceToHost,
                              d2h_stream),
                          "copy y chunk");
                CheckCuda(cudaEventRecord(y_copy_done[slot].evt, d2h_stream), "record y_copy_done");
                y_copy_inflight[slot] = true;
            }
        }

        if (prefetched < total_chunks) {
            const int reuse_slot = slot;
            chunk_steps[reuse_slot] = IssueChunkCopy(prefetched, reuse_slot, chunk_params);
            ++prefetched;
        }

        ++chunk_idx;
    }

    if (hy_device != nullptr) {
        const int hy_blocks = BlocksFor(bh_elements, threads);
        FloatToHalfKernel<<<hy_blocks, threads, 0, compute_stream>>>(
            h_prev.ptr,
            hy_device,
            bh_elements
        );
        CheckCuda(cudaGetLastError(), "FloatToHalfKernel hy");
    }

    if (cy_device != nullptr) {
        const int cy_blocks = BlocksFor(bh_elements, threads);
        FloatToHalfKernel<<<cy_blocks, threads, 0, compute_stream>>>(
            c_prev.ptr,
            cy_device,
            bh_elements
        );
        CheckCuda(cudaGetLastError(), "FloatToHalfKernel cy");
    }

    CheckCuda(cudaStreamSynchronize(h2d_stream), "final h2d transfer sync");
    CheckCuda(cudaStreamSynchronize(d2h_stream), "final d2h transfer sync");
    CheckCuda(cudaStreamSynchronize(compute_stream), "final compute sync");
    profiler.Finish();
}

} // namespace flstm

extern "C" void flstm_StreamingLstmForward(
    const size_t time_steps,
    const size_t batch_size,
    const size_t input_size,
    const size_t hidden_size,
    const size_t recompute_interval,

    const __half *x_tensor_host,
    const __half *h0_device,
    const __half *c0_device,

    const float *weights_ih,
    const float *weights_hh,
    const float *bias_ih,
    const float *bias_hh,

    __half *y_tensor_host,

    flstm_GateCacheHost gate_cache_host,
    const flstm_StreamingLstmOptions *options,
    __half *hy_device,
    __half *cy_device,

    const cudaStream_t compute_stream,
    const cudaStream_t h2d_stream,
    const cudaStream_t d2h_stream
) {
    try {
        flstm::GateCacheHost cache_host{
            gate_cache_host.h_ptr,
            gate_cache_host.c_ptr,
        };
        flstm::StreamingLstmOptions opts;
        if (options != nullptr) {
            opts.h_dtype = static_cast<flstm::GateCacheDType>(options->h_dtype);
            opts.c_dtype = static_cast<flstm::GateCacheDType>(options->c_dtype);
        }
        flstm::StreamingLstmForward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            x_tensor_host,
            h0_device,
            c0_device,
            weights_ih,
            weights_hh,
            bias_ih,
            bias_hh,
            y_tensor_host,
            cache_host,
            opts,
            hy_device,
            cy_device,
            compute_stream,
            h2d_stream,
            d2h_stream
        );
    } catch (const std::exception &exc) {
        fprintf(stderr, "flstm_StreamingLstmForward failed: %s\n", exc.what());
        fflush(stderr);
        std::abort();
    } catch (...) {
        fprintf(stderr, "flstm_StreamingLstmForward failed: unknown exception\n");
        fflush(stderr);
        std::abort();
    }
}
