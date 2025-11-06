#include "lstm.hpp"
#include "gputx.h"

#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

inline const char *CublasStatusString(const cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "alloc failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "arch mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
#if CUBLAS_VERSION >= 11000
        case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
        case CUBLAS_STATUS_LICENSE_ERROR: return "license error";
#endif
        default: return "unknown";
    }
}

inline void CheckCuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

inline void CheckCublas(cublasStatus_t status, const char *what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + CublasStatusString(status));
    }
}

// Helper for consistent 1D launch configuration.
inline int BlocksFor(size_t count, int threads = 256) {
    return static_cast<int>((count + threads - 1) / threads);
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

struct CublasHandle {
    cublasHandle_t handle{nullptr};
    CublasHandle() = default;
    CublasHandle(const CublasHandle &) = delete;
    CublasHandle &operator=(const CublasHandle &) = delete;
    ~CublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
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

__global__ void ConvertInputToZCacheKernel(
    const __half *x_src,           // (chunk_steps, B, I) half
    __half *z_cache_col,           // (I+H, T*B) column-major
    size_t time_offset,
    size_t chunk_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size
) {
    const size_t total = chunk_steps * batch_size * input_size;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    size_t tmp = idx;
    const size_t input_idx = tmp % input_size;
    tmp /= input_size;
    const size_t batch_idx = tmp % batch_size;
    const size_t step_idx = tmp / batch_size;
    const size_t time_idx = time_offset + step_idx;

    const size_t column = time_idx * batch_size + batch_idx;
    const size_t z_rows = input_size + hidden_size;
    z_cache_col[input_idx + column * z_rows] = x_src[idx];
}

__global__ void SeedHiddenColumnKernel(
    const float *h0,          // (B, H) row-major
    __half *z_cache_col,      // (I+H, T*B) column-major
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
    z_cache_col[input_size + hidden_idx + batch_idx * z_rows] = __float2half(value);
}

__global__ void ForwardPointwiseKernel(
    const float *gate_col,         // (4H, B) column-major
    const float *bias,             // (4H,)
    const float *c_prev,           // (B, H) row-major
    float *h_next,                 // (B, H) row-major
    float *c_next,                 // (B, H) row-major
    float *y_out,                  // (B, H) row-major
    __half *gate_cache_step,       // (B, 4H) row-major
    __half *h_cache,               // (T+1, B, H) row-major
    __half *c_cache,               // (T+1, B, H) row-major
    __half *z_cache_col,           // (I+H, T*B) column-major
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

    const float gi = gate_col[row_base + col * gate_dim] + bias[row_base + 0 * hidden_size];
    const float gf = gate_col[row_base + hidden_size + col * gate_dim] + bias[row_base + 1 * hidden_size];
    const float gg = gate_col[row_base + 2 * hidden_size + col * gate_dim] + bias[row_base + 2 * hidden_size];
    const float go = gate_col[row_base + 3 * hidden_size + col * gate_dim] + bias[row_base + 3 * hidden_size];

    const float i_gate = 1.0f / (1.0f + expf(-gi));
    const float f_gate = 1.0f / (1.0f + expf(-gf));
    const float g_gate = tanhf(gg);
    const float c_prev_val = c_prev[batch_idx * hidden_size + hidden_idx];
    const float c_val = f_gate * c_prev_val + i_gate * g_gate;
    const float o_gate = 1.0f / (1.0f + expf(-go));
    const float h_val = o_gate * tanhf(c_val);

    h_next[batch_idx * hidden_size + hidden_idx] = h_val;
    c_next[batch_idx * hidden_size + hidden_idx] = c_val;
    y_out[batch_idx * hidden_size + hidden_idx] = h_val;

    if (gate_cache_step != nullptr) {
        __half *gate_ptr = gate_cache_step + batch_idx * gate_dim;
        gate_ptr[hidden_idx + 0 * hidden_size] = __float2half(i_gate);
        gate_ptr[hidden_idx + 1 * hidden_size] = __float2half(f_gate);
        gate_ptr[hidden_idx + 2 * hidden_size] = __float2half(g_gate);
        gate_ptr[hidden_idx + 3 * hidden_size] = __float2half(o_gate);
    }
    if (h_cache != nullptr) {
        __half *dst = h_cache + cache_index * (batch_size * hidden_size);
        dst[batch_idx * hidden_size + hidden_idx] = __float2half(h_val);
    }
    if (c_cache != nullptr) {
        __half *dst = c_cache + cache_index * (batch_size * hidden_size);
        dst[batch_idx * hidden_size + hidden_idx] = __float2half(c_val);
    }
    if (has_next_column && z_cache_col != nullptr) {
        const size_t column = next_column_offset + batch_idx;
        z_cache_col[input_size + hidden_idx + column * z_rows] = __float2half(h_val);
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

void StreamingLstmForward(
    size_t time_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,

    const __half *x_tensor_host,
    const __half *h_0_device,
    const __half *c_0_device,

    const float *weights_ih,
    const float *weights_hh,
    const float *bias_ih,
    const float *bias_hh,

    __half *y_tensor_host,

    __half *gate_cache_host,

    cudaStream_t compute_stream,
    cudaStream_t h2d_stream,
    cudaStream_t d2h_stream
) {
    GPUTX_RANGE("StreamingLstmForward");
    if (time_steps == 0 || batch_size == 0 || input_size == 0 || hidden_size == 0) {
        return;
    }
    if (compute_stream == h2d_stream || compute_stream == d2h_stream || h2d_stream == d2h_stream) {
        throw std::runtime_error("StreamingLstmForward requires distinct compute/h2d/d2h streams");
    }

    const size_t gate_dim = 4 * hidden_size;
    const size_t z_rows = input_size + hidden_size;
    const size_t bh_elements = batch_size * hidden_size;
    const size_t total_tb = time_steps * batch_size;
    const size_t z_cache_elements = z_rows * total_tb;
    const size_t gate_cache_elements = time_steps * batch_size * gate_dim;
    const size_t x_step_bytes = batch_size * input_size * sizeof(__half);
    const size_t y_step_bytes = bh_elements * sizeof(__half);
    constexpr size_t kChunkSteps = 32;
    const size_t chunk_capacity = kChunkSteps;
    const size_t chunk_input_capacity = chunk_capacity * batch_size * input_size;
    const size_t chunk_output_capacity = chunk_capacity * bh_elements;
    const bool store_gate_cache = (gate_cache_host != nullptr);
    const bool needs_y_fallback = (y_tensor_host != nullptr);

    const int threads = 256;
    const int bh_blocks = BlocksFor(bh_elements, threads);

    HostRegistration x_host_registration;
    HostRegistration y_host_registration;
    HostRegistration gate_host_registration;

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
    if (store_gate_cache) {
        gate_host_registration.reset(
            gate_cache_host,
            gate_cache_elements * sizeof(__half),
            "cudaHostRegister gate_cache_host"
        );
    }

    DeviceBuffer<__half> z_cache_device_buffer;
    AllocateDeviceBuffer(z_cache_device_buffer, z_cache_elements, "cudaMalloc z_cache");
    __half *z_cache_device = z_cache_device_buffer.ptr;

    DeviceBuffer<__half> gate_cache_device_buffer;
    if (store_gate_cache) {
        AllocateDeviceBuffer(gate_cache_device_buffer, gate_cache_elements, "cudaMalloc gate_cache");
    }
    __half *gate_cache_device = gate_cache_device_buffer.ptr;

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

    const int seed_blocks = BlocksFor(bh_elements, threads);
    SeedHiddenColumnKernel<<<seed_blocks, threads, 0, compute_stream>>>(
        h0_float.ptr,
        z_cache_device,
        batch_size,
        input_size,
        hidden_size
    );
    CheckCuda(cudaGetLastError(), "SeedHiddenColumnKernel");

    CublasHandle cublas;
    CheckCublas(cublasCreate(&cublas.handle), "cublasCreate");
    CheckCublas(cublasSetStream(cublas.handle, compute_stream), "cublasSetStream");
    const float alpha_f = 1.0f;
    const float beta_zero_f = 0.0f;

    DeviceBuffer<__half> x_chunk_buffers[2];
    DeviceBuffer<float> y_chunk_float[2];
    DeviceBuffer<__half> y_chunk_half[2];
    AllocateDeviceBufferArray(x_chunk_buffers, chunk_input_capacity, "cudaMalloc x_chunk_buffer");
    AllocateDeviceBufferArray(y_chunk_float, chunk_output_capacity, "cudaMalloc y_chunk_float");
    if (needs_y_fallback) {
        AllocateDeviceBufferArray(y_chunk_half, chunk_output_capacity, "cudaMalloc y_chunk_half");
    }

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

        CheckCuda(cudaStreamWaitEvent(compute_stream, x_ready[slot].evt, 0), "wait x chunk ready");

        const size_t chunk_start_step = chunk_idx * chunk_capacity;
        const size_t chunk_input_elems = steps_in_chunk * batch_size * input_size;
        if (chunk_input_elems > 0) {
            const int convert_blocks = BlocksFor(chunk_input_elems, threads);
            ConvertInputToZCacheKernel<<<convert_blocks, threads, 0, compute_stream>>>(
                x_chunk_buffers[slot].ptr,
                z_cache_device,
                chunk_start_step,
                steps_in_chunk,
                batch_size,
                input_size,
                hidden_size
            );
            CheckCuda(cudaGetLastError(), "ConvertInputToZCacheKernel chunk");
        }

        for (size_t step = 0; step < steps_in_chunk; ++step) {
            const size_t global_step = chunk_start_step + step;
            const size_t column_offset = global_step * batch_size;
            const __half *z_step = z_cache_device + column_offset * z_rows;
            float *y_step = y_chunk_float[slot].ptr + step * bh_elements;
            __half *gate_cache_step = (gate_cache_device != nullptr)
                                          ? (gate_cache_device + global_step * batch_size * gate_dim)
                                          : nullptr;

            CheckCublas(cublasGemmEx(
                            cublas.handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            static_cast<int>(gate_dim),
                            static_cast<int>(batch_size),
                            static_cast<int>(z_rows),
                            &alpha_f,
                            weight_cat_half.ptr,
                            CUDA_R_16F,
                            static_cast<int>(z_rows),
                            z_step,
                            CUDA_R_16F,
                            static_cast<int>(z_rows),
                            &beta_zero_f,
                            gate_pre_col.ptr,
                            CUDA_R_32F,
                            static_cast<int>(gate_dim),
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                        "cublasGemmEx gates");

            const bool has_next_column = (global_step + 1 < time_steps);
            const size_t next_column_offset = has_next_column ? ((global_step + 1) * batch_size) : 0;

            ForwardPointwiseKernel<<<point_blocks, threads, 0, compute_stream>>>(
                gate_pre_col.ptr,
                bias_fused.ptr,
                c_prev.ptr,
                h_next.ptr,
                c_next.ptr,
                y_step,
                gate_cache_step,
                nullptr,
                nullptr,
                z_cache_device,
                z_rows,
                input_size,
                has_next_column ? 1 : 0,
                next_column_offset,
                global_step + 1,
                batch_size,
                hidden_size
            );
            CheckCuda(cudaGetLastError(), "ForwardPointwiseKernel");

            std::swap(h_prev.ptr, h_next.ptr);
            std::swap(c_prev.ptr, c_next.ptr);
        }

        if (needs_y_fallback) {
            const size_t chunk_y_elements = steps_in_chunk * bh_elements;
            if (chunk_y_elements > 0) {
                const int y_blocks = BlocksFor(chunk_y_elements, threads);
                FloatToHalfKernel<<<y_blocks, threads, 0, compute_stream>>>(
                    y_chunk_float[slot].ptr,
                    y_chunk_half[slot].ptr,
                    chunk_y_elements
                );
                CheckCuda(cudaGetLastError(), "FloatToHalfKernel chunk y");
            }
        }

        CheckCuda(cudaEventRecord(compute_done[slot].evt, compute_stream), "record compute_done");
        compute_done_valid[slot] = true;

        CheckCuda(cudaStreamWaitEvent(d2h_stream, compute_done[slot].evt, 0), "wait compute_done on d2h");

        if (store_gate_cache) {
            const size_t elements = steps_in_chunk * batch_size * gate_dim;
            if (elements > 0) {
                const size_t offset = chunk_start_step * batch_size * gate_dim;
                __half *dst = gate_cache_host + offset;
                const __half *src = gate_cache_device + offset;
                CheckCuda(cudaMemcpyAsync(
                              dst,
                              src,
                              elements * sizeof(__half),
                              cudaMemcpyDeviceToHost,
                              d2h_stream),
                          "copy gate cache chunk");
            }
        }

        if (y_tensor_host != nullptr) {
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

    CheckCuda(cudaStreamSynchronize(h2d_stream), "final h2d transfer sync");
    CheckCuda(cudaStreamSynchronize(d2h_stream), "final d2h transfer sync");
    CheckCuda(cudaStreamSynchronize(compute_stream), "final compute sync");
}

} // namespace flstm

extern "C" void flstm_StreamingLstmForward(
    size_t time_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,

    const __half *x_tensor_host,
    const __half *h0_device,
    const __half *c0_device,

    const float *weights_ih,
    const float *weights_hh,
    const float *bias_ih,
    const float *bias_hh,

    __half *y_tensor_host,

    __half *gate_cache_host,

    cudaStream_t compute_stream,
    cudaStream_t h2d_stream,
    cudaStream_t d2h_stream
) {
    try {
        flstm::StreamingLstmForward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            x_tensor_host,
            h0_device,
            c0_device,
            weights_ih,
            weights_hh,
            bias_ih,
            bias_hh,
            y_tensor_host,
            gate_cache_host,
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
