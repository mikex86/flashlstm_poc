#include "gemm.h"
#include "gputx.h"
#include "lstm.hpp"
#include "mfu_profiler.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
    void CheckCuda(const cudaError_t err, const char *what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
        }
    }

    /// Helper for 1D grid launches; mirrors the arithmetic sprinkled throughout the kernel site code.
    int BlocksFor(const size_t count, const int threads = 256) {
        return static_cast<int>((count + threads - 1) / threads);
    }

    const char *GateCacheDTypeName(flstm::GateCacheDType dtype) {
        switch (dtype) {
            case flstm::GateCacheDType::kFloat32: return "float32";
            case flstm::GateCacheDType::kFloat16: return "float16";
        }
        return "unknown";
    }

    void ValidateGateCacheOptions(const flstm::StreamingLstmOptions &options) {
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
    void AllocateDeviceBuffer(DeviceBuffer<T> &buffer, const size_t elements, const char *what) {
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

    /// Async zero fill that honours optional buffers and element counts.
    template<typename T>
    void ZeroDeviceMemory(T *ptr, const size_t elements, const cudaStream_t stream, const char *what) {
        if (elements == 0 || ptr == nullptr) {
            return;
        }
        CheckCuda(cudaMemsetAsync(ptr, 0, elements * sizeof(T), stream), what);
    }

    struct CudaEvent {
        cudaEvent_t evt{nullptr};

        CudaEvent() {
            CheckCuda(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming), "cudaEventCreate");
        }

        CudaEvent(const CudaEvent &) = delete;

        CudaEvent &operator=(const CudaEvent &) = delete;

        ~CudaEvent() {
            if (evt != nullptr) {
                cudaEventDestroy(evt);
            }
        }
    };

    __global__ void HalfToFloatKernel(const __half *src, float *dst, const size_t count) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        dst[idx] = __half2float(src[idx]);
    }

    __global__ void FuseWeightsKernel(
        const float *weight_ih, // (4H, I) row-major
        const float *weight_hh, // (4H, H) row-major
        __half *weight_cat, // (I+H, 4H) column-major
        const size_t input_size,
        const size_t hidden_size
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
        const float *bias_ih, // (4H,)
        const float *bias_hh, // (4H,)
        float *bias_out, // (4H,)
        const size_t hidden_size
    ) {
        const size_t gate_dim = 4 * hidden_size;
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= gate_dim) {
            return;
        }
        bias_out[idx] = bias_ih[idx] + bias_hh[idx];
    }

    __global__ void BackwardPointwiseKernel(
        const __half *dY_row, // (B, H) row-major half
        const float *dh_next_row, // (B, H) row-major
        const float *dc_next_row, // (B, H) row-major
        const float *gate_cache_row, // (B, 4H) row-major (i,f,g,o)
        const __half *y_row, // (B, H) row-major half
        float *dG_col_step, // (4H, B) column-major
        __half *dG_half_col_step, // (4H, B) column-major half
        float *dh_point_prev_col, // (H, B) column-major
        float *dc_prev_row, // (B, H) row-major
        const size_t batch_size,
        const size_t hidden_size
    ) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t total = batch_size * hidden_size;
        if (idx >= total) {
            return;
        }
        const size_t batch_idx = idx / hidden_size;
        const size_t hidden_idx = idx % hidden_size;
        const size_t gate_dim = 4 * hidden_size;

        const float dh_total = __half2float(dY_row[idx]) + dh_next_row[idx];
        const float dc_next = dc_next_row[idx];

        const float i_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 0 * hidden_size];
        const float f_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 1 * hidden_size];
        const float g_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 2 * hidden_size];
        const float o_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 3 * hidden_size];

        const float y_t = __half2float(y_row[idx]);
        constexpr float kEps = 1e-6f;

        const float denom_o = fabsf(o_gate) < kEps ? (o_gate >= 0.0f ? kEps : -kEps) : o_gate;
        float tanh_c = y_t / denom_o;
        tanh_c = fmaxf(fminf(tanh_c, 1.0f - kEps), -1.0f + kEps);
        const float c_t = atanhf(tanh_c);

        const float denom_f = fabsf(f_gate) < kEps ? (f_gate >= 0.0f ? kEps : -kEps) : f_gate;
        const float c_prev = (c_t - i_gate * g_gate) / denom_f;

        const float do_gate = dh_total * tanh_c;
        const float one_minus_tanh_sq = 1.0f - tanh_c * tanh_c;
        const float dc_total = dc_next + dh_total * o_gate * one_minus_tanh_sq;

        const float di = dc_total * g_gate;
        const float df = dc_total * c_prev;
        const float dg = dc_total * i_gate;
        const float dc_prev = dc_total * f_gate;

        const float dai = di * i_gate * (1.0f - i_gate);
        const float daf = df * f_gate * (1.0f - f_gate);
        const float dag = dg * (1.0f - g_gate * g_gate);
        const float dao = do_gate * o_gate * (1.0f - o_gate);

        dG_col_step[hidden_idx + batch_idx * gate_dim] = dai;
        dG_col_step[hidden_idx + hidden_size + batch_idx * gate_dim] = daf;
        dG_col_step[hidden_idx + 2 * hidden_size + batch_idx * gate_dim] = dag;
        dG_col_step[hidden_idx + 3 * hidden_size + batch_idx * gate_dim] = dao;
        if (dG_half_col_step != nullptr) {
            const size_t base = batch_idx * gate_dim;
            dG_half_col_step[hidden_idx + base] = __float2half(dai);
            dG_half_col_step[hidden_idx + hidden_size + base] = __float2half(daf);
            dG_half_col_step[hidden_idx + 2 * hidden_size + base] = __float2half(dag);
            dG_half_col_step[hidden_idx + 3 * hidden_size + base] = __float2half(dao);
        }

        dh_point_prev_col[hidden_idx + batch_idx * hidden_size] = 0.0f; // initialise accumulator
        dc_prev_row[idx] = dc_prev;
    }

    __global__ void PackInputStepKernel(
        const __half *x_step, // (B, I) row-major
        float *z_step, // (I+H, B) column-major float staging
        const size_t batch_size,
        const size_t input_size,
        const size_t hidden_size
    ) {
        const size_t total = batch_size * input_size;
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        const size_t tmp = idx;
        const size_t input_idx = tmp % input_size;
        const size_t batch_idx = tmp / input_size;
        const size_t z_rows = input_size + hidden_size;
        z_step[input_idx + batch_idx * z_rows] = __half2float(x_step[idx]);
    }

    __global__ void PackHiddenStateKernel(
        const float *h_prev, // (B, H) row-major
        float *z_step, // (I+H, B) column-major float staging
        const size_t batch_size,
        const size_t input_size,
        const size_t hidden_size
    ) {
        const size_t total = batch_size * hidden_size;
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        const size_t batch_idx = idx / hidden_size;
        const size_t hidden_idx = idx % hidden_size;
        const size_t z_rows = input_size + hidden_size;
        const float h_value = h_prev[idx];
        z_step[input_size + hidden_idx + batch_idx * z_rows] = h_value;
    }

    constexpr float kFp16SafeMax = 60000.0f;

    __global__ void ScaleAndPackColumnsKernel(
        const float *z_cols_float, // (I+H, B) column-major float
        __half *z_cols_half, // (I+H, B) column-major half
        float *column_scale, // (B,) scaling factors
        size_t z_rows,
        size_t batch_size
    ) {
        const size_t batch_idx = blockIdx.x;
        if (batch_idx >= batch_size) {
            return;
        }
        extern __shared__ float shared[];
        const float *src = z_cols_float + batch_idx * z_rows;
        __half *dst = z_cols_half + batch_idx * z_rows;

        float local_max = 0.0f;
        for (size_t row = threadIdx.x; row < z_rows; row += blockDim.x) {
            const float value = src[row];
            local_max = fmaxf(local_max, fabsf(value));
        }
        shared[threadIdx.x] = local_max;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        float inv_scale = 1.0f;
        if (threadIdx.x == 0) {
            const float max_abs = shared[0];
            float scale = 1.0f;
            if (max_abs > 0.0f) {
                scale = max_abs / kFp16SafeMax;
                if (!(scale > 0.0f)) {
                    scale = 1.0f;
                }
                inv_scale = 1.0f / scale;
            } else {
                inv_scale = 1.0f;
            }
            column_scale[batch_idx] = 1.0f / inv_scale;
            shared[0] = inv_scale;
        }
        __syncthreads();
        inv_scale = shared[0];

        for (size_t row = threadIdx.x; row < z_rows; row += blockDim.x) {
            float scaled = src[row] * inv_scale;
            scaled = fminf(fmaxf(scaled, -kFp16SafeMax), kFp16SafeMax);
            dst[row] = __float2half(scaled);
        }
    }

    __global__ void RecomputePointwiseKernel(
        const float *gate_col, // (4H, B) column-major
        const float *bias, // (4H,)
        const float *c_prev, // (B, H) row-major
        float *h_next, // (B, H) row-major
        float *c_next, // (B, H) row-major
        float *gate_out, // (B, 4H) row-major or nullptr
        const float *column_scale, // (B,) scaling factors for this step
        const size_t batch_size,
        const size_t hidden_size
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

        const float scale = column_scale[col];
        const float gi = gate_col[row_base + col * gate_dim] * scale + bias[row_base + 0 * hidden_size];
        const float gf =
            gate_col[row_base + hidden_size + col * gate_dim] * scale + bias[row_base + 1 * hidden_size];
        const float gg =
            gate_col[row_base + 2 * hidden_size + col * gate_dim] * scale + bias[row_base + 2 * hidden_size];
        const float go =
            gate_col[row_base + 3 * hidden_size + col * gate_dim] * scale + bias[row_base + 3 * hidden_size];

        const float i_gate = 1.0f / (1.0f + expf(-gi));
        const float f_gate = 1.0f / (1.0f + expf(-gf));
        const float g_gate = tanhf(gg);
        const float c_prev_val = c_prev[idx];
        const float c_val = f_gate * c_prev_val + i_gate * g_gate;
        const float o_gate = 1.0f / (1.0f + expf(-go));
        const float h_val = o_gate * tanhf(c_val);

        h_next[idx] = h_val;
        c_next[idx] = c_val;

        if (gate_out != nullptr) {
            float *gate_ptr = gate_out + batch_idx * gate_dim;
            gate_ptr[hidden_idx + 0 * hidden_size] = i_gate;
            gate_ptr[hidden_idx + 1 * hidden_size] = f_gate;
            gate_ptr[hidden_idx + 2 * hidden_size] = g_gate;
            gate_ptr[hidden_idx + 3 * hidden_size] = o_gate;
        }
    }

    __global__ void ConvertDxChunkToHalfKernel(
        const float *dx_col, // (I, chunk_tb) column-major
        __half *dx_half, // (chunk_steps, B, I) row-major
        const size_t steps,
        const size_t batch_size,
        const size_t input_size
    ) {
        const size_t total = steps * batch_size * input_size;
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        size_t tmp = idx;
        const size_t input_idx = tmp % input_size;
        tmp /= input_size;
        const size_t batch_idx = tmp % batch_size;
        const size_t step_idx = tmp / batch_size;

        const size_t column = step_idx * batch_size + batch_idx;
        const float value = dx_col[input_idx + column * input_size];
        dx_half[idx] = __float2half(value);
    }

    __global__ void ConvertInputToZChunkKernel(
        const __half *x_src, // (chunk_steps, B, I)
        __half *z_chunk_col, // (I+H, chunk_tb) column-major
        const size_t chunk_steps,
        const size_t batch_size,
        const size_t input_size,
        const size_t hidden_size
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

        const size_t column = step_idx * batch_size + batch_idx;
        const size_t z_rows = input_size + hidden_size;
        z_chunk_col[input_idx + column * z_rows] = x_src[idx];
    }

    __global__ void FillHiddenPartForZChunkKernel(
        const __half *y_chunk, // (chunk_steps, B, H) row-major
        const __half *first_prev, // (B, H) row-major
        const size_t chunk_steps,
        const size_t batch_size,
        const size_t hidden_size,
        const size_t input_size,
        __half *z_chunk_col // (I+H, chunk_tb) column-major
    ) {
        const size_t total = chunk_steps * batch_size * hidden_size;
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }

        size_t tmp = idx;
        const size_t hidden_idx = tmp % hidden_size;
        tmp /= hidden_size;
        const size_t batch_idx = tmp % batch_size;
        const size_t step_idx = tmp / batch_size;

        const size_t column = step_idx * batch_size + batch_idx;
        const size_t z_rows = input_size + hidden_size;

        __half h_prev{};
        if (step_idx == 0) {
            h_prev = first_prev[batch_idx * hidden_size + hidden_idx];
        } else {
            const __half *y_prev = y_chunk + ((step_idx - 1) * batch_size + batch_idx) * hidden_size;
            h_prev = y_prev[hidden_idx];
        }

        z_chunk_col[input_size + hidden_idx + column * z_rows] = h_prev;
    }

    __global__ void FillOnesKernel(__half *dst, const size_t count) {
        if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count) {
            dst[idx] = __float2half(1.0f);
        }
    }

    // Holds references shared between copy slots so double-buffered H2D transfers stay readable.
    struct ChunkCopyParams {
        size_t time_steps;
        size_t chunk_capacity;
        size_t recompute_stride;
        size_t batch_size;
        size_t gate_dim;
        size_t input_size;
        size_t bh_elements;
        flstm::GateCacheHost checkpoint_cache_host;
        flstm::StreamingLstmOptions options;
        int threads;
        const __half *x_tensor_host;
        const __half *y_tensor_host;
        const __half *dY_tensor_host;
        DeviceBuffer<float> *checkpoint_half;
        DeviceBuffer<__half> *checkpoint_h_half_quantized;
        DeviceBuffer<__half> *checkpoint_c_half_quantized;
        DeviceBuffer<__half> *x_chunk_half;
        DeviceBuffer<__half> *y_chunk_half;
        DeviceBuffer<__half> *dY_chunk_half;
        DeviceBuffer<__half> *y_prev_half;
        CudaEvent *h2d_ready;
        cudaStream_t h2d_stream;
        bool *compute_done_valid;
        CudaEvent *compute_done;
        size_t *chunk_ids;
        size_t *prefix_steps;
        size_t *recompute_steps;
    };

    // Schedules host-to-device copies for one chunk slot, waiting on outstanding compute if needed.
    size_t IssueChunkCopy(const ssize_t chunk_id, const int slot, const ChunkCopyParams &params) {
        if (chunk_id < 0) {
            params.chunk_ids[slot] = static_cast<size_t>(-1);
            params.prefix_steps[slot] = 0;
            params.recompute_steps[slot] = 0;
            return 0;
        }

        const size_t chunk_start = static_cast<size_t>(chunk_id) * params.chunk_capacity;
        if (chunk_start >= params.time_steps) {
            params.chunk_ids[slot] = static_cast<size_t>(-1);
            params.prefix_steps[slot] = 0;
            params.recompute_steps[slot] = 0;
            return 0;
        }

        if (params.compute_done_valid[slot]) {
            CheckCuda(cudaStreamWaitEvent(params.h2d_stream, params.compute_done[slot].evt, 0),
                      "wait compute_done before reuse");
            params.compute_done_valid[slot] = false;
        }

        const size_t steps = std::min(params.chunk_capacity, params.time_steps - chunk_start);
        const size_t checkpoint_stride = params.recompute_stride;
        const size_t checkpoint_step = (chunk_start / checkpoint_stride) * checkpoint_stride;
        const size_t checkpoint_index = checkpoint_stride == 0 ? 0 : (checkpoint_step / checkpoint_stride);
        const size_t prefix_steps = chunk_start - checkpoint_step;
        const size_t recompute_steps = prefix_steps + steps;

        const size_t x_elems = recompute_steps * params.batch_size * params.input_size;
        if (x_elems > 0 && params.x_chunk_half[slot].ptr != nullptr) {
            const __half *x_src = params.x_tensor_host + checkpoint_step * params.batch_size * params.input_size;
            CheckCuda(cudaMemcpyAsync(
                          params.x_chunk_half[slot].ptr,
                          x_src,
                          x_elems * sizeof(__half),
                          cudaMemcpyHostToDevice,
                          params.h2d_stream),
                      "copy x chunk (recompute span)");
        }

        const size_t y_elems = steps * params.bh_elements;
        if (y_elems > 0 && params.y_chunk_half[slot].ptr != nullptr) {
            CheckCuda(cudaMemcpyAsync(
                          params.y_chunk_half[slot].ptr,
                          params.y_tensor_host + chunk_start * params.bh_elements,
                          y_elems * sizeof(__half),
                          cudaMemcpyHostToDevice,
                          params.h2d_stream),
                      "copy y chunk");
        }
        if (y_elems > 0 && params.dY_chunk_half[slot].ptr != nullptr) {
            CheckCuda(cudaMemcpyAsync(
                          params.dY_chunk_half[slot].ptr,
                          params.dY_tensor_host + chunk_start * params.bh_elements,
                          y_elems * sizeof(__half),
                          cudaMemcpyHostToDevice,
                          params.h2d_stream),
                      "copy dY chunk");
        }

        if (chunk_start > 0 && params.y_prev_half[slot].ptr != nullptr) {
            CheckCuda(cudaMemcpyAsync(
                          params.y_prev_half[slot].ptr,
                          params.y_tensor_host + (chunk_start - 1) * params.bh_elements,
                          params.bh_elements * sizeof(__half),
                          cudaMemcpyHostToDevice,
                          params.h2d_stream),
                      "copy y_prev");
        }

        if (params.checkpoint_cache_host.h_ptr != nullptr && params.checkpoint_cache_host.c_ptr != nullptr &&
            params.checkpoint_half[slot].ptr != nullptr) {
            const size_t checkpoint_offset = checkpoint_index * params.bh_elements;
            float *checkpoint_dst_h = params.checkpoint_half[slot].ptr;
            float *checkpoint_dst_c = params.checkpoint_half[slot].ptr + params.bh_elements;

            if (params.options.h_dtype == flstm::GateCacheDType::kFloat32) {
                const float *checkpoint_src_h =
                        reinterpret_cast<const float *>(params.checkpoint_cache_host.h_ptr) + checkpoint_offset;
                CheckCuda(cudaMemcpyAsync(
                              checkpoint_dst_h,
                              checkpoint_src_h,
                              params.bh_elements * sizeof(float),
                              cudaMemcpyHostToDevice,
                              params.h2d_stream),
                          "copy checkpoint h state");
            } else {
                __half *quantized = params.checkpoint_h_half_quantized[slot].ptr;
                const __half *checkpoint_src_h =
                        reinterpret_cast<const __half *>(params.checkpoint_cache_host.h_ptr) + checkpoint_offset;
                CheckCuda(cudaMemcpyAsync(
                              quantized,
                              checkpoint_src_h,
                              params.bh_elements * sizeof(__half),
                              cudaMemcpyHostToDevice,
                              params.h2d_stream),
                          "copy checkpoint h state half");
                const int convert_blocks = BlocksFor(params.bh_elements, params.threads);
                HalfToFloatKernel<<<convert_blocks, params.threads, 0, params.h2d_stream>>>(
                    quantized,
                    checkpoint_dst_h,
                    params.bh_elements
                );
                CheckCuda(cudaGetLastError(), "HalfToFloatKernel checkpoint h");
            }

            if (params.options.c_dtype == flstm::GateCacheDType::kFloat32) {
                const float *checkpoint_src_c =
                        reinterpret_cast<const float *>(params.checkpoint_cache_host.c_ptr) + checkpoint_offset;
                CheckCuda(cudaMemcpyAsync(
                              checkpoint_dst_c,
                              checkpoint_src_c,
                              params.bh_elements * sizeof(float),
                              cudaMemcpyHostToDevice,
                              params.h2d_stream),
                          "copy checkpoint c state");
            } else {
                __half *quantized = params.checkpoint_c_half_quantized[slot].ptr;
                const __half *checkpoint_src_c =
                        reinterpret_cast<const __half *>(params.checkpoint_cache_host.c_ptr) + checkpoint_offset;
                CheckCuda(cudaMemcpyAsync(
                              quantized,
                              checkpoint_src_c,
                              params.bh_elements * sizeof(__half),
                              cudaMemcpyHostToDevice,
                              params.h2d_stream),
                          "copy checkpoint c state half");
                const int convert_blocks = BlocksFor(params.bh_elements, params.threads);
                HalfToFloatKernel<<<convert_blocks, params.threads, 0, params.h2d_stream>>>(
                    quantized,
                    checkpoint_dst_c,
                    params.bh_elements
                );
                CheckCuda(cudaGetLastError(), "HalfToFloatKernel checkpoint c");
            }
        }

        CheckCuda(cudaEventRecord(params.h2d_ready[slot].evt, params.h2d_stream), "record h2d_ready");
        params.chunk_ids[slot] = static_cast<size_t>(chunk_id);
        params.prefix_steps[slot] = prefix_steps;
        params.recompute_steps[slot] = recompute_steps;
        return steps;
    }
} // namespace

namespace flstm {
    void StreamingLstmBackward(
        size_t time_steps,
        size_t batch_size,
        size_t input_size,
        size_t hidden_size,
        size_t recompute_interval,

        const __half *x_tensor_host,
        const __half *y_tensor_host,
        GateCacheHost gate_cache_host,
        StreamingLstmOptions options,

        const __half *dY_tensor_host,
        const __half *d_hn_device,
        const __half *d_cn_device,
        const __half *h0_device,
        const __half *c0_device,

        const float *weights_ih,
        const float *weights_hh,
        const float *bias_ih,
        const float *bias_hh,

        __half *dx_tensor_host,
        float *dW_ih,
        float *dW_hh,
        float *db_ih,
        float *db_hh,
        float *dh0_out,
        float *dc0_out,

        cudaStream_t compute_stream,
        cudaStream_t h2d_stream,
        cudaStream_t d2h_stream
    ) {
        GPUTX_RANGE("StreamingLstmBackward");
        mfu::Profiler profiler("backward");
        if (time_steps == 0 || batch_size == 0 || input_size == 0 || hidden_size == 0) {
            return;
        }
        ValidateGateCacheOptions(options);
        if (recompute_interval == 0) {
            throw std::runtime_error("StreamingLstmBackward requires recompute_interval >= 1");
        }
        if (x_tensor_host == nullptr || y_tensor_host == nullptr || gate_cache_host.h_ptr == nullptr ||
            gate_cache_host.c_ptr == nullptr ||
            dY_tensor_host == nullptr || h0_device == nullptr || c0_device == nullptr) {
            throw std::runtime_error("StreamingLstmBackward requires forward caches");
        }

        (void) c0_device;

        if (compute_stream == h2d_stream || compute_stream == d2h_stream || h2d_stream == d2h_stream) {
            throw std::runtime_error("StreamingLstmBackward requires distinct compute/h2d/d2h streams");
        }

        const size_t gate_dim = 4 * hidden_size;
        const size_t z_rows = input_size + hidden_size;
        const size_t bh_elements = batch_size * hidden_size;
        constexpr size_t kChunkSteps = 32;

        DeviceBuffer<float> d_hn_float;
        DeviceBuffer<float> d_cn_float;
        DeviceBuffer<float> dh_cur;
        DeviceBuffer<float> dh_tmp;
        DeviceBuffer<float> dc_cur;
        DeviceBuffer<float> dc_tmp;
        DeviceBuffer<__half> weight_cat_col;
        DeviceBuffer<float> db_buffer;
        DeviceBuffer<float> bias_fused;
        DeviceBuffer<__half> ones_vec;

        const int threads = 256;

        AllocateDeviceBuffer(d_hn_float, bh_elements, "cudaMalloc d_hn_float");
        AllocateDeviceBuffer(d_cn_float, bh_elements, "cudaMalloc d_cn_float");
        const int bh_blocks = BlocksFor(bh_elements, threads);
        if (d_hn_device != nullptr) {
            HalfToFloatKernel<<<bh_blocks, threads, 0, compute_stream>>>(d_hn_device, d_hn_float.ptr, bh_elements);
            CheckCuda(cudaGetLastError(), "HalfToFloatKernel d_hn");
        } else {
            ZeroDeviceMemory(d_hn_float.ptr, bh_elements, compute_stream, "memset d_hn");
        }
        if (d_cn_device != nullptr) {
            HalfToFloatKernel<<<bh_blocks, threads, 0, compute_stream>>>(d_cn_device, d_cn_float.ptr, bh_elements);
            CheckCuda(cudaGetLastError(), "HalfToFloatKernel d_cn");
        } else {
            ZeroDeviceMemory(d_cn_float.ptr, bh_elements, compute_stream, "memset d_cn");
        }

        AllocateDeviceBuffer(dh_cur, bh_elements, "cudaMalloc dh_cur");
        AllocateDeviceBuffer(dh_tmp, bh_elements, "cudaMalloc dh_tmp");
        AllocateDeviceBuffer(dc_cur, bh_elements, "cudaMalloc dc_cur");
        AllocateDeviceBuffer(dc_tmp, bh_elements, "cudaMalloc dc_tmp");
        CheckCuda(cudaMemcpyAsync(
                      dh_cur.ptr,
                      d_hn_float.ptr,
                      bh_elements * sizeof(float),
                      cudaMemcpyDeviceToDevice,
                      compute_stream),
                  "copy d_hn");
        CheckCuda(cudaMemcpyAsync(
                      dc_cur.ptr,
                      d_cn_float.ptr,
                      bh_elements * sizeof(float),
                      cudaMemcpyDeviceToDevice,
                      compute_stream),
                  "copy d_cn");

        const size_t weight_elems = z_rows * gate_dim;
        AllocateDeviceBuffer(weight_cat_col, weight_elems, "cudaMalloc weight_cat_col");
        const int weight_blocks = BlocksFor(weight_elems, threads);
        FuseWeightsKernel<<<weight_blocks, threads, 0, compute_stream>>>(
            weights_ih,
            weights_hh,
            weight_cat_col.ptr,
            input_size,
            hidden_size
        );
        CheckCuda(cudaGetLastError(), "FuseWeightsKernel");

        AllocateDeviceBuffer(db_buffer, gate_dim, "cudaMalloc db_buffer");
        ZeroDeviceMemory(db_buffer.ptr, gate_dim, compute_stream, "memset db buffer");

        constexpr size_t chunk_capacity = kChunkSteps;
        const size_t recompute_stride = recompute_interval;
        const size_t max_chunk_window = chunk_capacity + recompute_stride - 1;
        const size_t chunk_gate_capacity = chunk_capacity * batch_size * gate_dim;
        const size_t chunk_input_capacity = max_chunk_window * batch_size * input_size;
        const size_t chunk_hidden_capacity = chunk_capacity * batch_size * hidden_size;
        const size_t chunk_tb_capacity = chunk_capacity * batch_size;

        AllocateDeviceBuffer(ones_vec, chunk_tb_capacity, "cudaMalloc ones_vec");
        const int ones_blocks = BlocksFor(chunk_tb_capacity, threads);
        FillOnesKernel<<<ones_blocks, threads, 0, compute_stream>>>(ones_vec.ptr, chunk_tb_capacity);
        CheckCuda(cudaGetLastError(), "FillOnesKernel");

        DeviceBuffer<__half> x_chunk_half[2];
        DeviceBuffer<__half> y_chunk_half[2];
        DeviceBuffer<__half> dY_chunk_half[2];
        DeviceBuffer<float> gate_chunk_float[2];
        DeviceBuffer<__half> z_chunk_col[2];
        DeviceBuffer<float> dG_chunk_col[2];
        DeviceBuffer<__half> dG_chunk_half[2];
        DeviceBuffer<float> dX_chunk_col[2];
        DeviceBuffer<__half> dx_chunk_half[2];
        DeviceBuffer<__half> y_prev_half[2];
        DeviceBuffer<float> checkpoint_half[2];
        DeviceBuffer<__half> checkpoint_h_half_quantized[2];
        DeviceBuffer<__half> checkpoint_c_half_quantized[2];
        DeviceBuffer<float> recompute_h_prev;
        DeviceBuffer<float> recompute_h_next;
        DeviceBuffer<float> recompute_c_prev;
        DeviceBuffer<float> recompute_c_next;
        DeviceBuffer<float> z_step_float;
        DeviceBuffer<__half> z_step_half;
        DeviceBuffer<float> column_scale_tmp;
        DeviceBuffer<float> gate_pre_col;

        const size_t z_chunk_elements = z_rows * chunk_tb_capacity;
        const size_t dX_chunk_elements = input_size * chunk_tb_capacity;
        const size_t dG_chunk_elements = gate_dim * chunk_tb_capacity;
        const size_t checkpoint_half_elements = 2 * bh_elements;
        const size_t z_step_elements = z_rows * batch_size;
        const size_t gate_step_elements = gate_dim * batch_size;

        AllocateDeviceBufferArray(gate_chunk_float, chunk_gate_capacity, "cudaMalloc gate_chunk_float");
        AllocateDeviceBufferArray(x_chunk_half, chunk_input_capacity, "cudaMalloc x_chunk_half");
        AllocateDeviceBufferArray(z_chunk_col, z_chunk_elements, "cudaMalloc z_chunk_col");
        AllocateDeviceBufferArray(dX_chunk_col, dX_chunk_elements, "cudaMalloc dX_chunk_col");
        AllocateDeviceBufferArray(dx_chunk_half, chunk_input_capacity, "cudaMalloc dx_chunk_half");
        AllocateDeviceBufferArray(y_chunk_half, chunk_hidden_capacity, "cudaMalloc y_chunk_half");
        AllocateDeviceBufferArray(dY_chunk_half, chunk_hidden_capacity, "cudaMalloc dY_chunk_half");
        AllocateDeviceBufferArray(dG_chunk_col, dG_chunk_elements, "cudaMalloc dG_chunk_col");
        AllocateDeviceBufferArray(dG_chunk_half, dG_chunk_elements, "cudaMalloc dG_chunk_half");
        AllocateDeviceBufferArray(y_prev_half, bh_elements, "cudaMalloc y_prev_half");
        AllocateDeviceBufferArray(checkpoint_half, checkpoint_half_elements, "cudaMalloc checkpoint_half");
        const bool checkpoint_h_is_half = (options.h_dtype == flstm::GateCacheDType::kFloat16);
        const bool checkpoint_c_is_half = (options.c_dtype == flstm::GateCacheDType::kFloat16);
        if (checkpoint_h_is_half) {
            AllocateDeviceBufferArray(checkpoint_h_half_quantized, bh_elements, "cudaMalloc checkpoint_h_quantized");
        }
        if (checkpoint_c_is_half) {
            AllocateDeviceBufferArray(checkpoint_c_half_quantized, bh_elements, "cudaMalloc checkpoint_c_quantized");
        }
        AllocateDeviceBuffer(bias_fused, gate_dim, "cudaMalloc bias_fused");
        AllocateDeviceBuffer(recompute_h_prev, bh_elements, "cudaMalloc recompute_h_prev");
        AllocateDeviceBuffer(recompute_h_next, bh_elements, "cudaMalloc recompute_h_next");
        AllocateDeviceBuffer(recompute_c_prev, bh_elements, "cudaMalloc recompute_c_prev");
        AllocateDeviceBuffer(recompute_c_next, bh_elements, "cudaMalloc recompute_c_next");
        AllocateDeviceBuffer(z_step_float, z_step_elements, "cudaMalloc z_step_float");
        AllocateDeviceBuffer(z_step_half, z_step_elements, "cudaMalloc z_step_half");
        AllocateDeviceBuffer(column_scale_tmp, batch_size, "cudaMalloc column_scale_tmp");
        AllocateDeviceBuffer(gate_pre_col, gate_step_elements, "cudaMalloc gate_pre_col");

        const int bias_blocks = BlocksFor(gate_dim, threads);
        FuseBiasKernel<<<bias_blocks, threads, 0, compute_stream>>>(
            bias_ih,
            bias_hh,
            bias_fused.ptr,
            hidden_size
        );
        CheckCuda(cudaGetLastError(), "FuseBiasKernel");

        ZeroDeviceMemory(dW_ih, static_cast<size_t>(gate_dim) * input_size, compute_stream, "memset dW_ih");
        ZeroDeviceMemory(dW_hh, static_cast<size_t>(gate_dim) * hidden_size, compute_stream, "memset dW_hh");

        const float alpha = 1.0f;
        const float beta_zero = 0.0f;
        const float beta_one = 1.0f;
        const int point_blocks = BlocksFor(bh_elements, threads);

        CudaEvent h2d_ready[2];
        CudaEvent compute_done[2];
        CudaEvent d2h_done[2];
        CudaEvent dx_ready[2];
        bool compute_done_valid[2] = {false, false};
        bool d2h_done_valid[2] = {false, false};
        size_t chunk_steps[2] = {0, 0};
        size_t chunk_ids[2] = {static_cast<size_t>(-1), static_cast<size_t>(-1)};
        size_t chunk_prefix_steps[2] = {0, 0};
        size_t chunk_recompute_steps[2] = {0, 0};

        // Shared metadata for the double-buffered copy helper.
        ChunkCopyParams chunk_params{
            .time_steps = time_steps,
            .chunk_capacity = chunk_capacity,
            .recompute_stride = recompute_stride,
            .batch_size = batch_size,
            .gate_dim = gate_dim,
            .input_size = input_size,
            .bh_elements = bh_elements,
            .checkpoint_cache_host = gate_cache_host,
            .options = options,
            .threads = threads,
            .x_tensor_host = x_tensor_host,
            .y_tensor_host = y_tensor_host,
            .dY_tensor_host = dY_tensor_host,
            .checkpoint_half = checkpoint_half,
            .checkpoint_h_half_quantized = checkpoint_h_half_quantized,
            .checkpoint_c_half_quantized = checkpoint_c_half_quantized,
            .x_chunk_half = x_chunk_half,
            .y_chunk_half = y_chunk_half,
            .dY_chunk_half = dY_chunk_half,
            .y_prev_half = y_prev_half,
            .h2d_ready = h2d_ready,
            .h2d_stream = h2d_stream,
            .compute_done_valid = compute_done_valid,
            .compute_done = compute_done,
            .chunk_ids = chunk_ids,
            .prefix_steps = chunk_prefix_steps,
            .recompute_steps = chunk_recompute_steps
        };

        const size_t total_chunks = (time_steps + chunk_capacity - 1) / chunk_capacity;

        ssize_t next_chunk = static_cast<ssize_t>(total_chunks) - 1;
        for (int pre = 0; pre < static_cast<int>(std::min<size_t>(total_chunks, 2)); ++pre) {
            const int slot = pre % 2;
            chunk_steps[slot] = IssueChunkCopy(next_chunk, slot, chunk_params);
            --next_chunk;
        }

        size_t processed_chunks = 0;
        while (processed_chunks < total_chunks) {
            int slot = -1;
            if (chunk_ids[0] != static_cast<size_t>(-1) && chunk_ids[1] != static_cast<size_t>(-1)) {
                slot = (chunk_ids[0] > chunk_ids[1]) ? 0 : 1;
            } else if (chunk_ids[0] != static_cast<size_t>(-1)) {
                slot = 0;
            } else if (chunk_ids[1] != static_cast<size_t>(-1)) {
                slot = 1;
            } else {
                break;
            }

            const size_t steps_in_chunk = chunk_steps[slot];
            if (steps_in_chunk == 0) {
                chunk_ids[slot] = static_cast<size_t>(-1);
                continue;
            }

            if (d2h_done_valid[slot]) {
                CheckCuda(cudaStreamWaitEvent(compute_stream, d2h_done[slot].evt, 0), "wait d2h_done before compute");
                d2h_done_valid[slot] = false;
            }

            CheckCuda(cudaStreamWaitEvent(compute_stream, h2d_ready[slot].evt, 0), "wait h2d_ready");

            const size_t chunk_start_step = chunk_ids[slot] * chunk_capacity;
            const size_t chunk_tb = steps_in_chunk * batch_size;
            const size_t prefix_steps = chunk_prefix_steps[slot];
            const size_t recompute_steps = chunk_recompute_steps[slot];

            const size_t chunk_hidden_elems = steps_in_chunk * bh_elements;

            const size_t chunk_input_elems = steps_in_chunk * batch_size * input_size;
            if (chunk_input_elems > 0 && x_chunk_half[slot].ptr != nullptr) {
                const int convert_blocks = BlocksFor(chunk_input_elems, threads);
                const __half *x_grad_src = x_chunk_half[slot].ptr + prefix_steps * batch_size * input_size;
                ConvertInputToZChunkKernel<<<convert_blocks, threads, 0, compute_stream>>>(
                    x_grad_src,
                    z_chunk_col[slot].ptr,
                    steps_in_chunk,
                    batch_size,
                    input_size,
                    hidden_size
                );
                CheckCuda(cudaGetLastError(), "ConvertInputToZChunkKernel");
            }

            const __half *first_prev_ptr = nullptr;
            if (chunk_start_step == 0) {
                first_prev_ptr = h0_device;
            } else {
                first_prev_ptr = y_prev_half[slot].ptr;
            }

            if (chunk_hidden_elems > 0) {
                const int hidden_blocks = BlocksFor(chunk_hidden_elems, threads);
                FillHiddenPartForZChunkKernel<<<hidden_blocks, threads, 0, compute_stream>>>(
                    y_chunk_half[slot].ptr,
                    first_prev_ptr,
                    steps_in_chunk,
                    batch_size,
                    hidden_size,
                    input_size,
                    z_chunk_col[slot].ptr
                );
                CheckCuda(cudaGetLastError(), "FillHiddenPartForZChunkKernel");
            }

            if (recompute_steps > 0) {
                CheckCuda(cudaMemcpyAsync(
                              recompute_h_prev.ptr,
                              checkpoint_half[slot].ptr,
                              bh_elements * sizeof(float),
                              cudaMemcpyDeviceToDevice,
                              compute_stream),
                          "copy checkpoint h");
                CheckCuda(cudaMemcpyAsync(
                              recompute_c_prev.ptr,
                              checkpoint_half[slot].ptr + bh_elements,
                              bh_elements * sizeof(float),
                              cudaMemcpyDeviceToDevice,
                              compute_stream),
                          "copy checkpoint c");

                for (size_t local_step = 0; local_step < recompute_steps; ++local_step) {
                    const __half *x_step_half =
                            x_chunk_half[slot].ptr + local_step * batch_size * input_size;
                    const int input_blocks = BlocksFor(batch_size * input_size, threads);
                    PackInputStepKernel<<<input_blocks, threads, 0, compute_stream>>>(
                        x_step_half,
                        z_step_float.ptr,
                        batch_size,
                        input_size,
                        hidden_size
                    );
                    CheckCuda(cudaGetLastError(), "PackInputStepKernel");
                    PackHiddenStateKernel<<<point_blocks, threads, 0, compute_stream>>>(
                        recompute_h_prev.ptr,
                        z_step_float.ptr,
                        batch_size,
                        input_size,
                        hidden_size
                    );
                    CheckCuda(cudaGetLastError(), "PackHiddenStateKernel");
                    const int scale_blocks = static_cast<int>(batch_size);
                    const size_t shared_bytes = threads * sizeof(float);
                    ScaleAndPackColumnsKernel<<<scale_blocks, threads, shared_bytes, compute_stream>>>(
                        z_step_float.ptr,
                        z_step_half.ptr,
                        column_scale_tmp.ptr,
                        z_rows,
                        batch_size
                    );
                    CheckCuda(cudaGetLastError(), "ScaleAndPackColumnsKernel recompute");
                    flstm::GemmTN(
                        static_cast<int>(gate_dim),
                        static_cast<int>(batch_size),
                        static_cast<int>(z_rows),
                        weight_cat_col.ptr,
                        static_cast<int>(z_rows),
                        z_step_half.ptr,
                        static_cast<int>(z_rows),
                        gate_pre_col.ptr,
                        static_cast<int>(gate_dim),
                        alpha,
                        beta_zero,
                        compute_stream
                    );
                    profiler.AddTotal(mfu::GemmFlops(gate_dim, batch_size, z_rows));
                    float *gate_out_ptr = nullptr;
                    if (local_step >= prefix_steps) {
                        const size_t chunk_local = local_step - prefix_steps;
                        gate_out_ptr = gate_chunk_float[slot].ptr + chunk_local * batch_size * gate_dim;
                    }
                    RecomputePointwiseKernel<<<point_blocks, threads, 0, compute_stream>>>(
                        gate_pre_col.ptr,
                        bias_fused.ptr,
                        recompute_c_prev.ptr,
                        recompute_h_next.ptr,
                        recompute_c_next.ptr,
                        gate_out_ptr,
                        column_scale_tmp.ptr,
                        batch_size,
                        hidden_size
                    );
                    CheckCuda(cudaGetLastError(), "RecomputePointwiseKernel");
                    std::swap(recompute_h_prev.ptr, recompute_h_next.ptr);
                    std::swap(recompute_c_prev.ptr, recompute_c_next.ptr);
                }
            }

            for (int step = static_cast<int>(steps_in_chunk) - 1; step >= 0; --step) {
                const size_t local_offset = static_cast<size_t>(step) * batch_size;
                float *dG_step = dG_chunk_col[slot].ptr + local_offset * gate_dim;
                __half *dG_half_step = dG_chunk_half[slot].ptr + local_offset * gate_dim;
                float *dh_out = dh_tmp.ptr;
                float *dc_out = dc_tmp.ptr;

                const __half *dY_t = dY_chunk_half[slot].ptr + static_cast<size_t>(step) * bh_elements;
                const float *gate_step = gate_chunk_float[slot].ptr + static_cast<size_t>(step) * batch_size * gate_dim;
                const __half *y_step = y_chunk_half[slot].ptr + static_cast<size_t>(step) * bh_elements;

                BackwardPointwiseKernel<<<point_blocks, threads, 0, compute_stream>>>(
                    dY_t,
                    dh_cur.ptr,
                    dc_cur.ptr,
                    gate_step,
                    y_step,
                    dG_step,
                    dG_half_step,
                    dh_out,
                    dc_out,
                    batch_size,
                    hidden_size
                );
                CheckCuda(cudaGetLastError(), "BackwardPointwiseKernel");

                const __half *W_hh_block = weight_cat_col.ptr + input_size;
                flstm::GemmNN(
                    static_cast<int>(hidden_size),
                    static_cast<int>(batch_size),
                    static_cast<int>(gate_dim),
                    W_hh_block,
                    static_cast<int>(z_rows),
                    dG_half_step,
                    static_cast<int>(gate_dim),
                    dh_out,
                    static_cast<int>(hidden_size),
                    alpha,
                    beta_one,
                    compute_stream
                );
                profiler.AddUseful(mfu::GemmFlops(hidden_size, batch_size, gate_dim));

                std::swap(dh_cur.ptr, dh_tmp.ptr);
                std::swap(dc_cur.ptr, dc_tmp.ptr);
            }

            bool issued_dx_copy = false;
            size_t dx_chunk_elems = 0;
            const int zgemm_k = static_cast<int>(chunk_tb);
            if (zgemm_k > 0) {
                const __half *z_chunk_base = z_chunk_col[slot].ptr;
                const __half *x_chunk_matrix = z_chunk_base;
                const __half *h_chunk_matrix = z_chunk_base + input_size;
                const __half *dG_chunk_half_ptr = dG_chunk_half[slot].ptr;

                flstm::GemmNT(
                    static_cast<int>(input_size),
                    static_cast<int>(gate_dim),
                    zgemm_k,
                    x_chunk_matrix,
                    static_cast<int>(z_rows),
                    dG_chunk_half_ptr,
                    static_cast<int>(gate_dim),
                    dW_ih,
                    static_cast<int>(input_size),
                    alpha,
                    beta_one,
                    compute_stream
                );
                profiler.AddUseful(mfu::GemmFlops(input_size, gate_dim, static_cast<size_t>(zgemm_k)));

                if (hidden_size > 0) {
                    flstm::GemmNT(
                        static_cast<int>(hidden_size),
                        static_cast<int>(gate_dim),
                        zgemm_k,
                        h_chunk_matrix,
                        static_cast<int>(z_rows),
                        dG_chunk_half_ptr,
                        static_cast<int>(gate_dim),
                        dW_hh,
                        static_cast<int>(hidden_size),
                        alpha,
                        beta_one,
                        compute_stream
                    );
                    profiler.AddUseful(mfu::GemmFlops(hidden_size, gate_dim, static_cast<size_t>(zgemm_k)));
                }

                flstm::GemmNN(
                    static_cast<int>(gate_dim),
                    1,
                    zgemm_k,
                    dG_chunk_half_ptr,
                    static_cast<int>(gate_dim),
                    ones_vec.ptr,
                    zgemm_k,
                    db_buffer.ptr,
                    static_cast<int>(gate_dim),
                    alpha,
                    beta_one,
                    compute_stream
                );
                profiler.AddUseful(mfu::GemvFlops(gate_dim, static_cast<size_t>(zgemm_k)));

                flstm::GemmNN(
                    static_cast<int>(input_size),
                    zgemm_k,
                    static_cast<int>(gate_dim),
                    weight_cat_col.ptr,
                    static_cast<int>(z_rows),
                    dG_chunk_half_ptr,
                    static_cast<int>(gate_dim),
                    dX_chunk_col[slot].ptr,
                    static_cast<int>(input_size),
                    alpha,
                    beta_zero,
                    compute_stream
                );
                profiler.AddUseful(mfu::GemmFlops(input_size, static_cast<size_t>(zgemm_k), gate_dim));

                dx_chunk_elems = steps_in_chunk * batch_size * input_size;
                if (dx_chunk_elems > 0) {
                    const int dx_blocks = BlocksFor(dx_chunk_elems, threads);
                    ConvertDxChunkToHalfKernel<<<dx_blocks, threads, 0, compute_stream>>>(
                        dX_chunk_col[slot].ptr,
                        dx_chunk_half[slot].ptr,
                        steps_in_chunk,
                        batch_size,
                        input_size
                    );
                    CheckCuda(cudaGetLastError(), "ConvertDxChunkToHalfKernel");
                    issued_dx_copy = true;
                }
            }

            CheckCuda(cudaEventRecord(dx_ready[slot].evt, compute_stream), "record dx_ready");
            CheckCuda(cudaEventRecord(compute_done[slot].evt, compute_stream), "record compute_done");
            compute_done_valid[slot] = true;
            CheckCuda(cudaStreamWaitEvent(d2h_stream, dx_ready[slot].evt, 0), "wait dx_ready on d2h");
            if (issued_dx_copy) {
                CheckCuda(cudaMemcpyAsync(
                              dx_tensor_host + chunk_start_step * batch_size * input_size,
                              dx_chunk_half[slot].ptr,
                              dx_chunk_elems * sizeof(__half),
                              cudaMemcpyDeviceToHost,
                              d2h_stream),
                          "copy dx chunk -> host");
            }
            CheckCuda(cudaEventRecord(d2h_done[slot].evt, d2h_stream), "record d2h_done");
            d2h_done_valid[slot] = true;
            ++processed_chunks;

            if (next_chunk >= 0) {
                chunk_steps[slot] = IssueChunkCopy(next_chunk, slot, chunk_params);
                --next_chunk;
            } else {
                chunk_steps[slot] = 0;
                chunk_ids[slot] = static_cast<size_t>(-1);
            }
        }

        CheckCuda(cudaStreamSynchronize(h2d_stream), "final h2d sync");
        CheckCuda(cudaStreamSynchronize(d2h_stream), "final d2h sync");

        CheckCuda(cudaMemcpyAsync(
                      db_ih,
                      db_buffer.ptr,
                      gate_dim * sizeof(float),
                      cudaMemcpyDeviceToDevice,
                      compute_stream),
                  "copy db -> db_ih");
        CheckCuda(cudaMemcpyAsync(
                      db_hh,
                      db_buffer.ptr,
                      gate_dim * sizeof(float),
                      cudaMemcpyDeviceToDevice,
                      compute_stream),
                  "copy db -> db_hh");

        CheckCuda(cudaMemcpyAsync(
                      dh0_out,
                      dh_cur.ptr,
                      bh_elements * sizeof(float),
                      cudaMemcpyDeviceToDevice,
                      compute_stream),
                  "copy dh0");
        CheckCuda(cudaMemcpyAsync(
                      dc0_out,
                      dc_cur.ptr,
                      bh_elements * sizeof(float),
                      cudaMemcpyDeviceToDevice,
                      compute_stream),
                  "copy dc0");

        CheckCuda(cudaStreamSynchronize(compute_stream), "final backward sync");
        profiler.Finish();
    }
} // namespace flstm

extern "C" void flstm_StreamingLstmBackward(
    const size_t time_steps,
    const size_t batch_size,
    const size_t input_size,
    const size_t hidden_size,
    const size_t recompute_interval,

    const __half *x_tensor_host,
    const __half *y_tensor_host,
    flstm_GateCacheHost gate_cache_host,

    const __half *dY_tensor_host,
    const __half *d_hn_device,
    const __half *d_cn_device,
    const __half *h0_device,
    const __half *c0_device,

    const float *weights_ih,
    const float *weights_hh,
    const float *bias_ih,
    const float *bias_hh,

    __half *dx_tensor_host,
    float *dW_ih,
    float *dW_hh,
    float *db_ih,
    float *db_hh,
    float *dh0_out,
    float *dc0_out,

    const cudaStream_t compute_stream,
    const cudaStream_t h2d_stream,
    const cudaStream_t d2h_stream,
    const flstm_StreamingLstmOptions *options
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
        flstm::StreamingLstmBackward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            x_tensor_host,
            y_tensor_host,
            cache_host,
            opts,
            dY_tensor_host,
            d_hn_device,
            d_cn_device,
            h0_device,
            c0_device,
            weights_ih,
            weights_hh,
            bias_ih,
            bias_hh,
            dx_tensor_host,
            dW_ih,
            dW_hh,
            db_ih,
            db_hh,
            dh0_out,
            dc0_out,
            compute_stream,
            h2d_stream,
            d2h_stream
        );
    } catch (const std::exception &exc) {
        fprintf(stderr, "flstm_StreamingLstmBackward failed: %s\n", exc.what());
        fflush(stderr);
        std::abort();
    } catch (...) {
        fprintf(stderr, "flstm_StreamingLstmBackward failed: unknown exception\n");
        fflush(stderr);
        std::abort();
    }
}
