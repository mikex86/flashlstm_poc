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

    void CheckCuda(const cudaError_t err, const char *what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
        }
    }

    void CheckCublas(const cublasStatus_t status, const char *what) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(what) + ": " + CublasStatusString(status));
        }
    }

    /// Helper for 1D grid launches; mirrors the arithmetic sprinkled throughout the kernel site code.
    int BlocksFor(const size_t count, const int threads = 256) {
        return static_cast<int>((count + threads - 1) / threads);
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
        float *weight_cat, // (I+H, 4H) column-major
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
        weight_cat[row + col * rows] = value;
    }

    __global__ void BackwardPointwiseKernel(
        const float *dY_row, // (B, H) row-major
        const float *dh_next_row, // (B, H) row-major
        const float *dc_next_row, // (B, H) row-major
        const float *gate_cache_row, // (B, 4H) row-major (i,f,g,o)
        const float *y_row, // (B, H) row-major
        float *dG_col_step, // (4H, B) column-major
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

        const float dh_total = dY_row[idx] + dh_next_row[idx];
        const float dc_next = dc_next_row[idx];

        const float i_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 0 * hidden_size];
        const float f_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 1 * hidden_size];
        const float g_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 2 * hidden_size];
        const float o_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 3 * hidden_size];

        const float y_t = y_row[idx];
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

        dh_point_prev_col[hidden_idx + batch_idx * hidden_size] = 0.0f; // initialise accumulator
        dc_prev_row[idx] = dc_prev;
    }

    __global__ void ColumnToRowKernel(
        const float *src_col, // (rows, cols) column-major
        float *dst_row, // (rows, cols) row-major
        const size_t rows,
        const size_t cols
    ) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= rows * cols) {
            return;
        }
        const size_t col = idx / rows;
        const size_t row = idx % rows;
        dst_row[row * cols + col] = src_col[row + col * rows];
    }

    __global__ void TransposeKernel(
        const float *src_row, // (rows, cols) row-major
        float *dst_row, // (cols, rows) row-major
        const size_t rows,
        const size_t cols
    ) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= rows * cols) {
            return;
        }
        const size_t row = idx / cols;
        const size_t col = idx % cols;
        dst_row[col * rows + row] = src_row[row * cols + col];
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
        float *z_chunk_col, // (I+H, chunk_tb) column-major
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
        z_chunk_col[input_idx + column * z_rows] = __half2float(x_src[idx]);
    }

    __global__ void FillHiddenPartForZChunkKernel(
        const float *y_chunk, // (chunk_steps, B, H) row-major
        const float *first_prev, // (B, H) row-major
        const size_t chunk_steps,
        const size_t batch_size,
        const size_t hidden_size,
        const size_t input_size,
        float *z_chunk_col // (I+H, chunk_tb) column-major
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

        float h_prev;
        if (step_idx == 0) {
            h_prev = first_prev[batch_idx * hidden_size + hidden_idx];
        } else {
            const float *y_prev = y_chunk + ((step_idx - 1) * batch_size + batch_idx) * hidden_size;
            h_prev = y_prev[hidden_idx];
        }

        z_chunk_col[input_size + hidden_idx + column * z_rows] = h_prev;
    }

    __global__ void FillOnesKernel(float *dst, const size_t count) {
        if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count) {
            dst[idx] = 1.0f;
        }
    }

    // Holds references shared between copy slots so double-buffered H2D transfers stay readable.
    struct ChunkCopyParams {
        size_t time_steps;
        size_t chunk_capacity;
        size_t batch_size;
        size_t gate_dim;
        size_t input_size;
        size_t bh_elements;
        const __half *gate_cache_host;
        const __half *x_tensor_host;
        const __half *y_tensor_host;
        const __half *dY_tensor_host;
        DeviceBuffer<__half> *gate_chunk_half;
        DeviceBuffer<__half> *x_chunk_half;
        DeviceBuffer<__half> *y_chunk_half;
        DeviceBuffer<__half> *dY_chunk_half;
        DeviceBuffer<__half> *y_prev_half;
        CudaEvent *h2d_ready;
        cudaStream_t h2d_stream;
        bool *compute_done_valid;
        CudaEvent *compute_done;
        size_t *chunk_ids;
    };

    // Schedules host-to-device copies for one chunk slot, waiting on outstanding compute if needed.
    size_t IssueChunkCopy(const ssize_t chunk_id, const int slot, const ChunkCopyParams &params) {
        if (chunk_id < 0) {
            params.chunk_ids[slot] = static_cast<size_t>(-1);
            return 0;
        }

        const size_t chunk_start = static_cast<size_t>(chunk_id) * params.chunk_capacity;
        if (chunk_start >= params.time_steps) {
            params.chunk_ids[slot] = static_cast<size_t>(-1);
            return 0;
        }

        if (params.compute_done_valid[slot]) {
            CheckCuda(cudaStreamWaitEvent(params.h2d_stream, params.compute_done[slot].evt, 0),
                      "wait compute_done before reuse");
            params.compute_done_valid[slot] = false;
        }

        const size_t steps = std::min(params.chunk_capacity, params.time_steps - chunk_start);
        const size_t gate_elems = steps * params.batch_size * params.gate_dim;
        if (gate_elems > 0 && params.gate_chunk_half[slot].ptr != nullptr) {
            CheckCuda(cudaMemcpyAsync(
                          params.gate_chunk_half[slot].ptr,
                          params.gate_cache_host + chunk_start * params.batch_size * params.gate_dim,
                          gate_elems * sizeof(__half),
                          cudaMemcpyHostToDevice,
                          params.h2d_stream),
                      "copy gate chunk");
        }

        const size_t x_elems = steps * params.batch_size * params.input_size;
        if (x_elems > 0 && params.x_chunk_half[slot].ptr != nullptr) {
            CheckCuda(cudaMemcpyAsync(
                          params.x_chunk_half[slot].ptr,
                          params.x_tensor_host + chunk_start * params.batch_size * params.input_size,
                          x_elems * sizeof(__half),
                          cudaMemcpyHostToDevice,
                          params.h2d_stream),
                      "copy x chunk");
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

        if (chunk_start > 0) {
            CheckCuda(cudaMemcpyAsync(
                          params.y_prev_half[slot].ptr,
                          params.y_tensor_host + (chunk_start - 1) * params.bh_elements,
                          params.bh_elements * sizeof(__half),
                          cudaMemcpyHostToDevice,
                          params.h2d_stream),
                      "copy y_prev");
        }

        CheckCuda(cudaEventRecord(params.h2d_ready[slot].evt, params.h2d_stream), "record h2d_ready");
        params.chunk_ids[slot] = static_cast<size_t>(chunk_id);
        return steps;
    }
} // namespace

namespace flstm {
    void StreamingLstmBackward(
        size_t time_steps,
        size_t batch_size,
        size_t input_size,
        size_t hidden_size,

        const __half *x_tensor_host,
        const __half *y_tensor_host,
        const __half *gate_cache_host,

        const __half *dY_tensor_host,
        const __half *d_hn_device,
        const __half *d_cn_device,
        const __half *h0_device,
        const __half *c0_device,

        const float *weights_ih,
        const float *weights_hh,

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
        if (time_steps == 0 || batch_size == 0 || input_size == 0 || hidden_size == 0) {
            return;
        }
        if (x_tensor_host == nullptr || y_tensor_host == nullptr || gate_cache_host == nullptr ||
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
        DeviceBuffer<float> h0_float;
        DeviceBuffer<float> weight_cat_col;
        DeviceBuffer<float> dWcat_col;
        DeviceBuffer<float> dWcat_row;
        DeviceBuffer<float> db_buffer;
        DeviceBuffer<float> ones_vec;

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

        AllocateDeviceBuffer(h0_float, bh_elements, "cudaMalloc h0_float");
        HalfToFloatKernel<<<bh_blocks, threads, 0, compute_stream>>>(h0_device, h0_float.ptr, bh_elements);
        CheckCuda(cudaGetLastError(), "HalfToFloatKernel h0");

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

        AllocateDeviceBuffer(dWcat_col, weight_elems, "cudaMalloc dWcat_col");
        ZeroDeviceMemory(dWcat_col.ptr, weight_elems, compute_stream, "memset dWcat_col");
        AllocateDeviceBuffer(dWcat_row, weight_elems, "cudaMalloc dWcat_row");

        AllocateDeviceBuffer(db_buffer, gate_dim, "cudaMalloc db_buffer");
        ZeroDeviceMemory(db_buffer.ptr, gate_dim, compute_stream, "memset db buffer");

        constexpr size_t chunk_capacity = kChunkSteps;
        const size_t chunk_gate_capacity = chunk_capacity * batch_size * gate_dim;
        const size_t chunk_input_capacity = chunk_capacity * batch_size * input_size;
        const size_t chunk_hidden_capacity = chunk_capacity * batch_size * hidden_size;
        const size_t chunk_tb_capacity = chunk_capacity * batch_size;

        AllocateDeviceBuffer(ones_vec, chunk_tb_capacity, "cudaMalloc ones_vec");
        const int ones_blocks = BlocksFor(chunk_tb_capacity, threads);
        FillOnesKernel<<<ones_blocks, threads, 0, compute_stream>>>(ones_vec.ptr, chunk_tb_capacity);
        CheckCuda(cudaGetLastError(), "FillOnesKernel");

        DeviceBuffer<__half> gate_chunk_half[2];
        DeviceBuffer<__half> x_chunk_half[2];
        DeviceBuffer<__half> y_chunk_half[2];
        DeviceBuffer<__half> dY_chunk_half[2];
        DeviceBuffer<float> gate_chunk_float[2];
        DeviceBuffer<float> dY_chunk_float[2];
        DeviceBuffer<float> y_chunk_float[2];
        DeviceBuffer<float> z_chunk_col[2];
        DeviceBuffer<float> dG_chunk_col[2];
        DeviceBuffer<float> dX_chunk_col[2];
        DeviceBuffer<__half> dx_chunk_half[2];
        DeviceBuffer<__half> y_prev_half[2];
        DeviceBuffer<float> h_prev_boundary_float[2];

        const size_t z_chunk_elements = z_rows * chunk_tb_capacity;
        const size_t dX_chunk_elements = input_size * chunk_tb_capacity;
        const size_t dG_chunk_elements = gate_dim * chunk_tb_capacity;

        AllocateDeviceBufferArray(gate_chunk_half, chunk_gate_capacity, "cudaMalloc gate_chunk_half");
        AllocateDeviceBufferArray(gate_chunk_float, chunk_gate_capacity, "cudaMalloc gate_chunk_float");
        AllocateDeviceBufferArray(x_chunk_half, chunk_input_capacity, "cudaMalloc x_chunk_half");
        AllocateDeviceBufferArray(z_chunk_col, z_chunk_elements, "cudaMalloc z_chunk_col");
        AllocateDeviceBufferArray(dX_chunk_col, dX_chunk_elements, "cudaMalloc dX_chunk_col");
        AllocateDeviceBufferArray(dx_chunk_half, chunk_input_capacity, "cudaMalloc dx_chunk_half");
        AllocateDeviceBufferArray(y_chunk_half, chunk_hidden_capacity, "cudaMalloc y_chunk_half");
        AllocateDeviceBufferArray(dY_chunk_half, chunk_hidden_capacity, "cudaMalloc dY_chunk_half");
        AllocateDeviceBufferArray(dY_chunk_float, chunk_hidden_capacity, "cudaMalloc dY_chunk_float");
        AllocateDeviceBufferArray(y_chunk_float, chunk_hidden_capacity, "cudaMalloc y_chunk_float");
        AllocateDeviceBufferArray(dG_chunk_col, dG_chunk_elements, "cudaMalloc dG_chunk_col");
        AllocateDeviceBufferArray(y_prev_half, bh_elements, "cudaMalloc y_prev_half");
        AllocateDeviceBufferArray(h_prev_boundary_float, bh_elements, "cudaMalloc h_prev_boundary_float");

        CublasHandle cublas;
        CheckCublas(cublasCreate(&cublas.handle), "cublasCreate");
        CheckCublas(cublasSetStream(cublas.handle, compute_stream), "cublasSetStream");
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

        // Shared metadata for the double-buffered copy helper.
        ChunkCopyParams chunk_params{
            .time_steps = time_steps,
            .chunk_capacity = chunk_capacity,
            .batch_size = batch_size,
            .gate_dim = gate_dim,
            .input_size = input_size,
            .bh_elements = bh_elements,
            .gate_cache_host = gate_cache_host,
            .x_tensor_host = x_tensor_host,
            .y_tensor_host = y_tensor_host,
            .dY_tensor_host = dY_tensor_host,
            .gate_chunk_half = gate_chunk_half,
            .x_chunk_half = x_chunk_half,
            .y_chunk_half = y_chunk_half,
            .dY_chunk_half = dY_chunk_half,
            .y_prev_half = y_prev_half,
            .h2d_ready = h2d_ready,
            .h2d_stream = h2d_stream,
            .compute_done_valid = compute_done_valid,
            .compute_done = compute_done,
            .chunk_ids = chunk_ids
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

            const size_t gate_elems = steps_in_chunk * batch_size * gate_dim;
            if (gate_elems > 0 && gate_chunk_half[slot].ptr != nullptr) {
                const int gate_blocks = BlocksFor(gate_elems, threads);
                HalfToFloatKernel<<<gate_blocks, threads, 0, compute_stream>>>(
                    gate_chunk_half[slot].ptr,
                    gate_chunk_float[slot].ptr,
                    gate_elems
                );
                CheckCuda(cudaGetLastError(), "HalfToFloatKernel gate chunk");
            }

            const size_t chunk_hidden_elems = steps_in_chunk * bh_elements;
            if (chunk_hidden_elems > 0) {
                if (dY_chunk_half[slot].ptr != nullptr && dY_chunk_float[slot].ptr != nullptr) {
                    const int dY_blocks = BlocksFor(chunk_hidden_elems, threads);
                    HalfToFloatKernel<<<dY_blocks, threads, 0, compute_stream>>>(
                        dY_chunk_half[slot].ptr,
                        dY_chunk_float[slot].ptr,
                        chunk_hidden_elems
                    );
                    CheckCuda(cudaGetLastError(), "HalfToFloatKernel dY chunk");
                }
                if (y_chunk_half[slot].ptr != nullptr && y_chunk_float[slot].ptr != nullptr) {
                    const int y_blocks = BlocksFor(chunk_hidden_elems, threads);
                    HalfToFloatKernel<<<y_blocks, threads, 0, compute_stream>>>(
                        y_chunk_half[slot].ptr,
                        y_chunk_float[slot].ptr,
                        chunk_hidden_elems
                    );
                    CheckCuda(cudaGetLastError(), "HalfToFloatKernel y chunk");
                }
            }

            const size_t chunk_input_elems = steps_in_chunk * batch_size * input_size;
            if (chunk_input_elems > 0 && x_chunk_half[slot].ptr != nullptr) {
                const int convert_blocks = BlocksFor(chunk_input_elems, threads);
                ConvertInputToZChunkKernel<<<convert_blocks, threads, 0, compute_stream>>>(
                    x_chunk_half[slot].ptr,
                    z_chunk_col[slot].ptr,
                    steps_in_chunk,
                    batch_size,
                    input_size,
                    hidden_size
                );
                CheckCuda(cudaGetLastError(), "ConvertInputToZChunkKernel");
            }

            const float *first_prev_ptr = nullptr;
            if (chunk_start_step == 0) {
                first_prev_ptr = h0_float.ptr;
            } else {
                const int prev_blocks = BlocksFor(bh_elements, threads);
                HalfToFloatKernel<<<prev_blocks, threads, 0, compute_stream>>>(
                    y_prev_half[slot].ptr,
                    h_prev_boundary_float[slot].ptr,
                    bh_elements
                );
                CheckCuda(cudaGetLastError(), "HalfToFloatKernel y_prev");
                first_prev_ptr = h_prev_boundary_float[slot].ptr;
            }

            if (chunk_hidden_elems > 0) {
                const int hidden_blocks = BlocksFor(chunk_hidden_elems, threads);
                FillHiddenPartForZChunkKernel<<<hidden_blocks, threads, 0, compute_stream>>>(
                    y_chunk_float[slot].ptr,
                    first_prev_ptr,
                    steps_in_chunk,
                    batch_size,
                    hidden_size,
                    input_size,
                    z_chunk_col[slot].ptr
                );
                CheckCuda(cudaGetLastError(), "FillHiddenPartForZChunkKernel");
            }

            for (int step = static_cast<int>(steps_in_chunk) - 1; step >= 0; --step) {
                const size_t local_offset = static_cast<size_t>(step) * batch_size;
                float *dG_step = dG_chunk_col[slot].ptr + local_offset * gate_dim;
                float *dh_out = dh_tmp.ptr;
                float *dc_out = dc_tmp.ptr;

                const float *dY_t = dY_chunk_float[slot].ptr + static_cast<size_t>(step) * bh_elements;
                const float *gate_step = gate_chunk_float[slot].ptr + static_cast<size_t>(step) * batch_size * gate_dim;
                const float *y_step = y_chunk_float[slot].ptr + static_cast<size_t>(step) * bh_elements;

                BackwardPointwiseKernel<<<point_blocks, threads, 0, compute_stream>>>(
                    dY_t,
                    dh_cur.ptr,
                    dc_cur.ptr,
                    gate_step,
                    y_step,
                    dG_step,
                    dh_out,
                    dc_out,
                    batch_size,
                    hidden_size
                );
                CheckCuda(cudaGetLastError(), "BackwardPointwiseKernel");

                const float *W_hh_block = weight_cat_col.ptr + input_size;
                CheckCublas(cublasSgemm(
                                cublas.handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                static_cast<int>(hidden_size),
                                static_cast<int>(batch_size),
                                static_cast<int>(gate_dim),
                                &alpha,
                                W_hh_block,
                                static_cast<int>(z_rows),
                                dG_step,
                                static_cast<int>(gate_dim),
                                &beta_one,
                                dh_out,
                                static_cast<int>(hidden_size)),
                            "cublasSgemm dh_prev");

                std::swap(dh_cur.ptr, dh_tmp.ptr);
                std::swap(dc_cur.ptr, dc_tmp.ptr);
            }

            bool issued_dx_copy = false;
            size_t dx_chunk_elems = 0;
            const int zgemm_m = static_cast<int>(z_rows);
            const int zgemm_n = static_cast<int>(gate_dim);
            const int zgemm_k = static_cast<int>(chunk_tb);
            if (zgemm_k > 0) {
                CheckCublas(cublasSgemm(
                                cublas.handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                zgemm_m,
                                zgemm_n,
                                zgemm_k,
                                &alpha,
                                z_chunk_col[slot].ptr,
                                static_cast<int>(z_rows),
                                dG_chunk_col[slot].ptr,
                                static_cast<int>(gate_dim),
                                &beta_one,
                                dWcat_col.ptr,
                                static_cast<int>(z_rows)),
                            "cublasSgemm dWcat chunk");

                CheckCublas(cublasSgemv(
                                cublas.handle,
                                CUBLAS_OP_N,
                                static_cast<int>(gate_dim),
                                zgemm_k,
                                &alpha,
                                dG_chunk_col[slot].ptr,
                                static_cast<int>(gate_dim),
                                ones_vec.ptr,
                                1,
                                &beta_one,
                                db_buffer.ptr,
                                1),
                            "cublasSgemv db chunk");

                CheckCublas(cublasSgemm(
                                cublas.handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                static_cast<int>(input_size),
                                zgemm_k,
                                static_cast<int>(gate_dim),
                                &alpha,
                                weight_cat_col.ptr,
                                static_cast<int>(z_rows),
                                dG_chunk_col[slot].ptr,
                                static_cast<int>(gate_dim),
                                &beta_zero,
                                dX_chunk_col[slot].ptr,
                                static_cast<int>(input_size)),
                            "cublasSgemm dX chunk");

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

        if (weight_elems > 0) {
            const int col_to_row_blocks = BlocksFor(weight_elems, threads);
            ColumnToRowKernel<<<col_to_row_blocks, threads, 0, compute_stream>>>(
                dWcat_col.ptr,
                dWcat_row.ptr,
                z_rows,
                gate_dim
            );
            CheckCuda(cudaGetLastError(), "ColumnToRowKernel dWcat");

            const float *dW_ih_src = dWcat_row.ptr;
            const float *dW_hh_src = dWcat_row.ptr + input_size * gate_dim;
            const size_t dW_ih_elems = static_cast<size_t>(input_size) * gate_dim;
            const size_t dW_hh_elems = static_cast<size_t>(hidden_size) * gate_dim;
            const int dW_ih_blocks = BlocksFor(dW_ih_elems, threads);
            const int dW_hh_blocks = BlocksFor(dW_hh_elems, threads);

            TransposeKernel<<<dW_ih_blocks, threads, 0, compute_stream>>>(
                dW_ih_src,
                dW_ih,
                input_size,
                gate_dim
            );
            CheckCuda(cudaGetLastError(), "TransposeKernel dW_ih");

            TransposeKernel<<<dW_hh_blocks, threads, 0, compute_stream>>>(
                dW_hh_src,
                dW_hh,
                hidden_size,
                gate_dim
            );
            CheckCuda(cudaGetLastError(), "TransposeKernel dW_hh");
        }

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
    }
} // namespace flstm

extern "C" void flstm_StreamingLstmBackward(
    const size_t time_steps,
    const size_t batch_size,
    const size_t input_size,
    const size_t hidden_size,

    const __half *x_tensor_host,
    const __half *y_tensor_host,
    const __half *gate_cache_host,

    const __half *dY_tensor_host,
    const __half *d_hn_device,
    const __half *d_cn_device,
    const __half *h0_device,
    const __half *c0_device,

    const float *weights_ih,
    const float *weights_hh,

    __half *dx_tensor_host,
    float *dW_ih,
    float *dW_hh,
    float *db_ih,
    float *db_hh,
    float *dh0_out,
    float *dc0_out,

    const cudaStream_t compute_stream,
    const cudaStream_t h2d_stream,
    const cudaStream_t d2h_stream
) {
    try {
        flstm::StreamingLstmBackward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            x_tensor_host,
            y_tensor_host,
            gate_cache_host,
            dY_tensor_host,
            d_hn_device,
            d_cn_device,
            h0_device,
            c0_device,
            weights_ih,
            weights_hh,
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
