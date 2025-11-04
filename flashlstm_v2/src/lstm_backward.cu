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

__global__ void FuseWeightsKernel(
    const float *weight_ih,  // (4H, I) row-major
    const float *weight_hh,  // (4H, H) row-major
    float *weight_cat,       // (I+H, 4H) column-major
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
    weight_cat[row + col * rows] = value;
}

__global__ void BackwardPointwiseKernel(
    const float *dY_row,          // (B, H) row-major
    const float *dh_next_row,     // (B, H) row-major
    const float *dc_next_row,     // (B, H) row-major
    const float *gate_cache_row,  // (B, 4H) row-major (i,f,g,o)
    const float *c_t_row,         // (B, H) row-major
    const float *c_prev_row,      // (B, H) row-major
    float *dG_col_step,           // (4H, B) column-major
    float *dh_point_prev_col,     // (H, B) column-major
    float *dc_prev_row,           // (B, H) row-major
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

    const float dh_total = dY_row[idx] + dh_next_row[idx];
    const float dc_next = dc_next_row[idx];

    const float i_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 0 * hidden_size];
    const float f_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 1 * hidden_size];
    const float g_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 2 * hidden_size];
    const float o_gate = gate_cache_row[batch_idx * gate_dim + hidden_idx + 3 * hidden_size];

    const float c_t = c_t_row[idx];
    const float c_prev = c_prev_row[idx];

    const float tanh_c = tanhf(c_t);
    const float do_gate = dh_total * tanh_c;
    const float dc_total = dc_next + dh_total * o_gate * (1.0f - tanh_c * tanh_c);

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
    const float *src_col,   // (rows, cols) column-major
    float *dst_row,         // (rows, cols) row-major
    size_t rows,
    size_t cols
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
    const float *src_row,   // (rows, cols) row-major
    float *dst_row,         // (cols, rows) row-major
    size_t rows,
    size_t cols
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }
    const size_t row = idx / cols;
    const size_t col = idx % cols;
    dst_row[col * rows + row] = src_row[row * cols + col];
}

__global__ void ConvertDxToHalfKernel(
    const float *dx_col,    // (I, T*B) column-major
    __half *dx_half,        // (T, B, I) row-major
    size_t time_steps,
    size_t batch_size,
    size_t input_size
) {
    const size_t total = time_steps * batch_size * input_size;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    size_t tmp = idx;
    const size_t input_idx = tmp % input_size;
    tmp /= input_size;
    const size_t batch_idx = tmp % batch_size;
    const size_t time_idx = tmp / batch_size;

    const size_t column = time_idx * batch_size + batch_idx;
    const float value = dx_col[input_idx + column * input_size];
    dx_half[idx] = __float2half(value);
}

__global__ void FillOnesKernel(float *dst, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = 1.0f;
    }
}

} // namespace

namespace flstm {

void StreamingLstmBackward(
    size_t time_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,

    const __half *z_cache_device,
    const __half *h_cache_device,
    const __half *c_cache_device,
    const __half *gate_cache_device,

    const __half *dY_tensor_host,
    const __half *d_hn_device,
    const __half *d_cn_device,

    const float *weights_ih,
    const float *weights_hh,

    __half *dx_tensor_host,
    float *dW_ih,
    float *dW_hh,
    float *db_ih,
    float *db_hh,
    float *dh0_out,
    float *dc0_out,

    cudaStream_t stream
) {
    GPUTX_RANGE("StreamingLstmBackward");
    if (time_steps == 0 || batch_size == 0 || input_size == 0 || hidden_size == 0) {
        return;
    }
    if (z_cache_device == nullptr || h_cache_device == nullptr ||
        c_cache_device == nullptr || gate_cache_device == nullptr) {
        throw std::runtime_error("StreamingLstmBackward requires forward caches");
    }

    const size_t gate_dim = 4 * hidden_size;
    const size_t z_rows = input_size + hidden_size;
    const size_t bh_elements = batch_size * hidden_size;
    const size_t total_tb = time_steps * batch_size;
    const size_t y_elements = time_steps * bh_elements;

    DeviceBuffer<__half> dY_half_device;
    DeviceBuffer<float> dY_float;
    DeviceBuffer<float> d_hn_float;
    DeviceBuffer<float> d_cn_float;
    DeviceBuffer<float> dh_cur;
    DeviceBuffer<float> dh_tmp;
    DeviceBuffer<float> dc_cur;
    DeviceBuffer<float> dc_tmp;
    DeviceBuffer<float> z_cache_float;
    DeviceBuffer<float> c_cache_float;
    DeviceBuffer<float> gate_cache_float;
    DeviceBuffer<float> dG_cache_col;
    DeviceBuffer<float> weight_cat_col;
    DeviceBuffer<float> dWcat_col;
    DeviceBuffer<float> dWcat_row;
    DeviceBuffer<float> dX_col;
    DeviceBuffer<float> ones_vec;
    DeviceBuffer<float> db_buffer;
    DeviceBuffer<__half> dx_half_device;

    CheckCuda(cudaMalloc(&dY_half_device.ptr, y_elements * sizeof(__half)), "cudaMalloc dY_half");
    CheckCuda(cudaMemcpyAsync(
                  dY_half_device.ptr,
                  dY_tensor_host,
                  y_elements * sizeof(__half),
                  cudaMemcpyHostToDevice,
                  stream),
              "copy dY -> device");

    CheckCuda(cudaMalloc(&dY_float.ptr, y_elements * sizeof(float)), "cudaMalloc dY_float");
    const int threads = 256;
    const int y_blocks = static_cast<int>((y_elements + threads - 1) / threads);
    HalfToFloatKernel<<<y_blocks, threads, 0, stream>>>(dY_half_device.ptr, dY_float.ptr, y_elements);
    CheckCuda(cudaGetLastError(), "HalfToFloatKernel dY");

    CheckCuda(cudaMalloc(&d_hn_float.ptr, bh_elements * sizeof(float)), "cudaMalloc d_hn_float");
    CheckCuda(cudaMalloc(&d_cn_float.ptr, bh_elements * sizeof(float)), "cudaMalloc d_cn_float");
    const int bh_blocks = static_cast<int>((bh_elements + threads - 1) / threads);
    if (d_hn_device != nullptr) {
        HalfToFloatKernel<<<bh_blocks, threads, 0, stream>>>(d_hn_device, d_hn_float.ptr, bh_elements);
        CheckCuda(cudaGetLastError(), "HalfToFloatKernel d_hn");
    } else {
        CheckCuda(cudaMemsetAsync(d_hn_float.ptr, 0, bh_elements * sizeof(float), stream), "memset d_hn");
    }
    if (d_cn_device != nullptr) {
        HalfToFloatKernel<<<bh_blocks, threads, 0, stream>>>(d_cn_device, d_cn_float.ptr, bh_elements);
        CheckCuda(cudaGetLastError(), "HalfToFloatKernel d_cn");
    } else {
        CheckCuda(cudaMemsetAsync(d_cn_float.ptr, 0, bh_elements * sizeof(float), stream), "memset d_cn");
    }

    CheckCuda(cudaMalloc(&dh_cur.ptr, bh_elements * sizeof(float)), "cudaMalloc dh_cur");
    CheckCuda(cudaMalloc(&dh_tmp.ptr, bh_elements * sizeof(float)), "cudaMalloc dh_tmp");
    CheckCuda(cudaMalloc(&dc_cur.ptr, bh_elements * sizeof(float)), "cudaMalloc dc_cur");
    CheckCuda(cudaMalloc(&dc_tmp.ptr, bh_elements * sizeof(float)), "cudaMalloc dc_tmp");
    CheckCuda(cudaMemcpyAsync(dh_cur.ptr, d_hn_float.ptr, bh_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream),
              "copy d_hn");
    CheckCuda(cudaMemcpyAsync(dc_cur.ptr, d_cn_float.ptr, bh_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream),
              "copy d_cn");

    const size_t z_cache_elements = z_rows * total_tb;
    CheckCuda(cudaMalloc(&z_cache_float.ptr, z_cache_elements * sizeof(float)), "cudaMalloc z_cache_float");
    const int z_cache_blocks = static_cast<int>((z_cache_elements + threads - 1) / threads);
    HalfToFloatKernel<<<z_cache_blocks, threads, 0, stream>>>(z_cache_device, z_cache_float.ptr, z_cache_elements);
    CheckCuda(cudaGetLastError(), "HalfToFloatKernel z_cache");

    const size_t c_cache_elements = static_cast<size_t>(time_steps + 1) * bh_elements;
    CheckCuda(cudaMalloc(&c_cache_float.ptr, c_cache_elements * sizeof(float)), "cudaMalloc c_cache_float");
    const int c_cache_blocks = static_cast<int>((c_cache_elements + threads - 1) / threads);
    HalfToFloatKernel<<<c_cache_blocks, threads, 0, stream>>>(c_cache_device, c_cache_float.ptr, c_cache_elements);
    CheckCuda(cudaGetLastError(), "HalfToFloatKernel c_cache");

    const size_t gate_cache_elements = time_steps * batch_size * gate_dim;
    CheckCuda(cudaMalloc(&gate_cache_float.ptr, gate_cache_elements * sizeof(float)), "cudaMalloc gate_cache_float");
    const int gate_cache_blocks = static_cast<int>((gate_cache_elements + threads - 1) / threads);
    HalfToFloatKernel<<<gate_cache_blocks, threads, 0, stream>>>(gate_cache_device, gate_cache_float.ptr, gate_cache_elements);
    CheckCuda(cudaGetLastError(), "HalfToFloatKernel gate_cache");

    CheckCuda(cudaMalloc(&dG_cache_col.ptr, gate_dim * total_tb * sizeof(float)), "cudaMalloc dG_cache");
    CheckCuda(cudaMalloc(&weight_cat_col.ptr, z_rows * gate_dim * sizeof(float)), "cudaMalloc weight_cat_col");
    CheckCuda(cudaMalloc(&dWcat_col.ptr, z_rows * gate_dim * sizeof(float)), "cudaMalloc dWcat_col");
    CheckCuda(cudaMalloc(&dWcat_row.ptr, z_rows * gate_dim * sizeof(float)), "cudaMalloc dWcat_row");
    CheckCuda(cudaMalloc(&dX_col.ptr, input_size * total_tb * sizeof(float)), "cudaMalloc dX_col");
    CheckCuda(cudaMalloc(&ones_vec.ptr, total_tb * sizeof(float)), "cudaMalloc ones_vec");
    CheckCuda(cudaMalloc(&db_buffer.ptr, gate_dim * sizeof(float)), "cudaMalloc db_buffer");
    CheckCuda(cudaMalloc(&dx_half_device.ptr, time_steps * batch_size * input_size * sizeof(__half)),
              "cudaMalloc dx_half");

    const int ones_blocks = static_cast<int>((total_tb + threads - 1) / threads);
    FillOnesKernel<<<ones_blocks, threads, 0, stream>>>(ones_vec.ptr, total_tb);
    CheckCuda(cudaGetLastError(), "FillOnesKernel");

    const int weight_blocks = static_cast<int>(((z_rows * gate_dim) + threads - 1) / threads);
    FuseWeightsKernel<<<weight_blocks, threads, 0, stream>>>(
        weights_ih,
        weights_hh,
        weight_cat_col.ptr,
        input_size,
        hidden_size
    );
    CheckCuda(cudaGetLastError(), "FuseWeightsKernel");

    CublasHandle cublas;
    CheckCublas(cublasCreate(&cublas.handle), "cublasCreate");
    CheckCublas(cublasSetStream(cublas.handle, stream), "cublasSetStream");
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;

    const int point_blocks = static_cast<int>((bh_elements + threads - 1) / threads);

    for (int t = static_cast<int>(time_steps) - 1; t >= 0; --t) {
        const size_t column_offset = static_cast<size_t>(t) * batch_size;
        float *dG_step = dG_cache_col.ptr + column_offset * gate_dim;
        float *dh_out = dh_tmp.ptr;
        float *dc_out = dc_tmp.ptr;

        const float *dY_t = dY_float.ptr + static_cast<size_t>(t) * bh_elements;
        const float *gate_step = gate_cache_float.ptr + static_cast<size_t>(t) * batch_size * gate_dim;
        const float *c_t = c_cache_float.ptr + static_cast<size_t>(t + 1) * bh_elements;
        const float *c_prev = c_cache_float.ptr + static_cast<size_t>(t) * bh_elements;

        BackwardPointwiseKernel<<<point_blocks, threads, 0, stream>>>(
            dY_t,
            dh_cur.ptr,
            dc_cur.ptr,
            gate_step,
            c_t,
            c_prev,
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

    // Parameter gradients
    CheckCublas(cublasSgemm(
                    cublas.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    static_cast<int>(z_rows),
                    static_cast<int>(gate_dim),
                    static_cast<int>(total_tb),
                    &alpha,
                    z_cache_float.ptr,
                    static_cast<int>(z_rows),
                    dG_cache_col.ptr,
                    static_cast<int>(gate_dim),
                    &beta_zero,
                    dWcat_col.ptr,
                    static_cast<int>(z_rows)),
                "cublasSgemm dWcat");

    const size_t dWcat_elems = z_rows * gate_dim;
    const int dWcat_blocks = static_cast<int>((dWcat_elems + threads - 1) / threads);
    ColumnToRowKernel<<<dWcat_blocks, threads, 0, stream>>>(
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
    const int dW_ih_blocks = static_cast<int>((dW_ih_elems + threads - 1) / threads);
    const int dW_hh_blocks = static_cast<int>((dW_hh_elems + threads - 1) / threads);
    TransposeKernel<<<dW_ih_blocks, threads, 0, stream>>>(
        dW_ih_src,
        dW_ih,
        input_size,
        gate_dim
    );
    CheckCuda(cudaGetLastError(), "TransposeKernel dW_ih");
    TransposeKernel<<<dW_hh_blocks, threads, 0, stream>>>(
        dW_hh_src,
        dW_hh,
        hidden_size,
        gate_dim
    );
    CheckCuda(cudaGetLastError(), "TransposeKernel dW_hh");

    // Bias gradients
    CheckCublas(cublasSgemv(
                    cublas.handle,
                    CUBLAS_OP_N,
                    static_cast<int>(gate_dim),
                    static_cast<int>(total_tb),
                    &alpha,
                    dG_cache_col.ptr,
                    static_cast<int>(gate_dim),
                    ones_vec.ptr,
                    1,
                    &beta_zero,
                    db_buffer.ptr,
                    1),
                "cublasSgemv db");
    CheckCuda(cudaMemcpyAsync(
                  db_ih,
                  db_buffer.ptr,
                  gate_dim * sizeof(float),
                  cudaMemcpyDeviceToDevice,
                  stream),
              "copy db -> db_ih");
    CheckCuda(cudaMemcpyAsync(
                  db_hh,
                  db_buffer.ptr,
                  gate_dim * sizeof(float),
                  cudaMemcpyDeviceToDevice,
                  stream),
              "copy db -> db_hh");

    // dx for entire sequence
    CheckCublas(cublasSgemm(
                    cublas.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    static_cast<int>(input_size),
                    static_cast<int>(total_tb),
                    static_cast<int>(gate_dim),
                    &alpha,
                    weight_cat_col.ptr,
                    static_cast<int>(z_rows),
                    dG_cache_col.ptr,
                    static_cast<int>(gate_dim),
                    &beta_zero,
                    dX_col.ptr,
                    static_cast<int>(input_size)),
                "cublasSgemm dX");

    const size_t dx_elements = time_steps * batch_size * input_size;
    const int dx_blocks = static_cast<int>((dx_elements + threads - 1) / threads);
    ConvertDxToHalfKernel<<<dx_blocks, threads, 0, stream>>>(
        dX_col.ptr,
        dx_half_device.ptr,
        time_steps,
        batch_size,
        input_size
    );
    CheckCuda(cudaGetLastError(), "ConvertDxToHalfKernel");
    CheckCuda(cudaMemcpyAsync(
                  dx_tensor_host,
                  dx_half_device.ptr,
                  dx_elements * sizeof(__half),
                  cudaMemcpyDeviceToHost,
                  stream),
              "copy dx -> host");

    CheckCuda(cudaMemcpyAsync(
                  dh0_out,
                  dh_cur.ptr,
                  bh_elements * sizeof(float),
                  cudaMemcpyDeviceToDevice,
                  stream),
              "copy dh0");
    CheckCuda(cudaMemcpyAsync(
                  dc0_out,
                  dc_cur.ptr,
                  bh_elements * sizeof(float),
                  cudaMemcpyDeviceToDevice,
                  stream),
              "copy dc0");

    CheckCuda(cudaStreamSynchronize(stream), "final backward sync");
}

} // namespace flstm

extern "C" void flstm_StreamingLstmBackward(
    size_t time_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,

    const __half *z_cache_device,
    const __half *h_cache_device,
    const __half *c_cache_device,
    const __half *gate_cache_device,

    const __half *dY_tensor_host,
    const __half *d_hn_device,
    const __half *d_cn_device,

    const float *weights_ih,
    const float *weights_hh,

    __half *dx_tensor_host,
    float *dW_ih,
    float *dW_hh,
    float *db_ih,
    float *db_hh,
    float *dh0_out,
    float *dc0_out,

    cudaStream_t compute_stream
) {
    try {
        flstm::StreamingLstmBackward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            z_cache_device,
            h_cache_device,
            c_cache_device,
            gate_cache_device,
            dY_tensor_host,
            d_hn_device,
            d_cn_device,
            weights_ih,
            weights_hh,
            dx_tensor_host,
            dW_ih,
            dW_hh,
            db_ih,
            db_hh,
            dh0_out,
            dc0_out,
            compute_stream
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
