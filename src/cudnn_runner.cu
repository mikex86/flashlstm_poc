#include "cudnn_runner.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_ops.h>
#include <cudnn_adv.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "nvtx_profinst.h"

namespace {

using CudnnRnnForwardFn = cudnnStatus_t (*)(cudnnHandle_t,
                                            cudnnRNNDescriptor_t,
                                            cudnnForwardMode_t,
                                            const int32_t*,
                                            cudnnRNNDataDescriptor_t,
                                            const void*,
                                            cudnnRNNDataDescriptor_t,
                                            void*,
                                            cudnnTensorDescriptor_t,
                                            const void*,
                                            void*,
                                            cudnnTensorDescriptor_t,
                                            const void*,
                                            void*,
                                            size_t,
                                            const void*,
                                            size_t,
                                            void*,
                                            size_t,
                                            void*);

constexpr CudnnRnnForwardFn kCudnnRnnForward = &cudnnRNNForward;

}  // namespace

int initialize_cudnn() {
    NVTX_SCOPED_RANGE("cuDNN::initialize");
    cudnnHandle_t handle = nullptr;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "cuDNN error %s (%d) in cudnnCreate during initialize_cudnn\n",
                     cudnnGetErrorString(status),
                     static_cast<int>(status));
        return static_cast<int>(status);
    }
    status = cudnnDestroy(handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "cuDNN error %s (%d) in cudnnDestroy during initialize_cudnn\n",
                     cudnnGetErrorString(status),
                     static_cast<int>(status));
        return static_cast<int>(status);
    }
    return 0;
}

int run_cudnn_lstm(const float* x_host,
                   const float* h0_host,
                   const float* c0_host,
                   float* y_host,
                   float* hy_host,
                   float* cy_host,
                   const std::size_t seq_len,
                   const std::size_t batch,
                   const std::size_t input_size,
                   const std::size_t hidden_size) {
    int status = 0;

    NVTX_SCOPED_RANGE("run_cudnn_lstm");

    const std::size_t x_bytes = seq_len * batch * input_size * sizeof(float);
    const std::size_t y_bytes = seq_len * batch * hidden_size * sizeof(float);
    const std::size_t state_bytes = batch * hidden_size * sizeof(float);
    const std::uint64_t seed = 1234ULL;

    const int tensor_dims = 3;
    int state_dims[tensor_dims] = {1,
                                   static_cast<int>(batch),
                                   static_cast<int>(hidden_size)};
    int state_strides[tensor_dims] = {static_cast<int>(batch * hidden_size),
                                      static_cast<int>(hidden_size),
                                      1};

    std::vector<int> seq_lengths_host(batch, static_cast<int>(seq_len));
    std::vector<int32_t> seq_lengths_device_host(batch, static_cast<int32_t>(seq_len));

    size_t dropout_states_bytes = 0;
    size_t workspace_bytes = 0;
    size_t reserve_space_bytes = 0;
    size_t weight_space_bytes = 0;

    cudnnHandle_t handle = nullptr;
    cudnnDropoutDescriptor_t dropout_desc = nullptr;
    cudnnRNNDescriptor_t rnn_desc = nullptr;
    cudnnRNNDataDescriptor_t x_data_desc = nullptr;
    cudnnRNNDataDescriptor_t y_data_desc = nullptr;
    cudnnTensorDescriptor_t h_desc = nullptr;
    cudnnTensorDescriptor_t c_desc = nullptr;

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_hx = nullptr;
    float* d_cx = nullptr;
    float* d_hy = nullptr;
    float* d_cy = nullptr;
    void* d_weight_space = nullptr;
    void* d_workspace = nullptr;
    void* d_reserve_space = nullptr;
    void* d_dropout_states = nullptr;
    int32_t* d_seq_lengths = nullptr;

    auto check_cuda = [&](cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "CUDA error %s (%d) in %s\n",
                         cudaGetErrorString(err),
                         static_cast<int>(err),
                         what);
            status = static_cast<int>(err);
            return false;
        }
        return true;
    };

    auto check_cudnn = [&](cudnnStatus_t err, const char* what) {
        if (err != CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr,
                         "cuDNN error %s (%d) in %s\n",
                         cudnnGetErrorString(err),
                         static_cast<int>(err),
                         what);
            status = static_cast<int>(err);
            return false;
        }
        return true;
    };

    auto check_cuda_wrap = [&](cudaError_t err, const char* what) {
        return check_cuda(err, what);
    };
    auto check_cudnn_wrap = [&](cudnnStatus_t err, const char* what) {
        return check_cudnn(err, what);
    };

#define CUDA_CALL(expr)                      \
    do {                                     \
        if (!check_cuda_wrap((expr), #expr)) \
            goto cleanup;                    \
    } while (0)

#define CUDNN_CALL(expr)                     \
    do {                                     \
        if (!check_cudnn_wrap((expr), #expr))\
            goto cleanup;                    \
    } while (0)

    CUDNN_CALL(cudnnCreate(&handle));
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
    CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&x_data_desc));
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&y_data_desc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&h_desc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&c_desc));

    CUDA_CALL(cudaMalloc(&d_x, x_bytes));
    CUDA_CALL(cudaMalloc(&d_y, y_bytes));
    CUDA_CALL(cudaMalloc(&d_hx, state_bytes));
    CUDA_CALL(cudaMalloc(&d_cx, state_bytes));
    CUDA_CALL(cudaMalloc(&d_hy, state_bytes));
    CUDA_CALL(cudaMalloc(&d_cy, state_bytes));
    CUDA_CALL(cudaMalloc(&d_seq_lengths, seq_lengths_device_host.size() * sizeof(int32_t)));

    CUDA_CALL(cudaMemcpy(d_x, x_host, x_bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_hx, h0_host, state_bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_cx, c0_host, state_bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_seq_lengths,
                         seq_lengths_device_host.data(),
                         seq_lengths_device_host.size() * sizeof(int32_t),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(d_y, 0, y_bytes));
    CUDA_CALL(cudaMemset(d_hy, 0, state_bytes));
    CUDA_CALL(cudaMemset(d_cy, 0, state_bytes));

    CUDNN_CALL(cudnnDropoutGetStatesSize(handle, &dropout_states_bytes));
    if (dropout_states_bytes > 0) {
        CUDA_CALL(cudaMalloc(&d_dropout_states, dropout_states_bytes));
        CUDA_CALL(cudaMemset(d_dropout_states, 0, dropout_states_bytes));
    }

    CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc,
                                         handle,
                                         0.0f,
                                         d_dropout_states,
                                         dropout_states_bytes,
                                         seed));

    CUDNN_CALL(cudnnSetRNNDescriptor_v8(rnn_desc,
                                        CUDNN_RNN_ALGO_STANDARD,
                                        CUDNN_LSTM,
                                        CUDNN_RNN_DOUBLE_BIAS,
                                        CUDNN_UNIDIRECTIONAL,
                                        CUDNN_LINEAR_INPUT,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
                                        static_cast<int32_t>(input_size),
                                        static_cast<int32_t>(hidden_size),
                                        static_cast<int32_t>(hidden_size),
                                        1,
                                        dropout_desc,
                                        0));

    CUDNN_CALL(cudnnBuildRNNDynamic(handle,
                                    rnn_desc,
                                    static_cast<int>(batch)));

    CUDNN_CALL(cudnnSetRNNDataDescriptor(x_data_desc,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                                         static_cast<int>(seq_len),
                                         static_cast<int>(batch),
                                         static_cast<int>(input_size),
                                         seq_lengths_host.data(),
                                         nullptr));

    CUDNN_CALL(cudnnSetRNNDataDescriptor(y_data_desc,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                                         static_cast<int>(seq_len),
                                         static_cast<int>(batch),
                                         static_cast<int>(hidden_size),
                                         seq_lengths_host.data(),
                                         nullptr));

    CUDNN_CALL(cudnnSetTensorNdDescriptor(h_desc,
                                          CUDNN_DATA_FLOAT,
                                          tensor_dims,
                                          state_dims,
                                          state_strides));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(c_desc,
                                          CUDNN_DATA_FLOAT,
                                          tensor_dims,
                                          state_dims,
                                          state_strides));

    CUDNN_CALL(cudnnGetRNNTempSpaceSizes(handle,
                                         rnn_desc,
                                         CUDNN_FWD_MODE_INFERENCE,
                                         x_data_desc,
                                         &workspace_bytes,
                                         &reserve_space_bytes));
    CUDNN_CALL(cudnnGetRNNWeightSpaceSize(handle,
                                          rnn_desc,
                                          &weight_space_bytes));

    if (workspace_bytes > 0) {
        CUDA_CALL(cudaMalloc(&d_workspace, workspace_bytes));
        CUDA_CALL(cudaMemset(d_workspace, 0, workspace_bytes));
    }
    if (reserve_space_bytes > 0) {
        CUDA_CALL(cudaMalloc(&d_reserve_space, reserve_space_bytes));
        CUDA_CALL(cudaMemset(d_reserve_space, 0, reserve_space_bytes));
    }
    if (weight_space_bytes > 0) {
        CUDA_CALL(cudaMalloc(&d_weight_space, weight_space_bytes));
        CUDA_CALL(cudaMemset(d_weight_space, 0, weight_space_bytes));
    }

    {
        NVTX_SCOPED_RANGE("cuDNN::cudnnRNNForward");
        CUDNN_CALL(kCudnnRnnForward(handle,
                                    rnn_desc,
                                    CUDNN_FWD_MODE_INFERENCE,
                                    d_seq_lengths,
                                    x_data_desc,
                                    d_x,
                                    y_data_desc,
                                    d_y,
                                    h_desc,
                                    d_hx,
                                    d_hy,
                                    c_desc,
                                    d_cx,
                                    d_cy,
                                    weight_space_bytes,
                                    d_weight_space,
                                    workspace_bytes,
                                    d_workspace,
                                    reserve_space_bytes,
                                    d_reserve_space));
    }

    CUDA_CALL(cudaMemcpy(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hy_host, d_hy, state_bytes, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cy_host, d_cy, state_bytes, cudaMemcpyDeviceToHost));

#undef CUDA_CALL
#undef CUDNN_CALL

cleanup:
    if (d_reserve_space != nullptr) {
        cudaFree(d_reserve_space);
    }
    if (d_workspace != nullptr) {
        cudaFree(d_workspace);
    }
    if (d_weight_space != nullptr) {
        cudaFree(d_weight_space);
    }
    if (d_dropout_states != nullptr) {
        cudaFree(d_dropout_states);
    }
    if (d_seq_lengths != nullptr) {
        cudaFree(d_seq_lengths);
    }
    if (d_cy != nullptr) {
        cudaFree(d_cy);
    }
    if (d_hy != nullptr) {
        cudaFree(d_hy);
    }
    if (d_cx != nullptr) {
        cudaFree(d_cx);
    }
    if (d_hx != nullptr) {
        cudaFree(d_hx);
    }
    if (d_y != nullptr) {
        cudaFree(d_y);
    }
    if (d_x != nullptr) {
        cudaFree(d_x);
    }

    if (c_desc != nullptr) {
        cudnnDestroyTensorDescriptor(c_desc);
    }
    if (h_desc != nullptr) {
        cudnnDestroyTensorDescriptor(h_desc);
    }
    if (y_data_desc != nullptr) {
        cudnnDestroyRNNDataDescriptor(y_data_desc);
    }
    if (x_data_desc != nullptr) {
        cudnnDestroyRNNDataDescriptor(x_data_desc);
    }
    if (rnn_desc != nullptr) {
        cudnnDestroyRNNDescriptor(rnn_desc);
    }
    if (dropout_desc != nullptr) {
        cudnnDestroyDropoutDescriptor(dropout_desc);
    }
    if (handle != nullptr) {
        cudnnDestroy(handle);
    }

    return status;
}
