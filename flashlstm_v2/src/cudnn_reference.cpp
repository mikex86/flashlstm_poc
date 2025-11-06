#include "cudnn_reference.hpp"
#include "gputx.h"

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_adv.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
    inline void CheckCuda(cudaError_t status, const char *what) {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
            std::abort();
        }
    }

    inline void CheckCudnn(cudnnStatus_t status, const char *what) {
        if (status != CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr, "%s failed: %s\n", what, cudnnGetErrorString(status));
            std::abort();
        }
    }

    struct CudnnRnnContext {
        size_t time_steps{};
        size_t batch_size{};
        size_t input_size{};
        size_t hidden_size{};

        cudnnHandle_t handle{};
        cudnnDropoutDescriptor_t dropout_desc{};
        cudnnRNNDescriptor_t rnn_desc{};
        cudnnRNNDataDescriptor_t x_data_desc{};
        cudnnRNNDataDescriptor_t y_data_desc{};
        cudnnTensorDescriptor_t h_state_desc{};
        cudnnTensorDescriptor_t c_state_desc{};

        void *dropout_states_dev{};
        float *x_dev{};
        float *y_dev{};
        float *hx_dev{};
        float *cx_dev{};
        float *hy_dev{};
        float *cy_dev{};
        float *weight_space_dev{};
        void *workspace_dev{};
        void *reserve_space_dev{};
        int32_t *seq_lengths_dev{};

        size_t dropout_states_size{};
        size_t workspace_size{};
        size_t reserve_size{};
        size_t weight_space_size{};
    };

    std::vector<float> ConvertHalfBufferToFloat(const __half *src, size_t count) {
        std::vector<float> dst(count);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = __half2float(src[i]);
        }
        return dst;
    }

    float ComputeMaxAbsDelta(const std::vector<float> &a, const std::vector<float> &b) {
        const size_t n = std::min(a.size(), b.size());
        float max_delta = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            max_delta = std::max(max_delta, std::fabs(a[i] - b[i]));
        }
        return max_delta;
    }

    CudnnRnnContext CreateCudnnRnnContext(
        size_t time_steps,
        size_t batch_size,
        size_t input_size,
        size_t hidden_size,
        const float *x_host_float,
        const float *weight_ih_host,
        const float *weight_hh_host,
        const float *bias_ih_host,
        const float *bias_hh_host,
        const float *h0_host_float,
        const float *c0_host_float
    ) {
        const size_t gate_dim = 4 * hidden_size;
        const size_t x_elements = time_steps * batch_size * input_size;
        const size_t state_elements = batch_size * hidden_size;

        CudnnRnnContext ctx{};
        ctx.time_steps = time_steps;
        ctx.batch_size = batch_size;
        ctx.input_size = input_size;
        ctx.hidden_size = hidden_size;

        CheckCudnn(cudnnCreate(&ctx.handle), "cudnnCreate");

        CheckCudnn(cudnnCreateDropoutDescriptor(&ctx.dropout_desc), "cudnnCreateDropoutDescriptor");
        CheckCudnn(cudnnDropoutGetStatesSize(ctx.handle, &ctx.dropout_states_size), "cudnnDropoutGetStatesSize");
        if (ctx.dropout_states_size > 0) {
            CheckCuda(cudaMalloc(&ctx.dropout_states_dev, ctx.dropout_states_size), "cudaMalloc dropout states");
        }
        CheckCudnn(
            cudnnSetDropoutDescriptor(
                ctx.dropout_desc,
                ctx.handle,
                0.0f,
                ctx.dropout_states_dev,
                ctx.dropout_states_size,
                0
            ),
            "cudnnSetDropoutDescriptor"
        );

        CheckCudnn(cudnnCreateRNNDescriptor(&ctx.rnn_desc), "cudnnCreateRNNDescriptor");
        CheckCudnn(
            cudnnSetRNNDescriptor_v8(
                ctx.rnn_desc,
                CUDNN_RNN_ALGO_STANDARD,
                CUDNN_LSTM,
                CUDNN_RNN_DOUBLE_BIAS,
                CUDNN_UNIDIRECTIONAL,
                CUDNN_LINEAR_INPUT,
                CUDNN_DATA_FLOAT,
                CUDNN_DATA_FLOAT,
                CUDNN_DEFAULT_MATH,
                static_cast<int>(input_size),
                static_cast<int>(hidden_size),
                static_cast<int>(hidden_size),
                1,
                ctx.dropout_desc,
                CUDNN_RNN_PADDED_IO_DISABLED
            ),
            "cudnnSetRNNDescriptor_v8"
        );
        CheckCudnn(cudnnBuildRNNDynamic(ctx.handle, ctx.rnn_desc, static_cast<int>(batch_size)),
                   "cudnnBuildRNNDynamic");

        CheckCudnn(cudnnCreateRNNDataDescriptor(&ctx.x_data_desc), "cudnnCreateRNNDataDescriptor x");
        CheckCudnn(cudnnCreateRNNDataDescriptor(&ctx.y_data_desc), "cudnnCreateRNNDataDescriptor y");
        std::vector<int> seq_lengths(batch_size, static_cast<int>(time_steps));
        CheckCudnn(
            cudnnSetRNNDataDescriptor(
                ctx.x_data_desc,
                CUDNN_DATA_FLOAT,
                CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                static_cast<int>(time_steps),
                static_cast<int>(batch_size),
                static_cast<int>(input_size),
                seq_lengths.data(),
                nullptr
            ),
            "cudnnSetRNNDataDescriptor x"
        );
        CheckCudnn(
            cudnnSetRNNDataDescriptor(
                ctx.y_data_desc,
                CUDNN_DATA_FLOAT,
                CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                static_cast<int>(time_steps),
                static_cast<int>(batch_size),
                static_cast<int>(hidden_size),
                seq_lengths.data(),
                nullptr
            ),
            "cudnnSetRNNDataDescriptor y"
        );

        CheckCudnn(cudnnCreateTensorDescriptor(&ctx.h_state_desc), "cudnnCreateTensorDescriptor h_state");
        CheckCudnn(cudnnCreateTensorDescriptor(&ctx.c_state_desc), "cudnnCreateTensorDescriptor c_state");
        int state_dims[3] = {1, static_cast<int>(batch_size), static_cast<int>(hidden_size)};
        int state_strides[3] = {static_cast<int>(batch_size * hidden_size), static_cast<int>(hidden_size), 1};
        CheckCudnn(cudnnSetTensorNdDescriptor(ctx.h_state_desc, CUDNN_DATA_FLOAT, 3, state_dims, state_strides),
                   "cudnnSetTensorNdDescriptor h_state");
        CheckCudnn(cudnnSetTensorNdDescriptor(ctx.c_state_desc, CUDNN_DATA_FLOAT, 3, state_dims, state_strides),
                   "cudnnSetTensorNdDescriptor c_state");

        CheckCuda(cudaMalloc(reinterpret_cast<void **>(&ctx.x_dev), x_elements * sizeof(float)), "cudaMalloc x_dev");
        CheckCuda(cudaMalloc(reinterpret_cast<void **>(&ctx.y_dev),
                             time_steps * batch_size * hidden_size * sizeof(float)),
                  "cudaMalloc y_dev");
        CheckCuda(cudaMalloc(reinterpret_cast<void **>(&ctx.hx_dev), state_elements * sizeof(float)),
                  "cudaMalloc hx_dev");
        CheckCuda(cudaMalloc(reinterpret_cast<void **>(&ctx.cx_dev), state_elements * sizeof(float)),
                  "cudaMalloc cx_dev");
        CheckCuda(cudaMalloc(reinterpret_cast<void **>(&ctx.hy_dev), state_elements * sizeof(float)),
                  "cudaMalloc hy_dev");
        CheckCuda(cudaMalloc(reinterpret_cast<void **>(&ctx.cy_dev), state_elements * sizeof(float)),
                  "cudaMalloc cy_dev");

        CheckCuda(cudaMemcpy(ctx.x_dev, x_host_float, x_elements * sizeof(float), cudaMemcpyHostToDevice),
                  "memcpy x_dev");
        CheckCuda(cudaMemcpy(ctx.hx_dev, h0_host_float, state_elements * sizeof(float), cudaMemcpyHostToDevice),
                  "memcpy hx_dev");
        CheckCuda(cudaMemcpy(ctx.cx_dev, c0_host_float, state_elements * sizeof(float), cudaMemcpyHostToDevice),
                  "memcpy cx_dev");

        CheckCudnn(
            cudnnGetRNNTempSpaceSizes(
                ctx.handle,
                ctx.rnn_desc,
                CUDNN_FWD_MODE_TRAINING,
                ctx.x_data_desc,
                &ctx.workspace_size,
                &ctx.reserve_size
            ),
            "cudnnGetRNNTempSpaceSizes"
        );
        if (ctx.workspace_size > 0) {
            CheckCuda(cudaMalloc(&ctx.workspace_dev, ctx.workspace_size), "cudaMalloc workspace_dev");
        }
        if (ctx.reserve_size > 0) {
            CheckCuda(cudaMalloc(&ctx.reserve_space_dev, ctx.reserve_size), "cudaMalloc reserve_space_dev");
        }

        CheckCudnn(cudnnGetRNNWeightSpaceSize(ctx.handle, ctx.rnn_desc, &ctx.weight_space_size),
                   "cudnnGetRNNWeightSpaceSize");
        if (ctx.weight_space_size > 0) {
            CheckCuda(cudaMalloc(reinterpret_cast<void **>(&ctx.weight_space_dev), ctx.weight_space_size),
                      "cudaMalloc weight_space_dev");
            CheckCuda(cudaMemset(ctx.weight_space_dev, 0, ctx.weight_space_size), "memset weight_space_dev");
        }

        auto set_params = [&](int lin_layer_id, bool is_input_weight, int gate_offset) {
            cudnnTensorDescriptor_t mat_desc;
            cudnnTensorDescriptor_t bias_desc;
            CheckCudnn(cudnnCreateTensorDescriptor(&mat_desc), "cudnnCreateTensorDescriptor mat");
            CheckCudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias");
            void *mat_ptr = nullptr;
            void *bias_ptr = nullptr;
            CheckCudnn(
                cudnnGetRNNWeightParams(
                    ctx.handle,
                    ctx.rnn_desc,
                    0,
                    ctx.weight_space_size,
                    static_cast<const void *>(ctx.weight_space_dev),
                    lin_layer_id,
                    mat_desc,
                    &mat_ptr,
                    bias_desc,
                    &bias_ptr
                ),
                "cudnnGetRNNWeightParams"
            );
            int mat_nb_dims = 0;
            int mat_dim_a[CUDNN_DIM_MAX];
            int mat_stride_a[CUDNN_DIM_MAX];
            cudnnDataType_t mat_data_type;
            CheckCudnn(cudnnGetTensorNdDescriptor(mat_desc, CUDNN_DIM_MAX, &mat_data_type, &mat_nb_dims, mat_dim_a,
                                                  mat_stride_a),
                       "cudnnGetTensorNdDescriptor mat");
            int mat_rows = 0;
            int mat_cols = 0;
            if (mat_nb_dims >= 3) {
                mat_rows = mat_dim_a[1];
                mat_cols = mat_dim_a[2];
            } else if (mat_nb_dims == 2) {
                mat_rows = mat_dim_a[0];
                mat_cols = mat_dim_a[1];
            } else {
                mat_rows = mat_dim_a[0];
                mat_cols = 1;
            }
            const float *weight_src = is_input_weight ? weight_ih_host : weight_hh_host;
            const int src_cols = is_input_weight ? static_cast<int>(input_size) : static_cast<int>(hidden_size);
            std::vector<float> host_matrix(static_cast<size_t>(mat_rows) * mat_cols, 0.0f);
            for (int r = 0; r < mat_rows; ++r) {
                const size_t src_row = static_cast<size_t>(gate_offset) * hidden_size + r;
                for (int c = 0; c < mat_cols; ++c) {
                    host_matrix[static_cast<size_t>(r) * mat_cols + c] = weight_src[src_row * src_cols + c];
                }
            }
            CheckCuda(cudaMemcpy(mat_ptr, host_matrix.data(), host_matrix.size() * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      "memcpy weight mat");

            int bias_nb_dims = 0;
            int bias_dim_a[CUDNN_DIM_MAX];
            int bias_stride_a[CUDNN_DIM_MAX];
            cudnnDataType_t bias_data_type;
            CheckCudnn(cudnnGetTensorNdDescriptor(bias_desc, CUDNN_DIM_MAX, &bias_data_type, &bias_nb_dims, bias_dim_a,
                                                  bias_stride_a),
                       "cudnnGetTensorNdDescriptor bias");
            size_t bias_elems = 1;
            for (int i = 0; i < bias_nb_dims; ++i) {
                bias_elems *= static_cast<size_t>(bias_dim_a[i]);
            }
            const float *bias_src = is_input_weight ? bias_ih_host : bias_hh_host;
            std::vector<float> host_bias(bias_elems, 0.0f);
            for (size_t r = 0; r < hidden_size && r < bias_elems; ++r) {
                host_bias[r] = bias_src[static_cast<size_t>(gate_offset) * hidden_size + r];
            }
            CheckCuda(cudaMemcpy(bias_ptr, host_bias.data(), host_bias.size() * sizeof(float), cudaMemcpyHostToDevice),
                      "memcpy bias");
            CheckCudnn(cudnnDestroyTensorDescriptor(mat_desc), "cudnnDestroyTensorDescriptor mat");
            CheckCudnn(cudnnDestroyTensorDescriptor(bias_desc), "cudnnDestroyTensorDescriptor bias");
        };

        for (int gate = 0; gate < 4; ++gate) {
            set_params(gate, true, gate);
            set_params(gate + 4, false, gate);
        }

        CheckCuda(cudaMalloc(reinterpret_cast<void **>(&ctx.seq_lengths_dev), batch_size * sizeof(int32_t)),
                  "cudaMalloc seq_lengths_dev");
        CheckCuda(cudaMemcpy(ctx.seq_lengths_dev, seq_lengths.data(), batch_size * sizeof(int32_t),
                             cudaMemcpyHostToDevice),
                  "memcpy seq_lengths_dev");

        return ctx;
    }

    void DestroyCudnnRnnContext(CudnnRnnContext &ctx) {
        if (ctx.seq_lengths_dev != nullptr) {
            cudaFree(ctx.seq_lengths_dev);
        }
        if (ctx.weight_space_dev != nullptr) {
            cudaFree(ctx.weight_space_dev);
        }
        if (ctx.reserve_space_dev != nullptr) {
            cudaFree(ctx.reserve_space_dev);
        }
        if (ctx.workspace_dev != nullptr) {
            cudaFree(ctx.workspace_dev);
        }
        if (ctx.cy_dev != nullptr) {
            cudaFree(ctx.cy_dev);
        }
        if (ctx.hy_dev != nullptr) {
            cudaFree(ctx.hy_dev);
        }
        if (ctx.cx_dev != nullptr) {
            cudaFree(ctx.cx_dev);
        }
        if (ctx.hx_dev != nullptr) {
            cudaFree(ctx.hx_dev);
        }
        if (ctx.y_dev != nullptr) {
            cudaFree(ctx.y_dev);
        }
        if (ctx.x_dev != nullptr) {
            cudaFree(ctx.x_dev);
        }
        if (ctx.dropout_states_dev != nullptr) {
            cudaFree(ctx.dropout_states_dev);
        }
        if (ctx.c_state_desc != nullptr) {
            cudnnDestroyTensorDescriptor(ctx.c_state_desc);
        }
        if (ctx.h_state_desc != nullptr) {
            cudnnDestroyTensorDescriptor(ctx.h_state_desc);
        }
        if (ctx.y_data_desc != nullptr) {
            cudnnDestroyRNNDataDescriptor(ctx.y_data_desc);
        }
        if (ctx.x_data_desc != nullptr) {
            cudnnDestroyRNNDataDescriptor(ctx.x_data_desc);
        }
        if (ctx.rnn_desc != nullptr) {
            cudnnDestroyRNNDescriptor(ctx.rnn_desc);
        }
        if (ctx.dropout_desc != nullptr) {
            cudnnDestroyDropoutDescriptor(ctx.dropout_desc);
        }
        if (ctx.handle != nullptr) {
            cudnnDestroy(ctx.handle);
        }
    }

    void ComputeFinalCellStates(
        size_t time_steps,
        size_t batch_size,
        size_t hidden_size,
        const float *c0_host_float,
        const std::vector<float> &gate_cache_float,
        std::vector<float> &c_stream_final
    ) {
        const size_t gate_dim = 4 * hidden_size;
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < hidden_size; ++h) {
                float c_prev = c0_host_float[b * hidden_size + h];
                for (size_t t = 0; t < time_steps; ++t) {
                    const size_t gate_base = (t * batch_size + b) * gate_dim + h;
                    const float i_gate = gate_cache_float[gate_base + 0 * hidden_size];
                    const float f_gate = gate_cache_float[gate_base + 1 * hidden_size];
                    const float g_gate = gate_cache_float[gate_base + 2 * hidden_size];
                    c_prev = f_gate * c_prev + i_gate * g_gate;
                }
                c_stream_final[b * hidden_size + h] = c_prev;
            }
        }
    }
} // namespace

CudnnForwardComparisonResult RunCudnnForwardComparison(
    size_t time_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,
    const float *x_host_float,
    const float *weight_ih_host,
    const float *weight_hh_host,
    const float *bias_ih_host,
    const float *bias_hh_host,
    const float *h0_host_float,
    const float *c0_host_float,
    const __half *y_host,
    const __half *gate_cache_host
) {
    const size_t y_elements = time_steps * batch_size * hidden_size;
    const size_t gate_dim = 4 * hidden_size;
    const size_t gate_elements = time_steps * batch_size * gate_dim;
    const size_t state_elements = batch_size * hidden_size;

    std::vector<float> y_stream_float = ConvertHalfBufferToFloat(y_host, y_elements);
    std::vector<float> gate_cache_float = ConvertHalfBufferToFloat(gate_cache_host, gate_elements);

    CudnnRnnContext ctx = CreateCudnnRnnContext(
        time_steps,
        batch_size,
        input_size,
        hidden_size,
        x_host_float,
        weight_ih_host,
        weight_hh_host,
        bias_ih_host,
        bias_hh_host,
        h0_host_float,
        c0_host_float
    ); {
        GPUTX_RANGE("cuDNN::RNNForward[warmup]");
        CheckCudnn(
            cudnnRNNForward(
                ctx.handle,
                ctx.rnn_desc,
                CUDNN_FWD_MODE_TRAINING,
                ctx.seq_lengths_dev,
                ctx.x_data_desc,
                ctx.x_dev,
                ctx.y_data_desc,
                ctx.y_dev,
                ctx.h_state_desc,
                ctx.hx_dev,
                ctx.hy_dev,
                ctx.c_state_desc,
                ctx.cx_dev,
                ctx.cy_dev,
                ctx.weight_space_size,
                ctx.weight_space_dev,
                ctx.workspace_size,
                ctx.workspace_dev,
                ctx.reserve_size,
                ctx.reserve_space_dev
            ),
            "cudnnRNNForward warmup"
        );
        CheckCuda(cudaStreamSynchronize(nullptr), "sync after cudnn forward warmup");
    } {
        GPUTX_RANGE("cuDNN::RNNForward");
        CheckCudnn(
            cudnnRNNForward(
                ctx.handle,
                ctx.rnn_desc,
                CUDNN_FWD_MODE_TRAINING,
                ctx.seq_lengths_dev,
                ctx.x_data_desc,
                ctx.x_dev,
                ctx.y_data_desc,
                ctx.y_dev,
                ctx.h_state_desc,
                ctx.hx_dev,
                ctx.hy_dev,
                ctx.c_state_desc,
                ctx.cx_dev,
                ctx.cy_dev,
                ctx.weight_space_size,
                ctx.weight_space_dev,
                ctx.workspace_size,
                ctx.workspace_dev,
                ctx.reserve_size,
                ctx.reserve_space_dev
            ),
            "cudnnRNNForward"
        );
        CheckCuda(cudaStreamSynchronize(nullptr), "sync after cudnn forward");
    }

    std::vector<float> y_cudnn_host(y_elements);
    std::vector<float> hy_cudnn_host(state_elements);
    std::vector<float> cy_cudnn_host(state_elements);
    CheckCuda(cudaMemcpy(y_cudnn_host.data(), ctx.y_dev, y_elements * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy y_dev");
    CheckCuda(cudaMemcpy(hy_cudnn_host.data(), ctx.hy_dev, state_elements * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy hy_dev");
    CheckCuda(cudaMemcpy(cy_cudnn_host.data(), ctx.cy_dev, state_elements * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy cy_dev");

    std::vector<float> c_stream_final(state_elements, 0.0f);
    ComputeFinalCellStates(time_steps, batch_size, hidden_size, c0_host_float, gate_cache_float, c_stream_final);

    const size_t last_step_offset = (time_steps - 1) * batch_size * hidden_size;
    std::vector<float> h_stream_final(state_elements);
    for (size_t i = 0; i < state_elements; ++i) {
        h_stream_final[i] = y_stream_float[last_step_offset + i];
    }

    CudnnForwardComparisonResult result{};
    result.max_y_delta = ComputeMaxAbsDelta(y_stream_float, y_cudnn_host);
    result.max_h_delta = ComputeMaxAbsDelta(h_stream_final, hy_cudnn_host);
    result.max_c_delta = ComputeMaxAbsDelta(c_stream_final, cy_cudnn_host);

    DestroyCudnnRnnContext(ctx);
    return result;
}

CudnnBackwardComparisonResult RunCudnnBackwardComparison(
    size_t time_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,
    const float *x_host_float,
    const float *weight_ih_host,
    const float *weight_hh_host,
    const float *bias_ih_host,
    const float *bias_hh_host,
    const float *h0_host_float,
    const float *c0_host_float,
    const __half *y_host,
    const __half *gate_cache_host,
    const __half *dY_host,
    const std::vector<__half> &dHN_host_half,
    const std::vector<__half> &dCN_host_half,
    const __half *dx_host_half,
    const float *dW_ih_device,
    const float *dW_hh_device,
    const float *db_ih_device,
    const float *db_hh_device,
    const float *dh0_device,
    const float *dc0_device
) {
    (void) gate_cache_host;
    const size_t gate_dim = 4 * hidden_size;
    const size_t x_elements = time_steps * batch_size * input_size;
    const size_t y_elements = time_steps * batch_size * hidden_size;
    const size_t state_elements = batch_size * hidden_size;

    CudnnRnnContext ctx = CreateCudnnRnnContext(
        time_steps,
        batch_size,
        input_size,
        hidden_size,
        x_host_float,
        weight_ih_host,
        weight_hh_host,
        bias_ih_host,
        bias_hh_host,
        h0_host_float,
        c0_host_float
    );

    // Warmup forward
    {
        GPUTX_RANGE("cuDNN::RNNForward[warmup]");
        CheckCudnn(
            cudnnRNNForward(
                ctx.handle,
                ctx.rnn_desc,
                CUDNN_FWD_MODE_TRAINING,
                ctx.seq_lengths_dev,
                ctx.x_data_desc,
                ctx.x_dev,
                ctx.y_data_desc,
                ctx.y_dev,
                ctx.h_state_desc,
                ctx.hx_dev,
                ctx.hy_dev,
                ctx.c_state_desc,
                ctx.cx_dev,
                ctx.cy_dev,
                ctx.weight_space_size,
                ctx.weight_space_dev,
                ctx.workspace_size,
                ctx.workspace_dev,
                ctx.reserve_size,
                ctx.reserve_space_dev
            ),
            "cudnnRNNForward warmup"
        );
        CheckCuda(cudaStreamSynchronize(nullptr), "sync after cudnn forward warmup");
    }

    // Actual forward pass to populate reserve space and outputs
    {
        GPUTX_RANGE("cuDNN::RNNForward");
        CheckCudnn(
            cudnnRNNForward(
                ctx.handle,
                ctx.rnn_desc,
                CUDNN_FWD_MODE_TRAINING,
                ctx.seq_lengths_dev,
                ctx.x_data_desc,
                ctx.x_dev,
                ctx.y_data_desc,
                ctx.y_dev,
                ctx.h_state_desc,
                ctx.hx_dev,
                ctx.hy_dev,
                ctx.c_state_desc,
                ctx.cx_dev,
                ctx.cy_dev,
                ctx.weight_space_size,
                ctx.weight_space_dev,
                ctx.workspace_size,
                ctx.workspace_dev,
                ctx.reserve_size,
                ctx.reserve_space_dev
            ),
            "cudnnRNNForward"
        );
        CheckCuda(cudaStreamSynchronize(nullptr), "sync after cudnn forward");
    }

    // Prepare upstream gradients
    std::vector<float> dY_host_float = ConvertHalfBufferToFloat(dY_host, y_elements);
    float *dY_dev = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&dY_dev), y_elements * sizeof(float)), "cudaMalloc dY_dev");
    CheckCuda(cudaMemcpy(dY_dev, dY_host_float.data(), y_elements * sizeof(float), cudaMemcpyHostToDevice),
              "memcpy dY_dev");

    std::vector<float> dHN_float(dHN_host_half.size());
    for (size_t i = 0; i < dHN_host_half.size(); ++i) {
        dHN_float[i] = __half2float(dHN_host_half[i]);
    }
    std::vector<float> dCN_float(dCN_host_half.size());
    for (size_t i = 0; i < dCN_host_half.size(); ++i) {
        dCN_float[i] = __half2float(dCN_host_half[i]);
    }

    float *dhy_dev = nullptr;
    float *dcy_dev = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&dhy_dev), state_elements * sizeof(float)), "cudaMalloc dhy_dev");
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&dcy_dev), state_elements * sizeof(float)), "cudaMalloc dcy_dev");
    CheckCuda(cudaMemcpy(dhy_dev, dHN_float.data(), state_elements * sizeof(float), cudaMemcpyHostToDevice),
              "memcpy dhy_dev");
    CheckCuda(cudaMemcpy(dcy_dev, dCN_float.data(), state_elements * sizeof(float), cudaMemcpyHostToDevice),
              "memcpy dcy_dev");

    float *dx_dev = nullptr;
    float *dhx_dev = nullptr;
    float *dcx_dev = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&dx_dev), x_elements * sizeof(float)), "cudaMalloc dx_dev");
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&dhx_dev), state_elements * sizeof(float)), "cudaMalloc dhx_dev");
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&dcx_dev), state_elements * sizeof(float)), "cudaMalloc dcx_dev");

    float *dweight_space_dev = nullptr;
    if (ctx.weight_space_size > 0) {
        CheckCuda(cudaMalloc(reinterpret_cast<void **>(&dweight_space_dev), ctx.weight_space_size),
                  "cudaMalloc dweight_space_dev");
        CheckCuda(cudaMemset(dweight_space_dev, 0, ctx.weight_space_size), "memset dweight_space_dev");
    }

    // Warm up backward kernels to trigger any internal cuDNN initialisation.
    CheckCudnn(
        cudnnRNNBackwardData_v8(
            ctx.handle,
            ctx.rnn_desc,
            ctx.seq_lengths_dev,
            ctx.y_data_desc,
            ctx.y_dev,
            dY_dev,
            ctx.x_data_desc,
            dx_dev,
            ctx.h_state_desc,
            ctx.hx_dev,
            dhy_dev,
            dhx_dev,
            ctx.c_state_desc,
            ctx.cx_dev,
            dcy_dev,
            dcx_dev,
            ctx.weight_space_size,
            ctx.weight_space_dev,
            ctx.workspace_size,
            ctx.workspace_dev,
            ctx.reserve_size,
            ctx.reserve_space_dev
        ),
        "cudnnRNNBackwardData_v8 warmup"
    );
    CheckCudnn(
        cudnnRNNBackwardWeights_v8(
            ctx.handle,
            ctx.rnn_desc,
            CUDNN_WGRAD_MODE_ADD,
            ctx.seq_lengths_dev,
            ctx.x_data_desc,
            ctx.x_dev,
            ctx.h_state_desc,
            ctx.hx_dev,
            ctx.y_data_desc,
            ctx.y_dev,
            ctx.weight_space_size,
            dweight_space_dev,
            ctx.workspace_size,
            ctx.workspace_dev,
            ctx.reserve_size,
            ctx.reserve_space_dev
        ),
        "cudnnRNNBackwardWeights_v8 warmup"
    );
    CheckCuda(cudaStreamSynchronize(nullptr), "sync after cudnn backward warmup");
    if (ctx.weight_space_size > 0) {
        CheckCuda(cudaMemset(dweight_space_dev, 0, ctx.weight_space_size), "memset dweight_space_dev");
    }
    CheckCuda(cudaMemset(dx_dev, 0, x_elements * sizeof(float)), "memset dx_dev");
    CheckCuda(cudaMemset(dhx_dev, 0, state_elements * sizeof(float)), "memset dhx_dev");
    CheckCuda(cudaMemset(dcx_dev, 0, state_elements * sizeof(float)), "memset dcx_dev");

    // Recompute forward pass to refresh reserve space after warm-up mutations.
    {
        GPUTX_RANGE("cuDNN::RNNForward");
        CheckCudnn(
            cudnnRNNForward(
                ctx.handle,
                ctx.rnn_desc,
                CUDNN_FWD_MODE_TRAINING,
                ctx.seq_lengths_dev,
                ctx.x_data_desc,
                ctx.x_dev,
                ctx.y_data_desc,
                ctx.y_dev,
                ctx.h_state_desc,
                ctx.hx_dev,
                ctx.hy_dev,
                ctx.c_state_desc,
                ctx.cx_dev,
                ctx.cy_dev,
                ctx.weight_space_size,
                ctx.weight_space_dev,
                ctx.workspace_size,
                ctx.workspace_dev,
                ctx.reserve_size,
                ctx.reserve_space_dev
            ),
            "cudnnRNNForward refresh"
        );
        CheckCuda(cudaStreamSynchronize(nullptr), "sync after cudnn forward refresh");
    } {
        GPUTX_RANGE("cuDNN::RNNBackwardData");
        CheckCudnn(
            cudnnRNNBackwardData_v8(
                ctx.handle,
                ctx.rnn_desc,
                ctx.seq_lengths_dev,
                ctx.y_data_desc,
                ctx.y_dev,
                dY_dev,
                ctx.x_data_desc,
                dx_dev,
                ctx.h_state_desc,
                ctx.hx_dev,
                dhy_dev,
                dhx_dev,
                ctx.c_state_desc,
                ctx.cx_dev,
                dcy_dev,
                dcx_dev,
                ctx.weight_space_size,
                ctx.weight_space_dev,
                ctx.workspace_size,
                ctx.workspace_dev,
                ctx.reserve_size,
                ctx.reserve_space_dev
            ),
            "cudnnRNNBackwardData_v8"
        );
        CheckCuda(cudaStreamSynchronize(nullptr), "sync after cudnn backward");
    } {
        GPUTX_RANGE("cuDNN::RNNBackwardWeights");
        CheckCudnn(
            cudnnRNNBackwardWeights_v8(
                ctx.handle,
                ctx.rnn_desc,
                CUDNN_WGRAD_MODE_ADD,
                ctx.seq_lengths_dev,
                ctx.x_data_desc,
                ctx.x_dev,
                ctx.h_state_desc,
                ctx.hx_dev,
                ctx.y_data_desc,
                ctx.y_dev,
                ctx.weight_space_size,
                dweight_space_dev,
                ctx.workspace_size,
                ctx.workspace_dev,
                ctx.reserve_size,
                ctx.reserve_space_dev
            ),
            "cudnnRNNBackwardWeights_v8"
        );
        CheckCuda(cudaStreamSynchronize(nullptr), "sync after cudnn backward");
    }
    // Copy cuDNN outputs to host
    std::vector<float> dx_cudnn(x_elements);
    std::vector<float> dh0_cudnn(state_elements);
    std::vector<float> dc0_cudnn(state_elements);
    CheckCuda(cudaMemcpy(dx_cudnn.data(), dx_dev, x_elements * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy dx_dev");
    CheckCuda(cudaMemcpy(dh0_cudnn.data(), dhx_dev, state_elements * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy dhx_dev");
    CheckCuda(cudaMemcpy(dc0_cudnn.data(), dcx_dev, state_elements * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy dcx_dev");

    std::vector<float> dW_ih_cudnn(gate_dim * input_size, 0.0f);
    std::vector<float> dW_hh_cudnn(gate_dim * hidden_size, 0.0f);
    std::vector<float> db_ih_cudnn(gate_dim, 0.0f);
    std::vector<float> db_hh_cudnn(gate_dim, 0.0f);

    auto gather_params = [&](int lin_layer_id, bool is_input_weight, int gate_offset) {
        cudnnTensorDescriptor_t mat_desc;
        cudnnTensorDescriptor_t bias_desc;
        CheckCudnn(cudnnCreateTensorDescriptor(&mat_desc), "cudnnCreateTensorDescriptor mat");
        CheckCudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias");
        void *mat_ptr = nullptr;
        void *bias_ptr = nullptr;
        CheckCudnn(
            cudnnGetRNNWeightParams(
                ctx.handle,
                ctx.rnn_desc,
                0,
                ctx.weight_space_size,
                dweight_space_dev,
                lin_layer_id,
                mat_desc,
                &mat_ptr,
                bias_desc,
                &bias_ptr
            ),
            "cudnnGetRNNWeightParams dgrad"
        );
        int mat_nb_dims = 0;
        int mat_dim_a[CUDNN_DIM_MAX];
        int mat_stride_a[CUDNN_DIM_MAX];
        cudnnDataType_t mat_data_type;
        CheckCudnn(cudnnGetTensorNdDescriptor(mat_desc, CUDNN_DIM_MAX, &mat_data_type, &mat_nb_dims, mat_dim_a,
                                              mat_stride_a),
                   "cudnnGetTensorNdDescriptor grad mat");
        int mat_rows = 0;
        int mat_cols = 0;
        if (mat_nb_dims >= 3) {
            mat_rows = mat_dim_a[1];
            mat_cols = mat_dim_a[2];
        } else if (mat_nb_dims == 2) {
            mat_rows = mat_dim_a[0];
            mat_cols = mat_dim_a[1];
        } else {
            mat_rows = mat_dim_a[0];
            mat_cols = 1;
        }
        std::vector<float> host_matrix(static_cast<size_t>(mat_rows) * mat_cols, 0.0f);
        CheckCuda(cudaMemcpy(host_matrix.data(), mat_ptr, host_matrix.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "memcpy grad mat");
        std::vector<float> &dest = is_input_weight ? dW_ih_cudnn : dW_hh_cudnn;
        const int dest_cols = is_input_weight ? static_cast<int>(input_size) : static_cast<int>(hidden_size);
        for (int r = 0; r < mat_rows; ++r) {
            const size_t dest_row = static_cast<size_t>(gate_offset) * hidden_size + r;
            for (int c = 0; c < mat_cols; ++c) {
                dest[dest_row * dest_cols + c] = host_matrix[static_cast<size_t>(r) * mat_cols + c];
            }
        }

        int bias_nb_dims = 0;
        int bias_dim_a[CUDNN_DIM_MAX];
        int bias_stride_a[CUDNN_DIM_MAX];
        cudnnDataType_t bias_data_type;
        CheckCudnn(cudnnGetTensorNdDescriptor(bias_desc, CUDNN_DIM_MAX, &bias_data_type, &bias_nb_dims, bias_dim_a,
                                              bias_stride_a),
                   "cudnnGetTensorNdDescriptor grad bias");
        size_t bias_elems = 1;
        for (int i = 0; i < bias_nb_dims; ++i) {
            bias_elems *= static_cast<size_t>(bias_dim_a[i]);
        }
        std::vector<float> host_bias(bias_elems, 0.0f);
        CheckCuda(cudaMemcpy(host_bias.data(), bias_ptr, host_bias.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "memcpy grad bias");
        std::vector<float> &bias_dest = is_input_weight ? db_ih_cudnn : db_hh_cudnn;
        for (size_t r = 0; r < hidden_size && r < bias_elems; ++r) {
            bias_dest[static_cast<size_t>(gate_offset) * hidden_size + r] = host_bias[r];
        }
        CheckCudnn(cudnnDestroyTensorDescriptor(mat_desc), "cudnnDestroyTensorDescriptor mat grad");
        CheckCudnn(cudnnDestroyTensorDescriptor(bias_desc), "cudnnDestroyTensorDescriptor bias grad");
    };

    for (int gate = 0; gate < 4; ++gate) {
        gather_params(gate, true, gate);
        gather_params(gate + 4, false, gate);
    }

    // Streaming outputs converted to float for comparison
    std::vector<float> dx_stream = ConvertHalfBufferToFloat(dx_host_half, x_elements);
    std::vector<float> dh0_stream(state_elements);
    std::vector<float> dc0_stream(state_elements);
    CheckCuda(cudaMemcpy(dh0_stream.data(), dh0_device, state_elements * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy dh0 stream");
    CheckCuda(cudaMemcpy(dc0_stream.data(), dc0_device, state_elements * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy dc0 stream");

    std::vector<float> dW_ih_stream(gate_dim * input_size);
    std::vector<float> dW_hh_stream(gate_dim * hidden_size);
    std::vector<float> db_ih_stream(gate_dim);
    std::vector<float> db_hh_stream(gate_dim);
    CheckCuda(
        cudaMemcpy(dW_ih_stream.data(), dW_ih_device, dW_ih_stream.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "memcpy dW_ih");
    CheckCuda(
        cudaMemcpy(dW_hh_stream.data(), dW_hh_device, dW_hh_stream.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "memcpy dW_hh");
    CheckCuda(
        cudaMemcpy(db_ih_stream.data(), db_ih_device, db_ih_stream.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "memcpy db_ih");
    CheckCuda(
        cudaMemcpy(db_hh_stream.data(), db_hh_device, db_hh_stream.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "memcpy db_hh");

    CudnnBackwardComparisonResult result{};
    result.max_dx_delta = ComputeMaxAbsDelta(dx_stream, dx_cudnn);
    result.max_dh0_delta = ComputeMaxAbsDelta(dh0_stream, dh0_cudnn);
    result.max_dc0_delta = ComputeMaxAbsDelta(dc0_stream, dc0_cudnn);
    result.max_dW_ih_delta = ComputeMaxAbsDelta(dW_ih_stream, dW_ih_cudnn);
    result.max_dW_hh_delta = ComputeMaxAbsDelta(dW_hh_stream, dW_hh_cudnn);
    result.max_db_ih_delta = ComputeMaxAbsDelta(db_ih_stream, db_ih_cudnn);
    result.max_db_hh_delta = ComputeMaxAbsDelta(db_hh_stream, db_hh_cudnn);


    if (result.max_db_ih_delta > 0.2f) {
        for (int gate = 0; gate < 4; ++gate) {
            float gate_max = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                const size_t offset = static_cast<size_t>(gate) * hidden_size + h;
                gate_max = std::max(gate_max, std::fabs(db_ih_stream[offset] - db_ih_cudnn[offset]));
            }
            std::fprintf(stderr, "[cuDNN] db_ih gate %d max delta %f\n", gate, gate_max);
        }
        const size_t gate2_offset = static_cast<size_t>(2) * hidden_size;
        const size_t samples = std::min<size_t>(8, hidden_size);
        for (size_t h = 0; h < samples; ++h) {
            const size_t idx = gate2_offset + h;
            std::fprintf(stderr, "[cuDNN] gate2 idx=%zu stream=%f cudnn=%f delta=%f\n",
                         idx,
                         db_ih_stream[idx],
                         db_ih_cudnn[idx],
                         std::fabs(db_ih_stream[idx] - db_ih_cudnn[idx]));
        }
    }

    // Cleanup temporary allocations
    if (dweight_space_dev != nullptr) {
        cudaFree(dweight_space_dev);
    }
    cudaFree(dcy_dev);
    cudaFree(dhy_dev);
    cudaFree(dx_dev);
    cudaFree(dhx_dev);
    cudaFree(dcx_dev);
    cudaFree(dY_dev);

    DestroyCudnnRnnContext(ctx);
    return result;
}
