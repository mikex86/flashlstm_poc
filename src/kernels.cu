#include "lstm_internal.h"

namespace flashlstm::kernels {
    using namespace flashlstm::internal;
    __global__ void populate_activations_kernel(const float *__restrict__ x,
                                                const float *__restrict__ h0,
                                                ElementInput *__restrict__ activations,
                                                const std::size_t seq_len,
                                                const std::size_t batch,
                                                const std::size_t input_size,
                                                const std::size_t hidden_size,
                                                const std::size_t activations_stride) {
        const std::size_t idx =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t input_elements = seq_len * batch * input_size;
        if (idx < input_elements) {
            const std::size_t features_per_step = batch * input_size;
            const std::size_t seq_idx = idx / features_per_step;
            const std::size_t batch_idx = (idx % features_per_step) / input_size;
            const std::size_t feature_idx = idx % input_size;
            const std::size_t dst_offset =
                    (seq_idx * batch + batch_idx) * activations_stride + feature_idx;
            activations[dst_offset] = ElementInput(x[idx]);
            return;
        }

        if (h0 == nullptr) {
            return;
        }

        const std::size_t hidden_elements = batch * hidden_size;
        const std::size_t hidden_idx_flat = idx - input_elements;
        if (hidden_idx_flat >= hidden_elements) {
            return;
        }

        const std::size_t batch_idx = hidden_idx_flat / hidden_size;
        const std::size_t hidden_idx = hidden_idx_flat % hidden_size;
        const std::size_t dst_offset =
                batch_idx * activations_stride + input_size + hidden_idx;

        activations[dst_offset] = ElementInput(h0[hidden_idx_flat]);
    }

    __global__ void combine_bias_kernel(const float *__restrict__ b_ih,
                                        const float *__restrict__ b_hh,
                                        float *__restrict__ bias_out,
                                        const std::size_t elements) {
        const std::size_t idx =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= elements) {
            return;
        }
        bias_out[idx] = b_ih[idx] + b_hh[idx];
    }

    __global__ void pack_weights_kernel(const float *__restrict__ weight_ih,
                                        const float *__restrict__ weight_hh,
                                        ElementInput *__restrict__ packed,
                                        const int hidden_size,
                                        const int input_size,
                                        const int input_hidden,
                                        const int input_hidden_stride) {
        const std::size_t idx =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t total =
                static_cast<std::size_t>(4 * hidden_size) * input_hidden_stride;
        if (idx >= total) {
            return;
        }

        const int stride = input_hidden_stride;
        const int gate = static_cast<int>(idx / stride);
        const int col = static_cast<int>(idx % stride);

        float value = 0.0f;
        if (col < input_size) {
            value = weight_ih[gate * input_size + col];
        } else if (col < input_hidden) {
            const int hidden_col = col - input_size;
            value = weight_hh[gate * hidden_size + hidden_col];
        }
        packed[idx] = ElementInput(value);
    }

    __global__ void lstm_pointwise_kernel(const float *gates,
                                          const int split_k_slices,
                                          const int slice_stride,
                                          const float *input_gates,
                                          const float *bias,
                                          const float *c_prev,
                                          float *c_next,
                                          float *h_next,
                                          ElementInput *activations,
                                          const int input_size,
                                          const int activations_stride,
                                          ElementInput *next_hidden_tail,
                                          float *output_t,
                                          const int batch_size,
                                          const int hidden_size,
                                          float *gates_out) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = batch_size * hidden_size;
        if (idx >= total) {
            return;
        }

        const int batch_idx = idx / hidden_size;
        const int hidden_idx = idx % hidden_size;
        const int gate_stride = hidden_size;
        const int row_offset = batch_idx * (4 * gate_stride);

        float i_gate_val = 0.0f;
        float f_gate_val = 0.0f;
        float g_gate_val = 0.0f;
        float o_gate_val = 0.0f;
        for (int slice = 0; slice < split_k_slices; ++slice) {
            const float *slice_ptr = gates + slice * slice_stride + row_offset;
            i_gate_val += slice_ptr[hidden_idx];
            f_gate_val += slice_ptr[gate_stride + hidden_idx];
            g_gate_val += slice_ptr[2 * gate_stride + hidden_idx];
            o_gate_val += slice_ptr[3 * gate_stride + hidden_idx];
        }

        if (input_gates != nullptr) {
            const float *input_row = input_gates + row_offset;
            i_gate_val += input_row[hidden_idx];
            f_gate_val += input_row[gate_stride + hidden_idx];
            g_gate_val += input_row[2 * gate_stride + hidden_idx];
            o_gate_val += input_row[3 * gate_stride + hidden_idx];
        }

        i_gate_val += bias[hidden_idx];
        f_gate_val += bias[gate_stride + hidden_idx];
        g_gate_val += bias[2 * gate_stride + hidden_idx];
        o_gate_val += bias[3 * gate_stride + hidden_idx];

        const float i_gate = sigmoidf(i_gate_val);
        const float f_gate = sigmoidf(f_gate_val);
        const float g_gate = ::tanhf(g_gate_val);
        const float o_gate = sigmoidf(o_gate_val);

        const float c_val = f_gate * c_prev[idx] + i_gate * g_gate;
        const float h_val = o_gate * ::tanhf(c_val);

        c_next[idx] = c_val;
        h_next[idx] = h_val;

        ElementInput *hidden_row = activations + batch_idx * activations_stride + input_size;
        const ElementInput h_half = ElementInput(h_val);
        hidden_row[hidden_idx] = h_half;
        if (next_hidden_tail != nullptr) {
            ElementInput *next_row = next_hidden_tail + batch_idx * activations_stride;
            next_row[hidden_idx] = h_half;
        }
        output_t[idx] = h_val;
        if (gates_out != nullptr) {
            float *out_row = gates_out + row_offset;
            out_row[hidden_idx] = i_gate_val;
            out_row[gate_stride + hidden_idx] = f_gate_val;
            out_row[2 * gate_stride + hidden_idx] = g_gate_val;
            out_row[3 * gate_stride + hidden_idx] = o_gate_val;
        }
    }
} // namespace flashlstm::internal
