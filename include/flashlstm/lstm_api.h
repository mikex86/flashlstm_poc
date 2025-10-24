#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LSTM_COMPUTE_PRECISION_FP16_ACC32 = 0,
    LSTM_COMPUTE_PRECISION_FP16_ACC16 = 1,
} lstm_compute_precision_t;

typedef enum {
    LSTM_EXECUTION_MODE_IMMEDIATE = 0,
    LSTM_EXECUTION_MODE_GRAPH = 1,
} lstm_execution_mode_t;

typedef struct lstm_buffers {
    void* impl;
} lstm_buffers;

int lstm_create_buffers(lstm_compute_precision_t precision,
                        std::size_t seq_len,
                        std::size_t batch,
                        std::size_t input_size,
                        std::size_t hidden_size,
                        int input_proj_chunk_size,
                        lstm_buffers* buffers);

int lstm_destroy_buffers(lstm_buffers* buffers);

int lstm_set_execution_mode(lstm_buffers* buffers, lstm_execution_mode_t mode);

int lstm_pack_weights(lstm_compute_precision_t precision,
                      const float* weight_ih,
                      const float* weight_hh,
                      std::size_t input_size,
                      std::size_t hidden_size,
                      lstm_buffers* buffers);

// All pointer arguments must reference device-accessible memory.
int lstm_forward(const float* x,
                 const float* b_ih,
                 const float* b_hh,
                 const float* h0,
                 const float* c0,
                 float* output,
                 float* hn,
                 float* cn,
                 std::size_t seq_len,
                 std::size_t batch,
                 std::size_t input_size,
                 std::size_t hidden_size,
                 const lstm_buffers* buffers,
                 lstm_compute_precision_t precision);

#ifdef __cplusplus
}  // extern "C"
#endif
