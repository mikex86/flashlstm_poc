#pragma once

#include <cstddef>

int run_cudnn_lstm(const float* x_host,
                   const float* h0_host,
                   const float* c0_host,
                   float* y_host,
                   float* hy_host,
                   float* cy_host,
                   std::size_t seq_len,
                   std::size_t batch,
                   std::size_t input_size,
                   std::size_t hidden_size);

int initialize_cudnn();
