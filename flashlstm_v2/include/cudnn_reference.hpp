#pragma once

#include <cstddef>
#include <vector>

#include <cuda_fp16.h>

struct CudnnForwardComparisonResult {
    float max_y_delta;
    float max_h_delta;
    float max_c_delta;
};

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
    const __half *gate_cache_host,
    const __half *hy_host,
    const __half *cy_host);

struct CudnnBackwardComparisonResult {
    float max_dx_delta;
    float max_dh0_delta;
    float max_dc0_delta;
    float max_dW_ih_delta;
    float max_dW_hh_delta;
    float max_db_ih_delta;
    float max_db_hh_delta;
};

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
    const float *dc0_device);
