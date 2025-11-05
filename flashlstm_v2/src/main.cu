#include "lstm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void CheckCuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        std::abort();
    }
}

template<typename T>
T *CudaMallocDevice(size_t count, const char *what) {
    T *ptr = nullptr;
    CheckCuda(cudaMalloc(&ptr, count * sizeof(T)), what);
    return ptr;
}

template<typename T>
T *CudaMallocHost(size_t count, const char *what) {
    T *ptr = nullptr;
    CheckCuda(cudaMallocHost(&ptr, count * sizeof(T)), what);
    return ptr;
}

} // namespace

int main() {
    constexpr size_t time_steps = 2048;
    constexpr size_t batch_size = 32;
    constexpr size_t input_size = 1024;
    constexpr size_t hidden_size = 1024;
    constexpr size_t gate_dim = 4 * hidden_size;

    const size_t x_elements = time_steps * batch_size * input_size;
    const size_t y_elements = time_steps * batch_size * hidden_size;
    const size_t gate_elements = time_steps * batch_size * gate_dim;
    const size_t state_elements = batch_size * hidden_size;

    std::mt19937 rng(0);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    auto sample_normal = [&]() -> float { return normal_dist(rng); };

    __half *x_host = CudaMallocHost<__half>(x_elements, "cudaMallocHost x_host");
    __half *y_host = CudaMallocHost<__half>(y_elements, "cudaMallocHost y_host");
    constexpr float kInputStd = 0.01f;
    for (size_t i = 0; i < x_elements; ++i) {
        x_host[i] = __float2half(kInputStd * sample_normal());
    }

    std::vector<float> weight_ih_host(gate_dim * input_size);
    std::vector<float> weight_hh_host(gate_dim * hidden_size);
    std::vector<float> bias_ih_host(gate_dim, 0.0f);
    std::vector<float> bias_hh_host(gate_dim, 0.0f);
    const float weight_limit = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    std::uniform_real_distribution<float> weight_dist(-weight_limit, weight_limit);
    for (auto &w : weight_ih_host) { w = weight_dist(rng); }
    for (auto &w : weight_hh_host) { w = weight_dist(rng); }

    std::vector<__half> h0_host(state_elements, __float2half(0.0f));
    std::vector<__half> c0_host(state_elements, __float2half(0.0f));

    __half *h0_device = CudaMallocDevice<__half>(state_elements, "cudaMalloc h0_device");
    __half *c0_device = CudaMallocDevice<__half>(state_elements, "cudaMalloc c0_device");
    float *weight_ih_device = CudaMallocDevice<float>(weight_ih_host.size(), "cudaMalloc weight_ih");
    float *weight_hh_device = CudaMallocDevice<float>(weight_hh_host.size(), "cudaMalloc weight_hh");
    float *bias_ih_device = CudaMallocDevice<float>(bias_ih_host.size(), "cudaMalloc bias_ih");
    float *bias_hh_device = CudaMallocDevice<float>(bias_hh_host.size(), "cudaMalloc bias_hh");

    __half *gate_cache_host = CudaMallocHost<__half>(gate_elements, "cudaMallocHost gate_cache");

    CheckCuda(cudaMemcpy(h0_device, h0_host.data(), state_elements * sizeof(__half), cudaMemcpyHostToDevice),
              "memcpy h0");
    CheckCuda(cudaMemcpy(c0_device, c0_host.data(), state_elements * sizeof(__half), cudaMemcpyHostToDevice),
              "memcpy c0");
    CheckCuda(cudaMemcpy(weight_ih_device, weight_ih_host.data(), weight_ih_host.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "memcpy weight_ih");
    CheckCuda(cudaMemcpy(weight_hh_device, weight_hh_host.data(), weight_hh_host.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "memcpy weight_hh");
    CheckCuda(cudaMemcpy(bias_ih_device, bias_ih_host.data(), bias_ih_host.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "memcpy bias_ih");
    CheckCuda(cudaMemcpy(bias_hh_device, bias_hh_host.data(), bias_hh_host.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "memcpy bias_hh");

    cudaStream_t compute_stream{};
    cudaStream_t h2d_stream{};
    cudaStream_t d2h_stream{};
    CheckCuda(cudaStreamCreate(&compute_stream), "cudaStreamCreate compute");
    CheckCuda(cudaStreamCreate(&h2d_stream), "cudaStreamCreate h2d");
    CheckCuda(cudaStreamCreate(&d2h_stream), "cudaStreamCreate d2h");

    flstm_StreamingLstmForward(
        time_steps,
        batch_size,
        input_size,
        hidden_size,
        x_host,
        h0_device,
        c0_device,
        weight_ih_device,
        weight_hh_device,
        bias_ih_device,
        bias_hh_device,
        y_host,
        gate_cache_host,
        compute_stream,
        h2d_stream,
        d2h_stream
    );
    CheckCuda(cudaDeviceSynchronize(), "forward synchronize");

    // Prepare backward inputs
    __half *dY_host = CudaMallocHost<__half>(y_elements, "cudaMallocHost dY_host");
    std::vector<__half> dHN_host(state_elements);
    std::vector<__half> dCN_host(state_elements);
    constexpr float kGradStd = 0.005f;
    for (size_t i = 0; i < y_elements; ++i) {
        dY_host[i] = __float2half(kGradStd * sample_normal());
    }
    for (size_t i = 0; i < state_elements; ++i) {
        dHN_host[i] = __float2half(kGradStd * sample_normal());
        dCN_host[i] = __float2half(kGradStd * sample_normal());
    }

    __half *dHN_device = CudaMallocDevice<__half>(state_elements, "cudaMalloc dHN_device");
    __half *dCN_device = CudaMallocDevice<__half>(state_elements, "cudaMalloc dCN_device");
    CheckCuda(cudaMemcpy(dHN_device, dHN_host.data(), state_elements * sizeof(__half), cudaMemcpyHostToDevice),
              "memcpy dHN");
    CheckCuda(cudaMemcpy(dCN_device, dCN_host.data(), state_elements * sizeof(__half), cudaMemcpyHostToDevice),
              "memcpy dCN");

    __half *dx_host = CudaMallocHost<__half>(time_steps * batch_size * input_size, "cudaMallocHost dx_host");
    float *dW_ih_device = CudaMallocDevice<float>(weight_ih_host.size(), "cudaMalloc dW_ih");
    float *dW_hh_device = CudaMallocDevice<float>(weight_hh_host.size(), "cudaMalloc dW_hh");
    float *db_ih_device = CudaMallocDevice<float>(bias_ih_host.size(), "cudaMalloc db_ih");
    float *db_hh_device = CudaMallocDevice<float>(bias_hh_host.size(), "cudaMalloc db_hh");
    float *dh0_out_device = CudaMallocDevice<float>(state_elements, "cudaMalloc dh0_out");
    float *dc0_out_device = CudaMallocDevice<float>(state_elements, "cudaMalloc dc0_out");

    flstm_StreamingLstmBackward(
        time_steps,
        batch_size,
        input_size,
        hidden_size,
        x_host,
        y_host,
        gate_cache_host,
        dY_host,
        dHN_device,
        dCN_device,
        h0_device,
        c0_device,
        weight_ih_device,
        weight_hh_device,
        dx_host,
        dW_ih_device,
        dW_hh_device,
        db_ih_device,
        db_hh_device,
        dh0_out_device,
        dc0_out_device,
        compute_stream
    );
    CheckCuda(cudaDeviceSynchronize(), "backward synchronize");

    size_t nan_count = 0;
    size_t inf_count = 0;
    float dx_max = 0.0f;
    float dx_mean_abs = 0.0f;
    for (size_t i = 0; i < x_elements; ++i) {
        const float val = __half2float(dx_host[i]);
        if (std::isnan(val)) {
            ++nan_count;
            continue;
        }
        if (std::isinf(val)) {
            ++inf_count;
            continue;
        }
        dx_max = std::max(dx_max, std::fabs(val));
        dx_mean_abs += std::fabs(val);
    }
    const float denom = static_cast<float>(x_elements - nan_count - inf_count);
    const float dx_mean = denom > 0 ? dx_mean_abs / denom : 0.0f;

    std::printf("Streaming LSTM forward/backward executed.\n");
    std::printf("Max |dx| = %.6f, mean |dx| = %.6f\n", dx_max, dx_mean);
    std::printf("NaN count = %zu, Inf count = %zu\n", nan_count, inf_count);

    cudaFreeHost(dx_host);
    cudaFreeHost(dY_host);
    cudaFreeHost(y_host);
    cudaFreeHost(gate_cache_host);
    cudaFreeHost(x_host);
    cudaFree(dW_ih_device);
    cudaFree(dW_hh_device);
    cudaFree(db_ih_device);
    cudaFree(db_hh_device);
    cudaFree(dh0_out_device);
    cudaFree(dc0_out_device);
    cudaFree(dCN_device);
    cudaFree(dHN_device);
    cudaFree(bias_hh_device);
    cudaFree(bias_ih_device);
    cudaFree(weight_hh_device);
    cudaFree(weight_ih_device);
    cudaFree(c0_device);
    cudaFree(h0_device);

    cudaStreamDestroy(d2h_stream);
    cudaStreamDestroy(h2d_stream);
    cudaStreamDestroy(compute_stream);

    return 0;
}
