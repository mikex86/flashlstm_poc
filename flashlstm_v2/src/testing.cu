#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace flstm {
namespace testing {
void LaunchConvertInputToZCacheKernel(
    const __half *x_src,
    float *z_cache_col,
    size_t time_offset,
    size_t chunk_steps,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,
    cudaStream_t stream
);
void LaunchScaleAndPackColumnsKernel(
    const float *z_cols_float,
    __half *z_cols_half,
    float *column_scale,
    size_t z_rows,
    size_t batch_size,
    cudaStream_t stream
);
void LaunchForwardPointwiseKernel(
    const float *gate_col,
    const float *bias,
    const float *c_prev,
    const float *h_prev,
    float *h_next,
    float *c_next,
    __half *y_half_out,
    __half *gate_cache_step,
    __half *h_cache,
    __half *c_cache,
    float *z_cache_col,
    const float *column_scale,
    float *checkpoint_dst_h,
    float *checkpoint_dst_c,
    size_t z_rows,
    size_t input_size,
    int has_next_column,
    size_t next_column_offset,
    size_t cache_index,
    size_t batch_size,
    size_t hidden_size,
    cudaStream_t stream
);
void LaunchScalePackPointwiseKernel(
    const float *gate_col,
    const float *bias,
    const float *c_prev,
    const float *h_prev,
    float *h_next,
    float *c_next,
    __half *y_half_out,
    __half *gate_cache_step,
    __half *h_cache,
    __half *c_cache,
    float *z_cache_col,
    const float *column_scale_cur,
    float *column_scale_next,
    const float *next_z_cols_float,
    __half *next_z_cols_half,
    float *checkpoint_dst_h,
    float *checkpoint_dst_c,
    size_t z_rows,
    size_t batch_size,
    size_t hidden_size,
    size_t input_size,
    size_t cache_index,
    int scale_next,
    int has_next_column,
    size_t next_column_offset,
    cudaStream_t stream
);
void LaunchRecomputePointwiseKernel(
    const float *gate_col,
    const float *bias,
    const float *c_prev,
    float *h_next,
    float *c_next,
    float *gate_out,
    float *c_prev_store,
    float *c_store,
    const float *column_scale,
    size_t batch_size,
    size_t hidden_size,
    cudaStream_t stream
);
void LaunchBackwardPointwiseKernel(
    const __half *dY_row,
    const float *dh_next_row,
    const float *dc_next_row,
    const float *gate_cache_row,
    const float *c_prev_row,
    const float *c_row,
    __half *dG_half_col_step,
    float *dc_prev_row,
    size_t batch_size,
    size_t hidden_size,
    cudaStream_t stream
);
} // namespace testing
} // namespace flstm

namespace {

void CheckCuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

template<typename F>
float BenchmarkKernel(F &&launch, cudaStream_t stream, int warmup = 5, int iters = 100) {
    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate start");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CheckCuda(cudaStreamSynchronize(stream), "warmup sync");
    CheckCuda(cudaEventRecord(start, stream), "event record start");
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    CheckCuda(cudaEventRecord(stop, stream), "event record stop");
    CheckCuda(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    CheckCuda(cudaEventDestroy(start), "destroy start");
    CheckCuda(cudaEventDestroy(stop), "destroy stop");
    return ms / static_cast<float>(iters);
}

template<typename T>
T *DeviceAlloc(size_t elements) {
    T *ptr = nullptr;
    if (elements > 0) {
        CheckCuda(cudaMalloc(&ptr, elements * sizeof(T)), "cudaMalloc");
    }
    return ptr;
}

void FillRandom(std::vector<float> &values) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : values) {
        v = dist(rng);
    }
}

} // namespace

int main() {
    constexpr size_t batch_size = 32;
    constexpr size_t hidden_size = 1024;
    constexpr size_t input_size = 1024;
    constexpr size_t chunk_steps = 32;
    constexpr size_t time_offset = 0;
    const size_t gate_dim = 4 * hidden_size;
    const size_t z_rows = input_size + hidden_size;
    const size_t bh_hidden = batch_size * hidden_size;
    const size_t batch_input = batch_size * input_size;
    const size_t chunk_elems = chunk_steps * batch_input;
    const size_t column_count = chunk_steps * batch_size;

    cudaStream_t stream;
    CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    std::vector<__half> x_host(chunk_elems);
    {
        std::vector<float> tmp(chunk_elems);
        FillRandom(tmp);
        for (size_t i = 0; i < chunk_elems; ++i) {
            x_host[i] = __float2half(tmp[i]);
        }
    }

    const size_t z_cache_elems = column_count * z_rows;
    const size_t z_cols_elems = batch_size * z_rows;
    const size_t gate_col_elems = gate_dim * batch_size;
    std::vector<float> z_cols_host(z_cols_elems);
    std::vector<float> gate_col_host(gate_col_elems);
    FillRandom(z_cols_host);
    FillRandom(gate_col_host);

    std::vector<float> bias_host(gate_dim);
    FillRandom(bias_host);

    std::vector<float> state_host(batch_size * hidden_size);
    FillRandom(state_host);

    __half *x_dev = DeviceAlloc<__half>(chunk_elems);
    float *z_cache_dev = DeviceAlloc<float>(z_cache_elems);
    float *z_cols_dev = DeviceAlloc<float>(batch_size * z_rows);
    __half *z_half_dev = DeviceAlloc<__half>(batch_size * z_rows);
    float *scale_dev = DeviceAlloc<float>(batch_size);
    float *scale_next_dev = DeviceAlloc<float>(batch_size);
    float *bias_dev = DeviceAlloc<float>(gate_dim);
    float *gate_col_dev = DeviceAlloc<float>(gate_dim * batch_size);
    float *c_prev_dev = DeviceAlloc<float>(batch_size * hidden_size);
    float *h_prev_dev = DeviceAlloc<float>(batch_size * hidden_size);
    float *h_next_dev = DeviceAlloc<float>(batch_size * hidden_size);
    float *c_next_dev = DeviceAlloc<float>(batch_size * hidden_size);
    __half *y_half_dev = DeviceAlloc<__half>(batch_size * hidden_size);
    __half *gate_cache_dev = DeviceAlloc<__half>(batch_size * gate_dim);
    __half *h_cache_dev = DeviceAlloc<__half>(batch_size * hidden_size);
    __half *c_cache_dev = DeviceAlloc<__half>(batch_size * hidden_size);
    float *checkpoint_h_dev = DeviceAlloc<float>(batch_size * hidden_size);
    float *checkpoint_c_dev = DeviceAlloc<float>(batch_size * hidden_size);
    float *h_next_ref_dev = DeviceAlloc<float>(batch_size * hidden_size);
    float *c_next_ref_dev = DeviceAlloc<float>(batch_size * hidden_size);

    CheckCuda(cudaMemcpyAsync(x_dev, x_host.data(), chunk_elems * sizeof(__half), cudaMemcpyHostToDevice, stream),
              "copy x_dev");
    CheckCuda(cudaMemcpyAsync(z_cols_dev, z_cols_host.data(), z_cols_elems * sizeof(float), cudaMemcpyHostToDevice, stream),
              "copy z cols");
    CheckCuda(cudaMemcpyAsync(gate_col_dev, gate_col_host.data(), gate_col_elems * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "copy gate_col");
    CheckCuda(cudaMemcpyAsync(bias_dev, bias_host.data(), gate_dim * sizeof(float), cudaMemcpyHostToDevice, stream),
              "copy bias");
    CheckCuda(cudaMemcpyAsync(c_prev_dev, state_host.data(), state_host.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "copy c_prev");
    CheckCuda(cudaMemcpyAsync(h_prev_dev, state_host.data(), state_host.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "copy h_prev");
    CheckCuda(cudaStreamSynchronize(stream), "initial sync");

    auto convert_ms = BenchmarkKernel(
        [&]() {
            flstm::testing::LaunchConvertInputToZCacheKernel(
                x_dev,
                z_cache_dev,
                time_offset,
                chunk_steps,
                batch_size,
                input_size,
                hidden_size,
                stream
            );
        },
        stream);

    flstm::testing::LaunchScaleAndPackColumnsKernel(
        z_cols_dev,
        z_half_dev,
        scale_dev,
        z_rows,
        batch_size,
        stream
    );

    flstm::testing::LaunchForwardPointwiseKernel(
        gate_col_dev,
        bias_dev,
        c_prev_dev,
        h_prev_dev,
        h_next_ref_dev,
        c_next_ref_dev,
        y_half_dev,
        gate_cache_dev,
        h_cache_dev,
        c_cache_dev,
        z_cache_dev,
        scale_dev,
        checkpoint_h_dev,
        checkpoint_c_dev,
        z_rows,
        input_size,
        0,
        0,
        0,
        batch_size,
        hidden_size,
        stream
    );

    CheckCuda(cudaStreamSynchronize(stream), "sync ref");

    std::vector<float> h_ref_host(batch_size * hidden_size);
    std::vector<float> h_fused_host(batch_size * hidden_size);
    CheckCuda(cudaMemcpy(h_ref_host.data(), h_next_ref_dev, h_ref_host.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "copy h_ref");

    flstm::testing::LaunchScalePackPointwiseKernel(
        gate_col_dev,
        bias_dev,
        c_prev_dev,
        h_prev_dev,
        h_next_dev,
        c_next_dev,
        y_half_dev,
        gate_cache_dev,
        h_cache_dev,
        c_cache_dev,
        z_cache_dev,
        scale_dev,
        scale_next_dev,
        nullptr,
        nullptr,
        checkpoint_h_dev,
        checkpoint_c_dev,
                z_rows,
                batch_size,
                hidden_size,
                input_size,
                1,
                0,
                0,
                0,
                stream
            );
    CheckCuda(cudaStreamSynchronize(stream), "sync fused");
    CheckCuda(cudaMemcpy(h_fused_host.data(), h_next_dev, h_fused_host.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "copy h_fused");
    float max_diff = 0.0f;
    for (size_t i = 0; i < h_ref_host.size(); ++i) {
        max_diff = std::max(max_diff, fabsf(h_ref_host[i] - h_fused_host[i]));
    }

    auto scale_ms = BenchmarkKernel(
        [&]() {
            flstm::testing::LaunchScaleAndPackColumnsKernel(
                z_cols_dev,
                z_half_dev,
                scale_dev,
                z_rows,
                batch_size,
                stream
            );
        },
        stream);

    auto pointwise_ms = BenchmarkKernel(
        [&]() {
            flstm::testing::LaunchForwardPointwiseKernel(
                gate_col_dev,
                bias_dev,
                c_prev_dev,
                h_prev_dev,
                h_next_dev,
                c_next_dev,
                y_half_dev,
                gate_cache_dev,
                h_cache_dev,
                c_cache_dev,
                z_cache_dev,
                scale_dev,
                checkpoint_h_dev,
                checkpoint_c_dev,
                z_rows,
                input_size,
                0,
                0,
                0,
                batch_size,
                hidden_size,
                stream
            );
        },
        stream);

    auto fused_ms = BenchmarkKernel(
        [&]() {
            flstm::testing::LaunchScalePackPointwiseKernel(
                gate_col_dev,
                bias_dev,
                c_prev_dev,
                h_prev_dev,
                h_next_dev,
                c_next_dev,
                y_half_dev,
                gate_cache_dev,
                h_cache_dev,
                c_cache_dev,
                z_cache_dev,
                scale_dev,
                scale_next_dev,
                z_cols_dev,
                z_half_dev,
                checkpoint_h_dev,
                checkpoint_c_dev,
                z_rows,
                batch_size,
                hidden_size,
                input_size,
                1,
                1,
                0,
                0,
                stream
            );
        },
        stream);

    std::cout << "ConvertInputToZCacheKernel: " << convert_ms << " ms\n";
    std::cout << "ScaleAndPackColumnsKernel: " << scale_ms << " ms\n";
    std::cout << "ForwardPointwiseKernel: " << pointwise_ms << " ms\n";
    std::cout << "ScalePackPointwiseKernel: " << fused_ms << " ms\n";
    std::cout << "Max fused/reference delta: " << max_diff << "\n";

    std::vector<__half> dY_host(bh_hidden);
    {
        std::vector<float> tmp(bh_hidden);
        FillRandom(tmp);
        for (size_t i = 0; i < bh_hidden; ++i) {
            dY_host[i] = __float2half(tmp[i]);
        }
    }
    std::vector<float> dh_next_host(bh_hidden);
    std::vector<float> dc_next_host(bh_hidden);
    FillRandom(dh_next_host);
    FillRandom(dc_next_host);
    std::vector<float> column_scale_host(batch_size, 1.0f);

    __half *dY_dev_bwd = DeviceAlloc<__half>(bh_hidden);
    float *dh_next_dev_bwd = DeviceAlloc<float>(bh_hidden);
    float *dc_next_dev_bwd = DeviceAlloc<float>(bh_hidden);
    float *c_prev_store_dev = DeviceAlloc<float>(bh_hidden);
    float *c_store_dev = DeviceAlloc<float>(bh_hidden);
    float *column_scale_dev_bwd = DeviceAlloc<float>(batch_size);
    float *gate_out_dev = DeviceAlloc<float>(gate_dim * batch_size);
    __half *dG_half_dev = DeviceAlloc<__half>(gate_dim * batch_size);
    float *dc_prev_dev = DeviceAlloc<float>(bh_hidden);

    CheckCuda(cudaMemcpyAsync(
                  dY_dev_bwd,
                  dY_host.data(),
                  dY_host.size() * sizeof(__half),
                  cudaMemcpyHostToDevice,
                  stream),
              "copy dY");
    CheckCuda(cudaMemcpyAsync(
                  dh_next_dev_bwd,
                  dh_next_host.data(),
                  dh_next_host.size() * sizeof(float),
                  cudaMemcpyHostToDevice,
                  stream),
              "copy dh_next");
    CheckCuda(cudaMemcpyAsync(
                  dc_next_dev_bwd,
                  dc_next_host.data(),
                  dc_next_host.size() * sizeof(float),
                  cudaMemcpyHostToDevice,
                  stream),
              "copy dc_next");
    CheckCuda(cudaMemcpyAsync(
                  column_scale_dev_bwd,
                  column_scale_host.data(),
                  column_scale_host.size() * sizeof(float),
                  cudaMemcpyHostToDevice,
                  stream),
              "copy column_scale");
    CheckCuda(cudaStreamSynchronize(stream), "sync bwd init");

    auto recompute_ms = BenchmarkKernel(
        [&]() {
            flstm::testing::LaunchRecomputePointwiseKernel(
                gate_col_dev,
                bias_dev,
                c_prev_dev,
                h_next_dev,
                c_next_dev,
                gate_out_dev,
                c_prev_store_dev,
                c_store_dev,
                column_scale_dev_bwd,
                batch_size,
                hidden_size,
                stream
            );
        },
        stream);

    auto backward_ms = BenchmarkKernel(
        [&]() {
            flstm::testing::LaunchBackwardPointwiseKernel(
                dY_dev_bwd,
                dh_next_dev_bwd,
                dc_next_dev_bwd,
                gate_out_dev,
                c_prev_store_dev,
                c_store_dev,
                dG_half_dev,
                dc_prev_dev,
                batch_size,
                hidden_size,
                stream
            );
        },
        stream);

    std::cout << "RecomputePointwiseKernel: " << recompute_ms << " ms\n";
    std::cout << "BackwardPointwiseKernel: " << backward_ms << " ms\n";

    cudaFree(x_dev);
    cudaFree(z_cache_dev);
    cudaFree(z_cols_dev);
    cudaFree(z_half_dev);
    cudaFree(scale_dev);
    cudaFree(scale_next_dev);
    cudaFree(bias_dev);
    cudaFree(gate_col_dev);
    cudaFree(c_prev_dev);
    cudaFree(h_prev_dev);
    cudaFree(h_next_dev);
    cudaFree(c_next_dev);
    cudaFree(y_half_dev);
    cudaFree(gate_cache_dev);
    cudaFree(h_cache_dev);
    cudaFree(c_cache_dev);
    cudaFree(checkpoint_h_dev);
    cudaFree(checkpoint_c_dev);
    cudaFree(dY_dev_bwd);
    cudaFree(dh_next_dev_bwd);
    cudaFree(dc_next_dev_bwd);
    cudaFree(c_prev_store_dev);
    cudaFree(c_store_dev);
    cudaFree(column_scale_dev_bwd);
    cudaFree(gate_out_dev);
    cudaFree(dG_half_dev);
    cudaFree(dc_prev_dev);
    cudaFree(h_next_ref_dev);
    cudaFree(c_next_ref_dev);
    CheckCuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
