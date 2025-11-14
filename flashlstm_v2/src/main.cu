#include "lstm.hpp"
#include "cudnn_reference.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <optional>

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


    struct FallbackHostAllocation {
        void *base{nullptr};
        size_t bytes{0};
        std::vector<void *> registered_ptrs;
    };

    std::mutex &HostAllocMutex() {
        static std::mutex mutex;
        return mutex;
    }

    std::vector<FallbackHostAllocation> &HostAllocTable() {
        static std::vector<FallbackHostAllocation> table;
        return table;
    }

    template<typename T>
    T *CudaMallocHost(const size_t count, const char *what) {
        if (count == 0) {
            return nullptr;
        }
        const size_t bytes = count * sizeof(T);
        constexpr size_t kPageAlignment = 4096;
        void *raw = nullptr;
        if (posix_memalign(&raw, kPageAlignment, bytes) != 0 || raw == nullptr) {
            std::fprintf(stderr, "%s posix_memalign failed\n", what);
            std::abort();
        }
        auto *base = static_cast<uint8_t *>(raw);
        constexpr size_t chunk_bytes = (1ull << 30); // 1 GiB per registration
        constexpr size_t aligned_chunk = (chunk_bytes / kPageAlignment) * kPageAlignment;
        std::vector<void *> registered_ptrs;
        registered_ptrs.reserve((bytes + aligned_chunk - 1) / aligned_chunk);
        size_t offset = 0;
        while (offset < bytes) {
            const size_t remaining = bytes - offset;
            const size_t this_size = std::min(aligned_chunk, remaining);
            const cudaError_t err = cudaHostRegister(base + offset, this_size, cudaHostRegisterPortable);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "%s cudaHostRegister failed: %s\n", what, cudaGetErrorString(err));
                for (void *p: registered_ptrs) {
                    cudaHostUnregister(p);
                }
                std::free(raw);
                std::abort();
            }
            registered_ptrs.push_back(base + offset);
            offset += this_size;
        }
        {
            std::lock_guard lock(HostAllocMutex());
            FallbackHostAllocation record{};
            record.base = base;
            record.bytes = bytes;
            record.registered_ptrs = std::move(registered_ptrs);
            HostAllocTable().push_back(std::move(record));
        }
        return reinterpret_cast<T *>(base);
    }

    void CudaFreeHostWrapper(void *ptr) {
        if (ptr == nullptr) {
            return;
        }
        FallbackHostAllocation entry{};
        {
            std::lock_guard lock(HostAllocMutex());
            auto &table = HostAllocTable();
            auto it = std::find_if(
                table.begin(),
                table.end(),
                [&](const FallbackHostAllocation &alloc) { return alloc.base == ptr; }
            );
            if (it == table.end()) {
                std::fprintf(stderr, "CudaFreeHostWrapper: pointer %p not tracked; freeing directly\n", ptr);
                std::free(ptr);
                return;
            }
            entry = std::move(*it);
            table.erase(it);
        }
        for (void *reg: entry.registered_ptrs) {
            cudaHostUnregister(reg);
        }
        std::free(entry.base);
    }

    __device__ __forceinline__ uint64_t SplitMix64(uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    __device__ __forceinline__ float UniformFloat01(uint64_t bits) {
        constexpr double kInv = 1.0 / static_cast<double>(1ull << 53);
        const double val = static_cast<double>(bits >> 11) * kInv;
        return static_cast<float>(val);
    }

    __global__ void GenerateNormalKernel(
        float *dst,
        size_t count,
        float mean,
        float stddev,
        uint64_t seed,
        uint64_t index_offset
    ) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;
        constexpr float kTwoPi = 6.283185307179586476925286766559f;
        for (size_t i = tid; i < count; i += stride) {
            const uint64_t global_idx = index_offset + i;
            const uint64_t bits0 = SplitMix64(seed + global_idx * 2ull);
            const uint64_t bits1 = SplitMix64(seed + global_idx * 2ull + 1ull);
            const float u1 = fmaxf(UniformFloat01(bits0), 1e-7f);
            const float u2 = UniformFloat01(bits1);
            const float magnitude = sqrtf(-2.0f * logf(u1));
            const float z = magnitude * cosf(kTwoPi * u2);
            dst[i] = mean + stddev * z;
        }
    }

    __global__ void GenerateUniformKernel(
        float *dst,
        size_t count,
        float low,
        float high,
        uint64_t seed,
        uint64_t index_offset
    ) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;
        const float range = high - low;
        for (size_t i = tid; i < count; i += stride) {
            const uint64_t global_idx = index_offset + i;
            const uint64_t bits = SplitMix64(seed + global_idx);
            const float u = UniformFloat01(bits);
            dst[i] = low + range * u;
        }
    }

    __global__ void FloatToHalfKernel(const float *src, __half *dst, size_t count) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        dst[idx] = __float2half(src[idx]);
    }

    inline int BlocksForCount(size_t count, int threads = 256) {
        if (count == 0) {
            return 1;
        }
        const size_t raw = (count + threads - 1) / threads;
        return static_cast<int>(std::min<size_t>(raw, 65535));
    }

    void FillNormalHostBuffers(
        float *device_float_buffer,
        __half *device_half_buffer,
        size_t chunk_capacity,
        float *host_float_out,
        __half *host_half_out,
        size_t count,
        float mean,
        float stddev,
        uint64_t seed
    ) {
        if (count == 0) {
            return;
        }
        const int threads = 256;
        uint64_t offset = 0;
        while (offset < count) {
            const size_t chunk = std::min(chunk_capacity, count - offset);
            const int blocks = BlocksForCount(chunk, threads);
            GenerateNormalKernel<<<blocks, threads>>>(
                device_float_buffer,
                chunk,
                mean,
                stddev,
                seed,
                offset
            );
            CheckCuda(cudaGetLastError(), "GenerateNormalKernel");
            if (host_float_out != nullptr) {
                CheckCuda(cudaMemcpy(
                              host_float_out + offset,
                              device_float_buffer,
                              chunk * sizeof(float),
                              cudaMemcpyDeviceToHost),
                          "copy normal floats -> host");
            }
            if (host_half_out != nullptr) {
                const int convert_blocks = BlocksForCount(chunk, threads);
                FloatToHalfKernel<<<convert_blocks, threads>>>(device_float_buffer, device_half_buffer, chunk);
                CheckCuda(cudaGetLastError(), "FloatToHalfKernel (normal init)");
                CheckCuda(cudaMemcpy(
                              host_half_out + offset,
                              device_half_buffer,
                              chunk * sizeof(__half),
                              cudaMemcpyDeviceToHost),
                          "copy normal halves -> host");
            }
            offset += chunk;
        }
    }

    void FillUniformHostBuffer(
        float *device_float_buffer,
        const size_t chunk_capacity,
        float *host_out,
        const size_t count,
        const float low,
        const float high,
        const uint64_t seed
    ) {
        if (count == 0) {
            return;
        }
        constexpr int threads = 256;
        uint64_t offset = 0;
        while (offset < count) {
            const size_t chunk = std::min(chunk_capacity, count - offset);
            const int blocks = BlocksForCount(chunk, threads);
            GenerateUniformKernel<<<blocks, threads>>>(
                device_float_buffer,
                chunk,
                low,
                high,
                seed,
                offset
            );
            CheckCuda(cudaGetLastError(), "GenerateUniformKernel");
            CheckCuda(cudaMemcpy(
                          host_out + offset,
                          device_float_buffer,
                          chunk * sizeof(float),
                          cudaMemcpyDeviceToHost),
                      "copy uniform floats -> host");
            offset += chunk;
        }
    }
} // namespace

int main() {
    constexpr size_t time_steps = 8192;
    constexpr size_t batch_size = 1024;
    constexpr size_t input_size = 1024;
    constexpr size_t hidden_size = 1024;
    constexpr size_t recompute_interval = 4;
    constexpr size_t gate_dim = 4 * hidden_size;

    constexpr size_t x_elements = time_steps * batch_size * input_size;
    constexpr size_t y_elements = time_steps * batch_size * hidden_size;
    constexpr size_t checkpoint_steps = (time_steps + recompute_interval - 1) / recompute_interval;
    constexpr size_t gate_elements = checkpoint_steps * batch_size * hidden_size * 2;
    constexpr size_t state_elements = batch_size * hidden_size;

    auto *x_host = CudaMallocHost<__half>(x_elements, "cudaMallocHost x_host");
    auto *y_host = CudaMallocHost<__half>(y_elements, "cudaMallocHost y_host");
    std::vector<float> x_host_float(x_elements);
    constexpr float kInputStd = 0.01f;
    constexpr size_t kNormalChunkElements = 1 << 22;
    float *rng_chunk_float = CudaMallocDevice<float>(kNormalChunkElements, "cudaMalloc rng_chunk_float");
    __half *rng_chunk_half = CudaMallocDevice<__half>(kNormalChunkElements, "cudaMalloc rng_chunk_half");
    constexpr uint64_t kBaseSeed = 1337;
    FillNormalHostBuffers(
        rng_chunk_float,
        rng_chunk_half,
        kNormalChunkElements,
        x_host_float.data(),
        x_host,
        x_elements,
        0.0f,
        kInputStd,
        kBaseSeed
    );

    std::vector<float> weight_ih_host(gate_dim * input_size);
    std::vector<float> weight_hh_host(gate_dim * hidden_size);
    std::vector<float> bias_ih_host(gate_dim, 0.0f);
    std::vector<float> bias_hh_host(gate_dim, 0.0f);
    const float weight_limit = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    FillUniformHostBuffer(
        rng_chunk_float,
        kNormalChunkElements,
        weight_ih_host.data(),
        weight_ih_host.size(),
        -weight_limit,
        weight_limit,
        kBaseSeed + 4
    );
    FillUniformHostBuffer(
        rng_chunk_float,
        kNormalChunkElements,
        weight_hh_host.data(),
        weight_hh_host.size(),
        -weight_limit,
        weight_limit,
        kBaseSeed + 5
    );

    std::vector h0_host(state_elements, __float2half(0.0f));
    std::vector c0_host(state_elements, __float2half(0.0f));
    std::vector h0_host_float(state_elements, 0.0f);
    std::vector c0_host_float(state_elements, 0.0f);

    __half *h0_device = CudaMallocDevice<__half>(state_elements, "cudaMalloc h0_device");
    __half *c0_device = CudaMallocDevice<__half>(state_elements, "cudaMalloc c0_device");
    float *weight_ih_device = CudaMallocDevice<float>(weight_ih_host.size(), "cudaMalloc weight_ih");
    float *weight_hh_device = CudaMallocDevice<float>(weight_hh_host.size(), "cudaMalloc weight_hh");
    float *bias_ih_device = CudaMallocDevice<float>(bias_ih_host.size(), "cudaMalloc bias_ih");
    float *bias_hh_device = CudaMallocDevice<float>(bias_hh_host.size(), "cudaMalloc bias_hh");
    __half *hy_device_out = CudaMallocDevice<__half>(state_elements, "cudaMalloc hy_device");
    __half *cy_device_out = CudaMallocDevice<__half>(state_elements, "cudaMalloc cy_device");

    float *gate_cache_h_host = CudaMallocHost<float>(gate_elements, "cudaMallocHost gate_cache_h");
    float *gate_cache_c_host = CudaMallocHost<float>(gate_elements, "cudaMallocHost gate_cache_c");
    flstm_GateCacheHost gate_cache_host{
        .h_ptr = gate_cache_h_host,
        .c_ptr = gate_cache_c_host,
    };
    flstm_StreamingLstmOptions gate_cache_options{};
    gate_cache_options.h_dtype = FLSTM_GATE_CACHE_FLOAT32;
    gate_cache_options.c_dtype = FLSTM_GATE_CACHE_FLOAT32;

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
        recompute_interval,
        x_host,
        h0_device,
        c0_device,
        weight_ih_device,
        weight_hh_device,
        bias_ih_device,
        bias_hh_device,
        y_host,
        gate_cache_host,
        &gate_cache_options,
        hy_device_out,
        cy_device_out,
        compute_stream,
        h2d_stream,
        d2h_stream
    );
    CheckCuda(cudaDeviceSynchronize(), "forward warmup synchronize");

    flstm_StreamingLstmForward(
        time_steps,
        batch_size,
        input_size,
        hidden_size,
        recompute_interval,
        x_host,
        h0_device,
        c0_device,
        weight_ih_device,
        weight_hh_device,
        bias_ih_device,
        bias_hh_device,
        y_host,
        gate_cache_host,
        &gate_cache_options,
        hy_device_out,
        cy_device_out,
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
    FillNormalHostBuffers(
        rng_chunk_float,
        rng_chunk_half,
        kNormalChunkElements,
        nullptr,
        dY_host,
        y_elements,
        0.0f,
        kGradStd,
        kBaseSeed + 1
    );
    FillNormalHostBuffers(
        rng_chunk_float,
        rng_chunk_half,
        kNormalChunkElements,
        nullptr,
        dHN_host.data(),
        state_elements,
        0.0f,
        kGradStd,
        kBaseSeed + 2
    );
    FillNormalHostBuffers(
        rng_chunk_float,
        rng_chunk_half,
        kNormalChunkElements,
        nullptr,
        dCN_host.data(),
        state_elements,
        0.0f,
        kGradStd,
        kBaseSeed + 3
    );
    cudaFree(rng_chunk_half);
    cudaFree(rng_chunk_float);

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
        recompute_interval,
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
        bias_ih_device,
        bias_hh_device,
        dx_host,
        dW_ih_device,
        dW_hh_device,
        db_ih_device,
        db_hh_device,
        dh0_out_device,
        dc0_out_device,
        compute_stream,
        h2d_stream,
        d2h_stream,
        &gate_cache_options
    );
    CheckCuda(cudaDeviceSynchronize(), "backward warmup synchronize");

    flstm_StreamingLstmBackward(
        time_steps,
        batch_size,
        input_size,
        hidden_size,
        recompute_interval,
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
        bias_ih_device,
        bias_hh_device,
        dx_host,
        dW_ih_device,
        dW_hh_device,
        db_ih_device,
        db_hh_device,
        dh0_out_device,
        dc0_out_device,
        compute_stream,
        h2d_stream,
        d2h_stream,
        &gate_cache_options
    );
    CheckCuda(cudaDeviceSynchronize(), "backward synchronize");

    std::vector<__half> hy_host_final(state_elements);
    std::vector<__half> cy_host_final(state_elements);
    CheckCuda(cudaMemcpy(
                  hy_host_final.data(),
                  hy_device_out,
                  state_elements * sizeof(__half),
                  cudaMemcpyDeviceToHost),
              "memcpy hy_host_final");
    CheckCuda(cudaMemcpy(
                  cy_host_final.data(),
                  cy_device_out,
                  state_elements * sizeof(__half),
                  cudaMemcpyDeviceToHost),
              "memcpy cy_host_final");

#ifdef FLASHLSTM_ENABLE_CUDNN
    CudnnForwardComparisonResult cudnn_forward_result = RunCudnnForwardComparison(
        time_steps,
        batch_size,
        input_size,
        hidden_size,
        x_host_float.data(),
        weight_ih_host.data(),
        weight_hh_host.data(),
        bias_ih_host.data(),
        bias_hh_host.data(),
        h0_host_float.data(),
        c0_host_float.data(),
        y_host,
        gate_cache_h_host,
        gate_cache_c_host,
        hy_host_final.data(),
        cy_host_final.data()
    );

    const float cudnn_forward_tol = 5e-2f;
    if (cudnn_forward_result.max_y_delta > cudnn_forward_tol ||
        cudnn_forward_result.max_h_delta > cudnn_forward_tol ||
        cudnn_forward_result.max_c_delta > cudnn_forward_tol) {
        std::fprintf(stderr,
                     "cuDNN comparison failed: max|Δy|=%g, max|Δh|=%g, max|Δc|=%g (tol=%g)\n",
                     cudnn_forward_result.max_y_delta,
                     cudnn_forward_result.max_h_delta,
                     cudnn_forward_result.max_c_delta,
                     cudnn_forward_tol);
        return EXIT_FAILURE;
    }

    std::printf("cuDNN reference: max|Δy|=%g, max|Δh|=%g, max|Δc|=%g\n",
                cudnn_forward_result.max_y_delta,
                cudnn_forward_result.max_h_delta,
                cudnn_forward_result.max_c_delta);

    CudnnBackwardComparisonResult cudnn_backward_result = RunCudnnBackwardComparison(
        time_steps,
        batch_size,
        input_size,
        hidden_size,
        x_host_float.data(),
        weight_ih_host.data(),
        weight_hh_host.data(),
        bias_ih_host.data(),
        bias_hh_host.data(),
        h0_host_float.data(),
        c0_host_float.data(),
        y_host,
        gate_cache_h_host,
        gate_cache_c_host,
        dY_host,
        dHN_host,
        dCN_host,
        dx_host,
        dW_ih_device,
        dW_hh_device,
        db_ih_device,
        db_hh_device,
        dh0_out_device,
        dc0_out_device
    );

    const float cudnn_backward_tol = 5e-2f;
    if (cudnn_backward_result.max_dx_delta > cudnn_backward_tol ||
        cudnn_backward_result.max_dh0_delta > cudnn_backward_tol ||
        cudnn_backward_result.max_dc0_delta > cudnn_backward_tol ||
        cudnn_backward_result.max_dW_ih_delta > cudnn_backward_tol ||
        cudnn_backward_result.max_dW_hh_delta > cudnn_backward_tol ||
        cudnn_backward_result.max_db_ih_delta > cudnn_backward_tol ||
        cudnn_backward_result.max_db_hh_delta > cudnn_backward_tol) {
        std::fprintf(stderr,
                     "cuDNN backward comparison failed: max|Δdx|=%g, max|Δdh0|=%g, max|Δdc0|=%g, "
                     "max|ΔdWih|=%g, max|ΔdWhh|=%g, max|Δdbih|=%g, max|Δdbhh|=%g (tol=%g)\n",
                     cudnn_backward_result.max_dx_delta,
                     cudnn_backward_result.max_dh0_delta,
                     cudnn_backward_result.max_dc0_delta,
                     cudnn_backward_result.max_dW_ih_delta,
                     cudnn_backward_result.max_dW_hh_delta,
                     cudnn_backward_result.max_db_ih_delta,
                     cudnn_backward_result.max_db_hh_delta,
                     cudnn_backward_tol);
        return EXIT_FAILURE;
    }

    std::printf("cuDNN backward reference: max|Δdx|=%g, max|Δdh0|=%g, max|Δdc0|=%g, max|ΔdWih|=%g, "
                "max|ΔdWhh|=%g, max|Δdbih|=%g, max|Δdbhh|=%g\n",
                cudnn_backward_result.max_dx_delta,
                cudnn_backward_result.max_dh0_delta,
                cudnn_backward_result.max_dc0_delta,
                cudnn_backward_result.max_dW_ih_delta,
                cudnn_backward_result.max_dW_hh_delta,
                cudnn_backward_result.max_db_ih_delta,
                cudnn_backward_result.max_db_hh_delta);
#else
#endif

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

    CudaFreeHostWrapper(dx_host);
    CudaFreeHostWrapper(dY_host);
    CudaFreeHostWrapper(y_host);
    CudaFreeHostWrapper(gate_cache_h_host);
    CudaFreeHostWrapper(gate_cache_c_host);
    CudaFreeHostWrapper(x_host);
    cudaFree(dW_ih_device);
    cudaFree(dW_hh_device);
    cudaFree(db_ih_device);
    cudaFree(db_hh_device);
    cudaFree(hy_device_out);
    cudaFree(cy_device_out);
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

    cudaDeviceReset();

    return 0;
}
