#include "flashlstm/lstm_api.h"
#include "error_checks.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstddef>
#include <limits>
#include <vector>

namespace {
template <typename T>
struct DeviceBuffer {
    T *data;

    DeviceBuffer() : data(nullptr) {}
    ~DeviceBuffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }

    cudaError_t allocate(std::size_t elements) {
        if (data != nullptr) {
            cudaFree(data);
            data = nullptr;
        }
        if (elements == 0) {
            return cudaSuccess;
        }
        return cudaMalloc(&data, elements * sizeof(T));
    }
};

int report_status(const char *name, int status) {
    if (status == 0) {
        return 0;
    }
    const auto cuda_status = static_cast<cudaError_t>(status);
    std::fprintf(stderr,
                 "%s failed with status %d (%s)\n",
                 name,
                 status,
                 cudaGetErrorString(cuda_status));
    return status;
}
}  // namespace

int main() {
    constexpr std::size_t seq_len = 2048;
    constexpr std::size_t batch = 32;
    constexpr std::size_t input_size = 1024;
    constexpr std::size_t hidden_size = 1024;
    constexpr auto compute_precision = LSTM_COMPUTE_PRECISION_FP16_ACC16;

    const std::vector<int> chunk_candidates = {0, 8, 16, 32, 64, 128, 256, 512, 1024};
    const int warmup_iters = 1;
    const int timing_iters = 5;

    const std::size_t x_elems = seq_len * batch * input_size;
    const std::size_t w_ih_elems = 4 * hidden_size * input_size;
    const std::size_t w_hh_elems = 4 * hidden_size * hidden_size;
    const std::size_t bias_elems = 4 * hidden_size;
    const std::size_t state_elems = batch * hidden_size;
    const std::size_t output_elems = seq_len * batch * hidden_size;

    const auto make_zero_vector = [](std::size_t count) {
        return std::vector<float>(count, 0.0f);
    };

    auto x = make_zero_vector(x_elems);
    auto w_ih = make_zero_vector(w_ih_elems);
    auto w_hh = make_zero_vector(w_hh_elems);
    auto b_ih = make_zero_vector(bias_elems);
    auto b_hh = make_zero_vector(bias_elems);
    auto h0 = make_zero_vector(state_elems);
    auto c0 = make_zero_vector(state_elems);

    DeviceBuffer<float> d_x;
    DeviceBuffer<float> d_w_ih;
    DeviceBuffer<float> d_w_hh;
    DeviceBuffer<float> d_b_ih;
    DeviceBuffer<float> d_b_hh;
    DeviceBuffer<float> d_h0;
    DeviceBuffer<float> d_c0;
    DeviceBuffer<float> d_output;
    DeviceBuffer<float> d_hn;
    DeviceBuffer<float> d_cn;

    struct AllocationSpec {
        DeviceBuffer<float> *buffer;
        std::size_t elements;
    };

    const AllocationSpec allocations[] = {
        {&d_x, x_elems},
        {&d_w_ih, w_ih_elems},
        {&d_w_hh, w_hh_elems},
        {&d_b_ih, bias_elems},
        {&d_b_hh, bias_elems},
        {&d_h0, state_elems},
        {&d_c0, state_elems},
        {&d_output, output_elems},
        {&d_hn, state_elems},
        {&d_cn, state_elems},
    };

    for (const auto &spec : allocations) {
        FLASHLSTM_CHECK_CUDA(spec.buffer->allocate(spec.elements));
    }

    struct HostToDeviceCopy {
        DeviceBuffer<float> *device;
        const std::vector<float> *host;
        std::size_t elements;
    };

    const HostToDeviceCopy copies[] = {
        {&d_x, &x, x_elems},
        {&d_w_ih, &w_ih, w_ih_elems},
        {&d_w_hh, &w_hh, w_hh_elems},
        {&d_b_ih, &b_ih, bias_elems},
        {&d_b_hh, &b_hh, bias_elems},
        {&d_h0, &h0, state_elems},
        {&d_c0, &c0, state_elems},
    };

    for (const auto &copy : copies) {
        FLASHLSTM_CHECK_CUDA(cudaMemcpy(copy.device->data,
                                        copy.host->data(),
                                        copy.elements * sizeof(float),
                                        cudaMemcpyHostToDevice));
    }

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    FLASHLSTM_CHECK_CUDA(cudaEventCreate(&start_event));
    FLASHLSTM_CHECK_CUDA(cudaEventCreate(&stop_event));

    float best_time_ms = std::numeric_limits<float>::max();
    int best_chunk = -1;

    for (int chunk_size : chunk_candidates) {
        std::printf("Testing chunk size %d...\n", chunk_size);
        lstm_buffers buffers{};
        int status = lstm_create_buffers(compute_precision,
                                         seq_len,
                                         batch,
                                         input_size,
                                         hidden_size,
                                         chunk_size,
                                         &buffers);
        if (report_status("lstm_create_buffers", status) != 0) {
            return status;
        }

        status = lstm_pack_weights(compute_precision,
                                   d_w_ih.data,
                                   d_w_hh.data,
                                   input_size,
                                   hidden_size,
                                   &buffers);
        if (report_status("lstm_pack_weights", status) != 0) {
            lstm_destroy_buffers(&buffers);
            return status;
        }

        const auto run_forward = [&]() -> int {
            return lstm_forward(d_x.data,
                                d_b_ih.data,
                                d_b_hh.data,
                                d_h0.data,
                                d_c0.data,
                                d_output.data,
                                d_hn.data,
                                d_cn.data,
                                seq_len,
                                batch,
                                input_size,
                                hidden_size,
                                &buffers,
                                compute_precision);
        };

        for (int i = 0; i < warmup_iters; ++i) {
            status = run_forward();
            if (report_status("lstm_forward (warmup)", status) != 0) {
                lstm_destroy_buffers(&buffers);
                return status;
            }
        }
        FLASHLSTM_CHECK_CUDA(cudaDeviceSynchronize());

        float total_ms = 0.0f;
        for (int iter = 0; iter < timing_iters; ++iter) {
            FLASHLSTM_CHECK_CUDA(cudaEventRecord(start_event));
            status = run_forward();
            if (report_status("lstm_forward", status) != 0) {
                lstm_destroy_buffers(&buffers);
                return status;
            }
            FLASHLSTM_CHECK_CUDA(cudaEventRecord(stop_event));
            FLASHLSTM_CHECK_CUDA(cudaEventSynchronize(stop_event));
            float elapsed_ms = 0.0f;
            FLASHLSTM_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
            total_ms += elapsed_ms;
        }
        FLASHLSTM_CHECK_CUDA(cudaDeviceSynchronize());

        const float avg_ms = total_ms / static_cast<float>(timing_iters);
        std::printf("Chunk size %d average time: %.3f ms\n", chunk_size, avg_ms);

        if (avg_ms < best_time_ms) {
            best_time_ms = avg_ms;
            best_chunk = chunk_size;
        }

        lstm_destroy_buffers(&buffers);
    }

    FLASHLSTM_CHECK_CUDA(cudaEventDestroy(start_event));
    FLASHLSTM_CHECK_CUDA(cudaEventDestroy(stop_event));

    if (best_chunk >= 0) {
        std::printf("Best chunk size: %d (%.3f ms)\n", best_chunk, best_time_ms);
    } else {
        std::printf("Unable to determine best chunk size.\n");
    }

    return 0;
}
