#include "flashlstm/lstm_api.h"
#include "cudnn_runner.h"
#include "error_checks.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstdlib>

#include "nvtx_profinst.h"

namespace {
    template<typename T>
    struct DeviceBuffer {
        T *data;

        DeviceBuffer() : data(nullptr) {
        }

        ~DeviceBuffer() {
            if (data != nullptr) {
                cudaFree(data);
            }
        }

        cudaError_t allocate(const std::size_t elements) {
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

    int report_cuda_error(const cudaError_t error) {
        std::fprintf(stderr,
                     "CUDA error %s (%d)\n",
                     cudaGetErrorString(error),
                     static_cast<int>(error));
        return error;
    }

    int handle_cuda_error(const cudaError_t error) {
        if (error == cudaSuccess) {
            return 0;
        }
        return report_cuda_error(error);
    }

    int handle_cublas_error(const cublasStatus_t status, const char *function_name) {
        if (status == CUBLAS_STATUS_SUCCESS) {
            return 0;
        }
        std::fprintf(stderr,
                     "cuBLAS error (%d) in %s during run_lstm::main\n",
                     static_cast<int>(status),
                     function_name);
        return status;
    }

    int handle_cublaslt_error(const cublasStatus_t status, const char *function_name) {
        if (status == CUBLAS_STATUS_SUCCESS) {
            return 0;
        }
        std::fprintf(stderr,
                     "cuBLASLt error (%d) in %s during run_lstm::main\n",
                     static_cast<int>(status),
                     function_name);
        return status;
    }
} // namespace

int main() {
    constexpr std::size_t seq_len = 2048;
    constexpr std::size_t batch = 32;
    constexpr std::size_t input_size = 1024;
    constexpr std::size_t hidden_size = 1024;
    constexpr auto compute_precision = LSTM_COMPUTE_PRECISION_FP16_ACC16;

    const std::size_t x_elems = seq_len * batch * input_size;
    const std::size_t w_ih_elems = 4 * hidden_size * input_size;
    const std::size_t w_hh_elems = 4 * hidden_size * hidden_size;
    const std::size_t bias_elems = 4 * hidden_size;
    const std::size_t state_elems = batch * hidden_size;
    const std::size_t output_elems = seq_len * batch * hidden_size;

    const auto make_zero_vector = [](const std::size_t count) {
        return std::vector<float>(count, 0.0f);
    };

    auto x = make_zero_vector(x_elems);
    auto w_ih = make_zero_vector(w_ih_elems);
    auto w_hh = make_zero_vector(w_hh_elems);
    auto b_ih = make_zero_vector(bias_elems);
    auto b_hh = make_zero_vector(bias_elems);
    auto h0 = make_zero_vector(state_elems);
    auto c0 = make_zero_vector(state_elems);

    auto flash_output = make_zero_vector(output_elems);
    auto flash_hn = make_zero_vector(state_elems);
    auto flash_cn = make_zero_vector(state_elems);

    DeviceBuffer<float> d_x;
    DeviceBuffer<float> d_w_ih;
    DeviceBuffer<float> d_w_hh;
    DeviceBuffer<float> d_b_ih;
    DeviceBuffer<float> d_b_hh;
    DeviceBuffer<float> d_h0;
    DeviceBuffer<float> d_c0;
    DeviceBuffer<float> d_flash_output;
    DeviceBuffer<float> d_flash_hn;
    DeviceBuffer<float> d_flash_cn;

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    FLASHLSTM_CHECK_CUDA(cudaEventCreate(&start_event));
    FLASHLSTM_CHECK_CUDA(cudaEventCreate(&stop_event));

    const auto cleanup_and_return = [&](const int status) -> int {
        int destroy_status = handle_cuda_error(cudaEventDestroy(start_event));
        if (destroy_status != 0) {
            return destroy_status;
        }
        destroy_status = handle_cuda_error(cudaEventDestroy(stop_event));
        if (destroy_status != 0) {
            return destroy_status;
        }
        return status;
    };

    cublasHandle_t cublas_handle = nullptr;
    if (int blas_status = handle_cublas_error(cublasCreate(&cublas_handle), "cublasCreate")) {
        return cleanup_and_return(blas_status);
    }
    if (int blas_status = handle_cublas_error(cublasDestroy(cublas_handle), "cublasDestroy")) {
        return cleanup_and_return(blas_status);
    }
    cublasLtHandle_t cublaslt_handle = nullptr;
    if (int blaslt_status = handle_cublaslt_error(cublasLtCreate(&cublaslt_handle), "cublasLtCreate")) {
        return cleanup_and_return(blaslt_status);
    }
    if (int blaslt_status = handle_cublaslt_error(cublasLtDestroy(cublaslt_handle), "cublasLtDestroy")) {
        return cleanup_and_return(blaslt_status);
    }

    int status = initialize_cudnn();
    if (status != 0) {
        return cleanup_and_return(status);
    }

    NVTX_SCOPED_RANGE("run_lstm::main");

    struct DeviceAllocation {
        DeviceBuffer<float> *buffer;
        std::size_t elements;
    };

    const DeviceAllocation device_allocations[] = {
        {&d_x, x_elems},
        {&d_w_ih, w_ih_elems},
        {&d_w_hh, w_hh_elems},
        {&d_b_ih, bias_elems},
        {&d_b_hh, bias_elems},
        {&d_h0, state_elems},
        {&d_c0, state_elems},
        {&d_flash_output, output_elems},
        {&d_flash_hn, state_elems},
        {&d_flash_cn, state_elems},
    };

    for (const auto &allocation: device_allocations) {
        FLASHLSTM_CHECK_CUDA(allocation.buffer->allocate(allocation.elements));
    }

    struct HostToDeviceCopy {
        DeviceBuffer<float> *device;
        const std::vector<float> *host;
        std::size_t elements;
    };

    const HostToDeviceCopy host_to_device_copies[] = {
        {&d_x, &x, x_elems},
        {&d_w_ih, &w_ih, w_ih_elems},
        {&d_w_hh, &w_hh, w_hh_elems},
        {&d_b_ih, &b_ih, bias_elems},
        {&d_b_hh, &b_hh, bias_elems},
        {&d_h0, &h0, state_elems},
        {&d_c0, &c0, state_elems},
    };

    for (const auto &copy: host_to_device_copies) {
        FLASHLSTM_CHECK_CUDA(cudaMemcpy(copy.device->data,
            copy.host->data(),
            copy.elements * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    constexpr int input_proj_chsz = 64; // optimal for seq_len=2048, batch=32, input_size=1024, hidden_size=1024

    lstm_buffers buffers{};
    status = lstm_create_buffers(compute_precision,
                                 seq_len,
                                 batch,
                                 input_size,
                                 hidden_size,
                                 input_proj_chsz,
                                 &buffers);
    if (status != 0) {
        return cleanup_and_return(status);
    }

    bool use_cuda_graph = false;
    if (const char *env = std::getenv("FLASHLSTM_USE_CUDA_GRAPH")) {
        use_cuda_graph = std::atoi(env) != 0;
    }
    if (use_cuda_graph) {
        status = lstm_set_execution_mode(&buffers, LSTM_EXECUTION_MODE_GRAPH);
        if (status != 0) {
            lstm_destroy_buffers(&buffers);
            return cleanup_and_return(status);
        }
    }

    status = lstm_pack_weights(compute_precision,
                               d_w_ih.data,
                               d_w_hh.data,
                               input_size,
                               hidden_size,
                               &buffers);
    if (status != 0) {
        lstm_destroy_buffers(&buffers);
        return cleanup_and_return(status);
    }

    status = lstm_forward(d_x.data,
                          d_b_ih.data,
                          d_b_hh.data,
                          d_h0.data,
                          d_c0.data,
                          d_flash_output.data,
                          d_flash_hn.data,
                          d_flash_cn.data,
                          seq_len,
                          batch,
                          input_size,
                          hidden_size,
                          &buffers,
                          compute_precision);
    if (status != 0) {
        lstm_destroy_buffers(&buffers);
        return cleanup_and_return(status);
    }
    FLASHLSTM_CHECK_CUDA(cudaDeviceSynchronize());

    const DeviceAllocation zeroed_buffers[] = {
        {&d_flash_output, output_elems},
        {&d_flash_hn, state_elems},
        {&d_flash_cn, state_elems},
    };

    for (const auto &zero_target: zeroed_buffers) {
        FLASHLSTM_CHECK_CUDA(cudaMemset(zero_target.buffer->data,
            0,
            zero_target.elements * sizeof(float)));
    }

    float flash_ms = 0.0f; {
        FLASHLSTM_CHECK_CUDA(cudaEventRecord(start_event));
        status = lstm_forward(d_x.data,
                              d_b_ih.data,
                              d_b_hh.data,
                              d_h0.data,
                              d_c0.data,
                              d_flash_output.data,
                              d_flash_hn.data,
                              d_flash_cn.data,
                              seq_len,
                              batch,
                              input_size,
                              hidden_size,
                              &buffers,
                              compute_precision);
        FLASHLSTM_CHECK_CUDA(cudaEventRecord(stop_event));
        FLASHLSTM_CHECK_CUDA(cudaEventSynchronize(stop_event));
        FLASHLSTM_CHECK_CUDA(cudaEventElapsedTime(&flash_ms, start_event, stop_event));
        if (status != 0) {
            lstm_destroy_buffers(&buffers);
            return cleanup_and_return(status);
        }
    }
    std::printf("FlashLSTM forward: %.3f ms\n", flash_ms);

    struct DeviceToHostCopy {
        std::vector<float> *host;
        const DeviceBuffer<float> *device;
        std::size_t elements;
    };

    const DeviceToHostCopy device_to_host_copies[] = {
        {&flash_output, &d_flash_output, output_elems},
        {&flash_hn, &d_flash_hn, state_elems},
        {&flash_cn, &d_flash_cn, state_elems},
    };

    for (const auto &copy: device_to_host_copies) {
        FLASHLSTM_CHECK_CUDA(cudaMemcpy(copy.host->data(),
            copy.device->data,
            copy.elements * sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    FLASHLSTM_CHECK_CUDA(cudaDeviceSynchronize());
    lstm_destroy_buffers(&buffers);

    auto cudnn_output = make_zero_vector(output_elems);
    auto cudnn_hn = make_zero_vector(state_elems);
    auto cudnn_cn = make_zero_vector(state_elems);

    // Warm up cuDNN path so any lazy library loading happens before timing.
    status = run_cudnn_lstm(x.data(),
                            h0.data(),
                            c0.data(),
                            cudnn_output.data(),
                            cudnn_hn.data(),
                            cudnn_cn.data(),
                            seq_len,
                            batch,
                            input_size,
                            hidden_size);
    if (status != 0) {
        return cleanup_and_return(status);
    }
    std::fill(cudnn_output.begin(), cudnn_output.end(), 0.0f);
    std::fill(cudnn_hn.begin(), cudnn_hn.end(), 0.0f);
    std::fill(cudnn_cn.begin(), cudnn_cn.end(), 0.0f);

    float cudnn_ms = 0.0f; {
        FLASHLSTM_CHECK_CUDA(cudaEventRecord(start_event));
        status = run_cudnn_lstm(x.data(),
                                h0.data(),
                                c0.data(),
                                cudnn_output.data(),
                                cudnn_hn.data(),
                                cudnn_cn.data(),
                                seq_len,
                                batch,
                                input_size,
                                hidden_size);
        FLASHLSTM_CHECK_CUDA(cudaEventRecord(stop_event));
        FLASHLSTM_CHECK_CUDA(cudaEventSynchronize(stop_event));
        FLASHLSTM_CHECK_CUDA(cudaEventElapsedTime(&cudnn_ms, start_event, stop_event));
        if (status != 0) {
            return cleanup_and_return(status);
        }
    }
    std::printf("cuDNN LSTM forward: %.3f ms\n", cudnn_ms);

    FLASHLSTM_CHECK_CUDA(cudaDeviceSynchronize());
    return cleanup_and_return(0);
}
