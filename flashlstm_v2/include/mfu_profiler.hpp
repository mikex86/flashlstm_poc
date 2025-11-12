#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace flstm::mfu {

inline double GemmFlops(size_t m, size_t n, size_t k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

inline double GemvFlops(size_t m, size_t n) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n);
}

class Profiler {
public:
    explicit Profiler(const char *label)
        : enabled_(std::getenv("FLASHLSTM_PROFILE_MFU") != nullptr),
          label_(label != nullptr ? label : "unknown") {
        if (!enabled_) {
            return;
        }
        peak_flops_ = DeterminePeakFlops();
        start_time_ = Clock::now();
    }

    ~Profiler() = default;

    void AddUseful(double flops) {
        if (!enabled_) {
            return;
        }
        useful_flops_ += flops;
        total_flops_ += flops;
    }

    void AddTotal(double flops) {
        if (!enabled_) {
            return;
        }
        total_flops_ += flops;
    }

    void Finish() {
        if (!enabled_ || finished_) {
            return;
        }
        finished_ = true;
        const double seconds = std::chrono::duration<double>(Clock::now() - start_time_).count();
        if (seconds <= 0.0) {
            return;
        }
        const double peak_tflops = peak_flops_ / 1e12;
        const double useful_tflops = useful_flops_ / seconds / 1e12;
        const double total_tflops = total_flops_ / seconds / 1e12;
        const double mfu = (peak_tflops > 0.0) ? (useful_tflops / peak_tflops * 100.0) : 0.0;
        const double hfu = (peak_tflops > 0.0) ? (total_tflops / peak_tflops * 100.0) : 0.0;
        std::printf(
            "[flashlstm][%s] MFU=%.2f%% HFU=%.2f%% (useful=%.3f TFLOP/s, matmul=%.3f TFLOP/s, peak=%.3f TFLOP/s)\n",
            label_.c_str(),
            mfu,
            hfu,
            useful_tflops,
            total_tflops,
            peak_tflops
        );
    }

private:
    using Clock = std::chrono::steady_clock;

    static double DeterminePeakFlops() {
        static double cached = -1.0;
        if (cached >= 0.0) {
            return cached;
        }
        const char *override_value = std::getenv("FLASHLSTM_MFU_PEAK_TFLOPS");
        if (override_value != nullptr && override_value[0] != '\0') {
            cached = std::strtod(override_value, nullptr) * 1e12;
            return cached;
        }
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
            cached = 0.0;
            return cached;
        }
        const double clock_hz = static_cast<double>(prop.clockRate) * 1000.0;
        const double sm_count = static_cast<double>(prop.multiProcessorCount);
        constexpr double ops_per_cycle = 2048.0; // Assumes FP16 tensor cores (1024 FMA => 2048 flops).
        cached = clock_hz * sm_count * ops_per_cycle;
        return cached;
    }

    bool enabled_{false};
    bool finished_{false};
    std::string label_;
    double useful_flops_{0.0};
    double total_flops_{0.0};
    double peak_flops_{0.0};
    Clock::time_point start_time_{};
};

} // namespace flstm::mfu
