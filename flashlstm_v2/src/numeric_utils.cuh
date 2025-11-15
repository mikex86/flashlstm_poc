#pragma once

#include <cmath>

namespace flstm {
namespace numeric {

constexpr float kFp16SafeMax = 60000.0f;
constexpr float kSigmoidEpsilon = 1.0e-6f;

__host__ __device__ __forceinline__ float FastExp(float value) {
#if defined(__CUDA_ARCH__)
    return __expf(value);
#else
    return std::exp(value);
#endif
}

__host__ __device__ __forceinline__ float StableSigmoid(float value) {
    float result;
    if (value >= 0.0f) {
        const float neg = FastExp(-value);
        result = 1.0f / (1.0f + neg);
    } else {
        const float pos = FastExp(value);
        result = pos / (1.0f + pos);
    }
    if (result < kSigmoidEpsilon) {
        result = kSigmoidEpsilon;
    } else if (result > 1.0f - kSigmoidEpsilon) {
        result = 1.0f - kSigmoidEpsilon;
    }
    return result;
}

__host__ __device__ __forceinline__ float ClampToHalfRange(float value) {
    if (value > kFp16SafeMax) {
        return kFp16SafeMax;
    }
    if (value < -kFp16SafeMax) {
        return -kFp16SafeMax;
    }
    return value;
}

__host__ __device__ __forceinline__ float FiniteOrZero(float value) {
#if defined(__CUDA_ARCH__)
    if (isnan(value)) {
        return 0.0f;
    }
    if (isinf(value)) {
        return copysignf(kFp16SafeMax, value);
    }
    return value;
#else
    if (std::isnan(value)) {
        return 0.0f;
    }
    if (std::isinf(value)) {
        return std::copysign(kFp16SafeMax, value);
    }
    return value;
#endif
}

__host__ __device__ __forceinline__ float FiniteOrDefault(float value, float fallback) {
#if defined(__CUDA_ARCH__)
    return isfinite(value) ? value : fallback;
#else
    return std::isfinite(value) ? value : fallback;
#endif
}

} // namespace numeric
} // namespace flstm
