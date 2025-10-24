#pragma once

#include <cstdio>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>

#define FLASHLSTM_CHECK_CUDA(call)                                             \
    do {                                                                       \
        cudaError_t _flashlstm_cuda_status = (call);                           \
        if (_flashlstm_cuda_status != cudaSuccess) {                           \
            std::fprintf(stderr,                                               \
                         "CUDA error %s (%d) at %s:%d\n",                      \
                         cudaGetErrorString(_flashlstm_cuda_status),           \
                         static_cast<int>(_flashlstm_cuda_status),             \
                         __FILE__,                                             \
                         __LINE__);                                            \
            return static_cast<int>(_flashlstm_cuda_status);                   \
        }                                                                      \
    } while (0)

#define FLASHLSTM_VALIDATE(condition, status) \
    do {                                      \
        if (!(condition)) {                   \
            return (status);                  \
        }                                     \
    } while (0)

#define FLASHLSTM_CHECK_CUTLASS(call)                                         \
    do {                                                                      \
        cutlass::Status _flashlstm_cutlass_status = (call);                   \
        if (_flashlstm_cutlass_status != cutlass::Status::kSuccess) {         \
            std::fprintf(stderr,                                              \
                         "CUTLASS error %s (%d) at %s:%d\n",                  \
                         cutlassGetStatusString(_flashlstm_cutlass_status),   \
                         static_cast<int>(_flashlstm_cutlass_status),         \
                         __FILE__,                                            \
                         __LINE__);                                           \
            return static_cast<int>(cudaErrorUnknown);                        \
        }                                                                     \
    } while (0)

#define ERR_PROPAGATE(expr)                   \
    do {                                      \
        int _flashlstm_err = (expr);          \
        if (_flashlstm_err != 0) {            \
            return _flashlstm_err;            \
        }                                     \
    } while (0)
