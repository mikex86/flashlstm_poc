#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace flstm {

void GemmTN(
    int m,
    int n,
    int k,
    const __half *A,
    int lda,
    const __half *B,
    int ldb,
    float *C,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream
);

void GemmNN(
    int m,
    int n,
    int k,
    const __half *A,
    int lda,
    const __half *B,
    int ldb,
    float *C,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream
);

void GemmNT(
    int m,
    int n,
    int k,
    const __half *A,
    int lda,
    const __half *B,
    int ldb,
    float *C,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream
);

} // namespace flstm
