cmake_minimum_required(VERSION 3.18)

project(flashlstm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CUTLASS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)
if (EXISTS ${CUTLASS_ROOT}/include)
    message(STATUS "Found CUTLASS at ${CUTLASS_ROOT}")
else()
    message(WARNING "CUTLASS headers not found at ${CUTLASS_ROOT}. Run scripts/ensure_cutlass.py")
endif()

add_library(flashlstm SHARED src/lstm.cu src/kernels.cu)
target_include_directories(flashlstm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (EXISTS ${CUTLASS_ROOT}/include)
    target_include_directories(flashlstm PRIVATE ${CUTLASS_ROOT}/include)
endif()

find_path(NVTOOLSEXT_INCLUDE_DIR nvToolsExt.h
    HINTS
        /usr/local/cuda/include
        /usr/include
        /usr/local/include)
find_library(NVTOOLSEXT_LIBRARY nvToolsExt
    HINTS
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/lib64
        /usr/local/lib)
if (NOT NVTOOLSEXT_INCLUDE_DIR OR NOT NVTOOLSEXT_LIBRARY)
    message(FATAL_ERROR "nvToolsExt not found. Install CUDA profiling tools support.")
endif()
target_include_directories(flashlstm PRIVATE ${NVTOOLSEXT_INCLUDE_DIR})
target_compile_definitions(flashlstm PRIVATE CUTLASS_ARCH_MMA_SM80_SUPPORTED=1 CUTLASS_ARCH_ENABLED_SM80=1)
set_target_properties(flashlstm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(flashlstm PUBLIC ${NVTOOLSEXT_LIBRARY})

add_executable(run_lstm src/run_lstm.cu src/cudnn_runner.cu)
target_include_directories(run_lstm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(run_lstm PRIVATE ${NVTOOLSEXT_INCLUDE_DIR})
if (EXISTS ${CUTLASS_ROOT}/include)
    target_include_directories(run_lstm PRIVATE ${CUTLASS_ROOT}/include)
endif()
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS
        $ENV{CUDNN_ROOT}/include
        /usr/local/cuda/include
        /usr/include
        /usr/local/include)
find_library(CUDNN_LIBRARY cudnn
    HINTS
        $ENV{CUDNN_ROOT}/lib64
        $ENV{CUDNN_ROOT}/lib
        /usr/local/cuda/lib64
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib64
        /usr/local/lib)
if (NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found. Set CUDNN_ROOT or install cuDNN.")
endif()

find_library(CUBLAS_LIBRARY cublas
    HINTS
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/lib64
        /usr/local/lib)
if (NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "cuBLAS not found. Install CUDA toolkit or set library path.")
endif()

find_library(CUBLASLT_LIBRARY cublasLt
    HINTS
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/lib64
        /usr/local/lib)
if (NOT CUBLASLT_LIBRARY)
    message(FATAL_ERROR "cuBLASLt not found. Install CUDA toolkit or set library path.")
endif()

target_include_directories(run_lstm PRIVATE ${CUDNN_INCLUDE_DIR})
target_link_libraries(run_lstm PRIVATE flashlstm ${CUDNN_LIBRARY} ${CUBLAS_LIBRARY} ${CUBLASLT_LIBRARY})

add_subdirectory(flashlstm_v2)
