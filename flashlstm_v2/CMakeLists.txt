cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES 89)
project(flashlstm CUDA)

find_library(NVTX_LIBRARY NAMES nvToolsExt64_1 nvToolsExt)
if (NOT NVTX_LIBRARY)
    message(FATAL_ERROR "nvToolsExt library not found; required for NVTX profiling ranges")
endif()

add_library(flashlstm SHARED src/lstm_forward.cu src/lstm_backward.cu)
target_include_directories(flashlstm PUBLIC include)
set_target_properties(flashlstm PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(flashlstm PUBLIC cudnn cublasLt cublas ${NVTX_LIBRARY})

add_executable(test_flashlstm src/main.cu)
target_link_libraries(test_flashlstm PRIVATE flashlstm)
set_target_properties(test_flashlstm PROPERTIES POSITION_INDEPENDENT_CODE ON)
