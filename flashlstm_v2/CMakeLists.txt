cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 89)
project(flashlstm LANGUAGES CUDA CXX)

find_library(NVTX_LIBRARY NAMES nvToolsExt64_1 nvToolsExt)
if (NOT NVTX_LIBRARY)
    message(FATAL_ERROR "nvToolsExt library not found; required for NVTX profiling ranges")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

add_library(flashlstm SHARED src/lstm_forward.cu src/lstm_backward.cu)
target_include_directories(flashlstm PUBLIC include)
set_target_properties(flashlstm PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(flashlstm PUBLIC cudnn cublasLt cublas ${NVTX_LIBRARY})
set_target_properties(flashlstm PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")

add_executable(test_flashlstm src/main.cu src/cudnn_reference.cpp)
target_link_libraries(test_flashlstm PRIVATE flashlstm)
set_target_properties(test_flashlstm PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(_flashlstm MODULE python/flashlstm/_flashlstm_module.cpp)
target_include_directories(_flashlstm PRIVATE include)
target_link_libraries(_flashlstm PRIVATE flashlstm Python::Module)
target_compile_features(_flashlstm PRIVATE cxx_std_17)
set_target_properties(_flashlstm PROPERTIES PREFIX "" OUTPUT_NAME "_flashlstm" BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")

install(TARGETS flashlstm DESTINATION flashlstm)
install(TARGETS _flashlstm DESTINATION flashlstm)
install(FILES python/flashlstm/__init__.py DESTINATION flashlstm)
