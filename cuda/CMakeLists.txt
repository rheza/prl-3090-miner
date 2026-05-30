cmake_minimum_required(VERSION 3.24)
project(prl_cuda LANGUAGES CXX CUDA)

# RTX 3090 == sm_86. Override with -DCMAKE_CUDA_ARCHITECTURES="86;80".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Where to find upstream PORTABLE kernels to vendor (blake3, quantize, denoise, noise-gen).
# These compile on sm_86 unchanged (docs/cuda-sm86-port.md §4).
set(PEARL_GEMM_DIR "" CACHE PATH "path to pearl-official/miner/pearl-gemm")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(PRL_SOURCES
  src/device_manager.cpp
  src/nvml_monitor.cpp
  src/noisy_gemm_sm86.cu
  src/proof_extract_sm86.cu
  src/job_runner.cu
  bindings/prl_cuda.cpp
)

pybind11_add_module(prl_cuda ${PRL_SOURCES})
target_include_directories(prl_cuda PRIVATE include)
if(PEARL_GEMM_DIR)
  target_include_directories(prl_cuda PRIVATE "${PEARL_GEMM_DIR}/csrc")
endif()
target_link_libraries(prl_cuda PRIVATE CUDA::cudart CUDA::nvml)
set_target_properties(prl_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Standalone C++ smoke tests (not the python golden harness).
add_executable(cuda_correctness_test tests/cuda_correctness_test.cpp
               src/device_manager.cpp src/nvml_monitor.cpp)
target_include_directories(cuda_correctness_test PRIVATE include)
target_link_libraries(cuda_correctness_test PRIVATE CUDA::cudart CUDA::nvml)

add_executable(kernel_benchmark tests/kernel_benchmark.cpp
               src/device_manager.cpp)
target_include_directories(kernel_benchmark PRIVATE include)
target_link_libraries(kernel_benchmark PRIVATE CUDA::cudart)
