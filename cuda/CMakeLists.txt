cmake_minimum_required(VERSION 3.24)
project(gpu_kernel_lab_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure explicitly, for example:
# cmake -S cuda -B build/cuda -DCMAKE_CUDA_ARCHITECTURES=70
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES explicitly (70 for V100, 80 for A100, 90 for H100)")
endif()

add_executable(naive_gemm naive_gemm.cu)
target_compile_options(naive_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-lineinfo>)
