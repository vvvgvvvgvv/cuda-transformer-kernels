cmake_minimum_required(VERSION 3.18)

project(cuda_transformer_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(benchmark_vector_add
    benchmarks/benchmark_vector_add.cu
    kernels/vector_add.cu
)

set_target_properties(benchmark_vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(benchmark_reduction_sum
    benchmarks/benchmark_reduction_sum.cu
    kernels/reduction.cu
)

set_target_properties(benchmark_reduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(benchmark_reduction_max
    benchmarks/benchmark_reduction_max.cu
    kernels/reduction.cu
)

set_target_properties(benchmark_reduction_max PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)