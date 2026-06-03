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

set_target_properties(benchmark_reduction_sum PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(benchmark_reduction_max
    benchmarks/benchmark_reduction_max.cu
    kernels/reduction.cu
)

set_target_properties(benchmark_reduction_max PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(benchmark_softmax
    benchmarks/benchmark_softmax.cu
    kernels/softmax.cu
)

set_target_properties(benchmark_softmax PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(benchmark_rmsnorm
    benchmarks/benchmark_rmsnorm.cu
    kernels/rmsnorm.cu
)

set_target_properties(benchmark_rmsnorm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(benchmark_gemm_naive
    benchmarks/benchmark_gemm_naive.cu
    kernels/gemm.cu
)

set_target_properties(benchmark_gemm_naive PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(benchmark_gemm_tiled
    benchmarks/benchmark_gemm_tiled.cu
    kernels/gemm.cu
)

set_target_properties(benchmark_gemm_tiled PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(benchmark_gemm_compare
    benchmarks/benchmark_gemm_compare.cu
    kernels/gemm.cu
)

set_target_properties(benchmark_gemm_compare PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)