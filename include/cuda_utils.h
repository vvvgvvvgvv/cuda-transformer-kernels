#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
//错误检查宏
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::fprintf(stderr,                                       \
                     "CUDA error at %s:%d: %s\n",                  \
                     __FILE__,                                     \
                     __LINE__,                                     \
                     cudaGetErrorString(err));                     \
        std::exit(EXIT_FAILURE);                                   \
    }                                                             \
} while (0)

#define CUDA_KERNEL_CHECK() do {                                   \
    CUDA_CHECK(cudaGetLastError());                                \
    CUDA_CHECK(cudaDeviceSynchronize());                           \
} while (0)