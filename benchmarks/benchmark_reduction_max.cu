#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cfloat>

#include "cuda_utils.h"
#include "reduction.h"

float reduction_max_cpu(const std::vector<float>& x) {
    float max_val = -FLT_MAX;

    for (float v : x) {
        max_val = std::max(max_val, v);
    }

    return max_val;
}

int main() {
    const int n = 1 << 20;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    const size_t input_bytes = n * sizeof(float);
    const size_t partial_bytes = grid_size * sizeof(float);

    std::vector<float> h_x(n);
    std::vector<float> h_partial_max(grid_size);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        h_x[i] = dist(rng);
    }

    float cpu_result = reduction_max_cpu(h_x);

    float* d_x = nullptr;
    float* d_partial_max = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_partial_max, partial_bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), input_bytes, cudaMemcpyHostToDevice));

    reduction_max_cuda(d_x, d_partial_max, n, block_size);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_partial_max.data(),
                          d_partial_max,
                          partial_bytes,
                          cudaMemcpyDeviceToHost));

    // The kernel emits one partial maximum per block; finish the reduction on host.
    float gpu_result = reduction_max_cpu(h_partial_max);

    float abs_error = std::fabs(cpu_result - gpu_result);
    bool passed = abs_error < 1e-6f;

    std::cout << "Reduction Max Correctness\n";
    std::cout << "N: " << n << "\n";
    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Grid size: " << grid_size << "\n";
    std::cout << "CPU result: " << cpu_result << "\n";
    std::cout << "GPU result: " << gpu_result << "\n";
    std::cout << "Abs error:  " << abs_error << "\n";
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_partial_max));

    return passed ? 0 : 1;
}
