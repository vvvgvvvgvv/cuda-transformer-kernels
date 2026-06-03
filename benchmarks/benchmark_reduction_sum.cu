#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include "cuda_utils.h"
#include "reduction.h"

float reduction_sum_cpu(const std::vector<float>& x) {
    float sum_cpu=0.0f;
    for(const float num:x){
        sum_cpu+=num;
    }
    return sum_cpu;
}

int main() {
    const int n = 1 << 20;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    const size_t input_bytes = n * sizeof(float);
    const size_t partial_bytes = grid_size * sizeof(float);

    std::vector<float> h_x(n);
    std::vector<float> h_partial_sums(grid_size);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for(int i=0;i<n;i++){
        h_x[i] = dist(rng);
    }

    float cpu_result = reduction_sum_cpu(h_x);

    float* d_x = nullptr;
    float* d_partial_sums = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_partial_sums, partial_bytes));

    CUDA_CHECK(cudaMemcpy(d_x,h_x.data(),input_bytes,cudaMemcpyHostToDevice));

    reduction_sum_cuda(d_x, d_partial_sums, n, block_size);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(),d_partial_sums,partial_bytes,cudaMemcpyDeviceToHost));

    // The kernel emits one partial sum per block; finish the reduction on host.
    float gpu_result = reduction_sum_cpu(h_partial_sums);

    float abs_error = std::fabs(cpu_result - gpu_result);
    bool passed = abs_error < 1e-2f;

    std::cout << "Reduction Sum Correctness\n";
    std::cout << "CPU result: " << cpu_result << "\n";
    std::cout << "GPU result: " << gpu_result << "\n";
    std::cout << "Abs error:  " << abs_error << "\n";
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_partial_sums));

    return 0;
}
