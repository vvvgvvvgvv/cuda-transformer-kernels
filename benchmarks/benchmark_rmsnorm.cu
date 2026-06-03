#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include "cuda_utils.h"
#include "rmsnorm.h"

void rmsnorm_cpu(const std::vector<float>& input,
                 const std::vector<float>& weight,
                 std::vector<float>& output,
                 int num_rows,
                 int hidden_dim,
                 float eps) {
    for (int row = 0; row < num_rows; ++row) {
        int offset = row * hidden_dim;

        float sum_sq = 0.0f;

        for (int col = 0; col < hidden_dim; ++col) {
            float v = input[offset + col];
            sum_sq += v * v;
        }

        float rms = std::sqrt(sum_sq / hidden_dim + eps);

        for (int col = 0; col < hidden_dim; ++col) {
            output[offset + col] = input[offset + col] / rms * weight[col];
        }
    }
}

float max_abs_error(const std::vector<float>& ref,
                    const std::vector<float>& out) {
    float max_err = 0.0f;

    for (size_t i = 0; i < ref.size(); ++i) {
        float err = std::fabs(ref[i] - out[i]);
        max_err = std::max(max_err, err);
    }

    return max_err;
}

int main() {
    const int num_rows = 32;
    const int hidden_dim = 4096;
    const int block_size = 256;
    const float eps = 1e-6f;

    const int n = num_rows * hidden_dim;
    const size_t input_bytes = n * sizeof(float);
    const size_t weight_bytes = hidden_dim * sizeof(float);

    std::vector<float> h_input(n);
    std::vector<float> h_weight(hidden_dim);
    std::vector<float> h_output_cpu(n);
    std::vector<float> h_output_gpu(n);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(0.5f, 1.5f);

    for (int i = 0; i < n; ++i) {
        h_input[i] = input_dist(rng);
    }

    for (int i = 0; i < hidden_dim; ++i) {
        h_weight[i] = weight_dist(rng);
    }

    rmsnorm_cpu(h_input,
                h_weight,
                h_output_cpu,
                num_rows,
                hidden_dim,
                eps);

    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, input_bytes));

    CUDA_CHECK(cudaMemcpy(d_input,
                          h_input.data(),
                          input_bytes,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_weight,
                          h_weight.data(),
                          weight_bytes,
                          cudaMemcpyHostToDevice));

    rmsnorm_cuda(d_input,
                 d_weight,
                 d_output,
                 num_rows,
                 hidden_dim,
                 eps,
                 block_size);

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(),
                          d_output,
                          input_bytes,
                          cudaMemcpyDeviceToHost));

    float max_err = max_abs_error(h_output_cpu, h_output_gpu);
    bool passed = max_err < 1e-5f;

    std::cout << "RMSNorm Correctness\n";
    std::cout << "Shape: " << num_rows << " x " << hidden_dim << "\n";
    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Max abs error: " << max_err << "\n";
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));

    return passed ? 0 : 1;
}