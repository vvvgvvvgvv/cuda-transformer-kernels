#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cfloat>

#include "cuda_utils.h"
#include "softmax.h"

void softmax_cpu(const std::vector<float>& input,
                 std::vector<float>& output,
                 int num_rows,
                 int hidden_dim) {
    for (int row = 0; row < num_rows; ++row) {
        int offset = row * hidden_dim;

        // Subtract the row maximum for numerical stability.
        float max_val = -FLT_MAX;
        for (int col = 0; col < hidden_dim; ++col) {
            max_val = std::max(max_val, input[offset + col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < hidden_dim; ++col) {
            float val = std::exp(input[offset + col] - max_val);
            output[offset + col] = val;
            sum += val;
        }

        for (int col = 0; col < hidden_dim; ++col) {
            output[offset + col] /= sum;
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
    const int num_rows = 1024;
    const int hidden_dim = 1024;
    const int block_size = 256;

    const int n = num_rows * hidden_dim;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_input(n);
    std::vector<float> h_output_cpu(n);
    std::vector<float> h_output_gpu(n);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (int i = 0; i < n; ++i) {
        h_input[i] = dist(rng);
    }

    softmax_cpu(h_input, h_output_cpu, num_rows, hidden_dim);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(d_input,
                          h_input.data(),
                          bytes,
                          cudaMemcpyHostToDevice));

    softmax_cuda(d_input,
                 d_output,
                 num_rows,
                 hidden_dim,
                 block_size);

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(),
                          d_output,
                          bytes,
                          cudaMemcpyDeviceToHost));

    float max_err = max_abs_error(h_output_cpu, h_output_gpu);
    bool passed = max_err < 1e-5f;

    std::cout << "Softmax Correctness\n";
    std::cout << "Shape: " << num_rows << " x " << hidden_dim << "\n";
    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Max abs error: " << max_err << "\n";
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return passed ? 0 : 1;
}
