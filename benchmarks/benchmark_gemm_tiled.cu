#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include "cuda_utils.h"
#include "gemm.h"

void gemm_cpu(const std::vector<float>& A,
              const std::vector<float>& B,
              std::vector<float>& C,
              int M,
              int N,
              int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
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
    const int M = 256;
    const int N = 256;
    const int K = 256;

    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M * N);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = dist(rng);
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = dist(rng);
    }

    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A,
                          h_A.data(),
                          bytes_A,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B,
                          h_B.data(),
                          bytes_B,
                          cudaMemcpyHostToDevice));

    gemm_tiled_cuda(d_A, d_B, d_C, M, N, K);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(),
                          d_C,
                          bytes_C,
                          cudaMemcpyDeviceToHost));

    float max_err = max_abs_error(h_C_cpu, h_C_gpu);
    bool passed = max_err < 1e-3f;

    std::cout << "Tiled GEMM Correctness\n";
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "Max abs error: " << max_err << "\n";
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? 0 : 1;
}