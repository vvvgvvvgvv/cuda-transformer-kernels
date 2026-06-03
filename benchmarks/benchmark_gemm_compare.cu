#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include "cuda_utils.h"
#include "benchmark_utils.h"
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

float benchmark_naive(const float* d_A,
                      const float* d_B,
                      float* d_C,
                      int M,
                      int N,
                      int K,
                      int warmup,
                      int repeat) {
    for (int i = 0; i < warmup; ++i) {
        gemm_naive_cuda(d_A, d_B, d_C, M, N, K);
    }
    CUDA_KERNEL_CHECK();

    GpuTimer timer;
    timer.start();

    for (int i = 0; i < repeat; ++i) {
        gemm_naive_cuda(d_A, d_B, d_C, M, N, K);
    }

    float total_ms = timer.stop();
    CUDA_KERNEL_CHECK();

    return total_ms / repeat;
}

float benchmark_tiled(const float* d_A,
                      const float* d_B,
                      float* d_C,
                      int M,
                      int N,
                      int K,
                      int warmup,
                      int repeat) {
    for (int i = 0; i < warmup; ++i) {
        gemm_tiled_cuda(d_A, d_B, d_C, M, N, K);
    }
    CUDA_KERNEL_CHECK();

    GpuTimer timer;
    timer.start();

    for (int i = 0; i < repeat; ++i) {
        gemm_tiled_cuda(d_A, d_B, d_C, M, N, K);
    }

    float total_ms = timer.stop();
    CUDA_KERNEL_CHECK();

    return total_ms / repeat;
}

int main() {
    const int M = 512;
    const int N = 512;
    const int K = 512;

    const int warmup = 10;
    const int repeat = 50;

    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_naive(M * N);
    std::vector<float> h_C_tiled(M * N);

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

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // Correctness: naive
    gemm_naive_cuda(d_A, d_B, d_C, M, N, K);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    float naive_err = max_abs_error(h_C_cpu, h_C_naive);

    // Correctness: tiled
    gemm_tiled_cuda(d_A, d_B, d_C, M, N, K);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_C_tiled.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    float tiled_err = max_abs_error(h_C_cpu, h_C_tiled);

    bool naive_passed = naive_err < 1e-2f;
    bool tiled_passed = tiled_err < 1e-2f;

    // Benchmark
    float naive_ms = benchmark_naive(d_A, d_B, d_C, M, N, K, warmup, repeat);
    float tiled_ms = benchmark_tiled(d_A, d_B, d_C, M, N, K, warmup, repeat);

    float speedup = naive_ms / tiled_ms;

    std::cout << "GEMM Benchmark Compare\n";
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "Warmup: " << warmup << ", Repeat: " << repeat << "\n\n";

    std::cout << "Naive GEMM\n";
    std::cout << "  Time: " << naive_ms << " ms\n";
    std::cout << "  Max abs error: " << naive_err << "\n";
    std::cout << "  Status: " << (naive_passed ? "PASS" : "FAIL") << "\n\n";

    std::cout << "Tiled GEMM\n";
    std::cout << "  Time: " << tiled_ms << " ms\n";
    std::cout << "  Max abs error: " << tiled_err << "\n";
    std::cout << "  Status: " << (tiled_passed ? "PASS" : "FAIL") << "\n\n";

    std::cout << "Speedup: " << speedup << "x\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return (naive_passed && tiled_passed) ? 0 : 1;
}