#pragma once

void gemm_naive_cuda(const float* d_A,
                     const float* d_B,
                     float* d_C,
                     int M,
                     int N,
                     int K);

void gemm_tiled_cuda(const float* d_A,
                     const float* d_B,
                     float* d_C,
                     int M,
                     int N,
                     int K);