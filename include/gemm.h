#pragma once

// Computes C = A * B with a straightforward CUDA kernel.
// Matrices are row-major: A is M x K, B is K x N, and C is M x N.
void gemm_naive_cuda(const float* d_A,
                     const float* d_B,
                     float* d_C,
                     int M,
                     int N,
                     int K);

// Computes C = A * B using TILE_SIZE x TILE_SIZE shared-memory tiles.
// Matrices are row-major: A is M x K, B is K x N, and C is M x N.
void gemm_tiled_cuda(const float* d_A,
                     const float* d_B,
                     float* d_C,
                     int M,
                     int N,
                     int K);
