#pragma once

// Computes one partial sum per CUDA block.
// The caller finishes the final accumulation over partial sums.
void reduction_sum_cuda(const float* d_input,
                        float* d_partial_sums,
                        int n,
                        int block_size);

// Computes one partial maximum per CUDA block.
// The caller finishes the final maximum over partial values.
void reduction_max_cuda(const float* d_input,
                        float* d_partial_max,
                        int n,
                        int block_size);
