#pragma once

void reduction_sum_cuda(const float* d_input,
                        float* d_partial_sums,
                        int n,
                        int block_size);

void reduction_max_cuda(const float* d_input,
                        float* d_partial_max,
                        int n,
                        int block_size);