#pragma once

void reduction_sum_cuda(const float* d_input,
                        float* d_partial_sums,
                        int n,
                        int block_size);

