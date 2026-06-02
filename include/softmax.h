#pragma once

void softmax_cuda(const float* d_input,
                  float* d_output,
                  int num_rows,
                  int hidden_dim,
                  int block_size);