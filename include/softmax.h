#pragma once

// Applies numerically stable row-wise softmax to a row-major matrix on device
// memory.
void softmax_cuda(const float* d_input,
                  float* d_output,
                  int num_rows,
                  int hidden_dim,
                  int block_size);
