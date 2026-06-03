#pragma once

// Applies row-wise RMSNorm to a row-major matrix on device memory.
// weight contains hidden_dim scale values shared by all rows.
void rmsnorm_cuda(const float* d_input,
                  const float* d_weight,
                  float* d_output,
                  int num_rows,
                  int hidden_dim,
                  float eps,
                  int block_size);
