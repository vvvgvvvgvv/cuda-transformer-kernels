#include "rmsnorm.h"

#include <cuda_runtime.h>
#include <cmath>

__global__ void rmsnorm_kernel(const float* input,
                               const float* weight,
                               float* output,
                               int num_rows,
                               int hidden_dim,
                               float eps) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= num_rows) {
        return;
    }

    const float* row_input = input + row * hidden_dim;
    float* row_output = output + row * hidden_dim;

    // Each block handles one row; threads stride across the hidden dimension.
    float local_sum = 0.0f;

    for (int col = tid; col < hidden_dim; col += blockDim.x) {
        float v = row_input[col];
        local_sum += v * v;
    }

    // Reduce per-thread squared sums to the row total.
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float sum_sq = sdata[0];
    float rms = sqrtf(sum_sq / hidden_dim + eps);

    for (int col = tid; col < hidden_dim; col += blockDim.x) {
        row_output[col] = row_input[col] / rms * weight[col];
    }
}

void rmsnorm_cuda(const float* d_input,
                  const float* d_weight,
                  float* d_output,
                  int num_rows,
                  int hidden_dim,
                  float eps,
                  int block_size) {
    int grid_size = num_rows;
    size_t shared_mem_bytes = block_size * sizeof(float);

    rmsnorm_kernel<<<grid_size, block_size, shared_mem_bytes>>>(
        d_input,
        d_weight,
        d_output,
        num_rows,
        hidden_dim,
        eps
    );
}
