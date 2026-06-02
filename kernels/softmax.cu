#include "softmax.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void softmax_kernel(const float* input,
                               float* output,
                               int num_rows,
                               int hidden_dim) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= num_rows) {
        return;
    }

    const float* row_input = input + row * hidden_dim;
    float* row_output = output + row * hidden_dim;

    // TODO 1:
    // 每个线程负责 row 里的多个元素，求 local_max

    // TODO 2:
    // 把 local_max 写入 sdata[tid]
    // block 内 reduction 求 row_max

    // TODO 3:
    // 每个线程计算 exp(x - row_max)，同时求 local_sum
    // 可以先把 exp 结果写到 row_output[col]

    // TODO 4:
    // 把 local_sum 写入 sdata[tid]
    // block 内 reduction 求 row_sum

    // TODO 5:
    // 每个线程把 row_output[col] /= row_sum
}

void softmax_cuda(const float* d_input,
                  float* d_output,
                  int num_rows,
                  int hidden_dim,
                  int block_size) {
    int grid_size = num_rows;
    size_t shared_mem_bytes = block_size * sizeof(float);

    softmax_kernel<<<grid_size, block_size, shared_mem_bytes>>>(
        d_input,
        d_output,
        num_rows,
        hidden_dim
    );
}