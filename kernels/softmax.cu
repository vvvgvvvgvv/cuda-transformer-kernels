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

    float local_max = -FLT_MAX;
    for (int col = tid; col < hidden_dim; col += blockDim.x) {
        local_max = fmaxf(local_max, row_input[col]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int active = blockDim.x; active > 1; active = (active + 1) >> 1) {
        int half = (active + 1) >> 1;
        if (tid + half < active) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + half]);
        }
        __syncthreads();
    }

    float row_max = sdata[0];
    float local_sum = 0.0f;
    for (int col = tid; col < hidden_dim; col += blockDim.x) {
        float value = __expf(row_input[col] - row_max);
        row_output[col] = value;
        local_sum += value;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int active = blockDim.x; active > 1; active = (active + 1) >> 1) {
        int half = (active + 1) >> 1;
        if (tid + half < active) {
            sdata[tid] += sdata[tid + half];
        }
        __syncthreads();
    }

    float inv_row_sum = 1.0f / sdata[0];
    for (int col = tid; col < hidden_dim; col += blockDim.x) {
        row_output[col] *= inv_row_sum;
    }
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
