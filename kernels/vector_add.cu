#include "vector_add.h"

#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int n) {
    // TODO 1:
    // 计算当前线程负责的全局 index
    int index=blockIdx.x*blockDim.x+threadIdx.x;

    // TODO 2:
    // 如果 index < n，就计算 c[index] = a[index] + b[index]
    if(index<n){
        c[index]=a[index]+b[index];
    }
}

void vector_add_cuda(const float* d_a,
                     const float* d_b,
                     float* d_c,
                     int n) {
    // TODO 3:
    // 设置 block_size = 256
    int block_size=256;
    // TODO 4:
    // 根据 n 和 block_size 计算 grid_size
    int grid_size=(n+block_size-1)/block_size;
    // TODO 5:
    // 启动 vector_add_kernel
    vector_add_kernel<<<grid_size,block_size>>>(d_a,d_b,d_c,n);
}