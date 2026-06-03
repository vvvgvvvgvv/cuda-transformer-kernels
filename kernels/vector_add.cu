#include "vector_add.h"

#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int n) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;

    // Guard the final partial block.
    if(index<n){
        c[index]=a[index]+b[index];
    }
}

void vector_add_cuda(const float* d_a,
                     const float* d_b,
                     float* d_c,
                     int n) {
    int block_size=256;
    int grid_size=(n+block_size-1)/block_size;

    vector_add_kernel<<<grid_size,block_size>>>(d_a,d_b,d_c,n);
}
