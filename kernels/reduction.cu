#include "reduction.h"

#include <cuda_runtime.h>

__global__ void reduction_sum_kernels(const float* d_input,
                                      float* d_partial_sums,
                                      int n,){
    extern __shared__ float sdata[];
    int tid=threadIdx.x;
    int global_idx=blockIdx.x*blockDim.x+threadIdx.x;
    // 每个线程从 global memory 读一个数到 shared memory
    if(global_idx<n){
        sdata[tid]=d_input[global_idx];
    }
    else{
        sdata[tid]=0.0f;
    }
    __syncthreads();
    //在 shared memory 里做 reduction
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            sdata[tid]+=sdata[tid+stride];
        }
        __syncthreads();
    }

    if(tid==0){
        d__partial_sums[blockIdx.x]=sdata[0];
    }

}