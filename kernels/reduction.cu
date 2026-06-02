#include "reduction.h"

#include <cuda_runtime.h>
#include <cfloat>

__global__ void reduction_sum_kernels(const float* d_input,
                                      float* d_partial_sums,
                                      int n){
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
        d_partial_sums[blockIdx.x]=sdata[0];
    }

}

void reduction_sum_cuda(const float* d_input,
                        float* d_partial_sums,
                        int n,
                        int block_size) {
    int grid_size = (n + block_size - 1) / block_size;
    size_t shared_mem_bytes = block_size * sizeof(float);

    reduction_sum_kernel<<<grid_size, block_size, shared_mem_bytes>>>(
        d_input,
        d_partial_sums,
        n
    );
}
__global__ void reduction_max_kernels(const float*d_input,
                                            float*d_partail_max,
                                            int n){
    extern __shared__ float sdata[];

    int tid=threadIdx.x;
    int global_idx=blockIdx.x*blockDim.x+threadIdx.x;
    
    if(global<n){
        sdata[tid]=d_input[global_idx];
    }
    else{
        sdata[tid]=-FLT_MAX;
    }
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            sdata[tid]=fmax(sdata[tid],sdata[tid+stride]);
        }
    }
    if(tid==0){
        d_partail_max[blockIdx.x]=sdata[0];
    }
}

void reduction_max_cuda(const float* d_input,
                        float* d_partial_max,
                        int n,
                        int block_size){
    int grid_size=(block_size+n-1)/block_size;
    size_t shared_mem_bytes = block_size * sizeof(float);

    reduction_max_cuda <<<grid_size,block_size,shared_mem_bytes>>>(
        d_input,
        d_partial_sums,
        n
    );
} 