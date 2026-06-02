#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include "cuda_utils.h"

void vector_add_cuda(const float* d_a,
                     const float* d_b,
                     float* d_c,
                     int n);

void vector_add_cpu(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& c) {

    // 用普通 for 循环计算 c[i] = a[i] + b[i]
    int n=static_cast<int>(a.size());
    for(int i=0;i<n;i++){
        c[i]=a[i]+b[i];
    }

}

float max_abs_error(const std::vector<float>& ref,
                    const std::vector<float>& out) {

    // 返回 ref 和 out 的最大绝对误差
    float max_error=0.0f;
    int n=static_cast<int>(ref.size());
    for(int i=0;i<n;i++){
        max_error=std::max(max_error,std::fabs(ref[i]-out[i]));
    }
    return max_error;
    
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    // 创建 h_a, h_b, h_c_cpu, h_c_gpu 四个 vector<float>
    std::vector<float>h_a(n);
    std::vector<float>h_b(n);
    std::vector<float>h_c_cpu(n);
    std::vector<float>h_c_gpu(n);

    // 用 std::mt19937 和 uniform_real_distribution 初始化 h_a/h_b
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for(int i=0;i<n;i++){
        h_a[i]=dist(rng);
        h_b[i]=dist(rng);
    }

    // 调用 vector_add_cpu 得到 h_c_cpu
    vector_add_cpu(h_a,h_b,h_c_cpu);

    // 定义 float* d_a, d_b, d_c
    float* d_a=nullptr;
    float* d_b=nullptr;
    float* d_c=nullptr;
   
    // cudaMalloc 三个 device 数组
    CUDA_CHECK(cudaMalloc(&d_a,bytes));
    CUDA_CHECK(cudaMalloc(&d_b,bytes));
    CUDA_CHECK(cudaMalloc(&d_c,bytes));

    // cudaMemcpy h_a/h_b 到 d_a/d_b
    CUDA_CHECK(cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,h_b.data(),bytes,cudaMemcpyHostToDevice));


    // 调用 vector_add_cuda，然后 CUDA_KERNEL_CHECK()
    vector_add_cuda(d_a,d_b,d_c,n);
    CUDA_KERNEL_CHECK();

    // cudaMemcpy d_c 到 h_c_gpu
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(),d_c,bytes,cudaMemcpyDeviceToHost));
    // TODO 9:
    // 计算 max_abs_error
    float max_err=max_abs_error(h_c_cpu,h_c_gpu);


    // 打印 PASS / FAIL
    std::cout << (err < 1e-6f ? "PASS" : "FAIL") << " max_abs_error=" << max_err << '\n';

    // cudaFree
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}