#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "cuda_utils.h"

void vector_add_cuda(const float* d_a,
                     const float* d_b,
                     float* d_c,
                     int n);

void vector_add_cpu(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& c) {
    // TODO:
    // 用普通 for 循环计算 c[i] = a[i] + b[i]
}

float max_abs_error(const std::vector<float>& ref,
                    const std::vector<float>& out) {
    // TODO:
    // 返回 ref 和 out 的最大绝对误差
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    // TODO 1:
    // 创建 h_a, h_b, h_c_cpu, h_c_gpu 四个 vector<float>

    // TODO 2:
    // 用 std::mt19937 和 uniform_real_distribution 初始化 h_a/h_b

    // TODO 3:
    // 调用 vector_add_cpu 得到 h_c_cpu

    // TODO 4:
    // 定义 float* d_a, d_b, d_c

    // TODO 5:
    // cudaMalloc 三个 device 数组

    // TODO 6:
    // cudaMemcpy h_a/h_b 到 d_a/d_b

    // TODO 7:
    // 调用 vector_add_cuda，然后 CUDA_KERNEL_CHECK()

    // TODO 8:
    // cudaMemcpy d_c 到 h_c_gpu

    // TODO 9:
    // 计算 max_abs_error

    // TODO 10:
    // 打印 PASS / FAIL

    // TODO 11:
    // cudaFree

    return 0;
}