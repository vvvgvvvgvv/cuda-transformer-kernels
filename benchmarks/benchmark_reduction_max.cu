#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include "cuda_utils.h"
#include "reduction.h"

float reduction_max_cpu(const std::vector<float>& x) {
    float max_val = -FLT_MAX;

    for(float v : x) {
        max_val = std::max(max_val, v);
    }

    return max_val;
}

int main(){
    const int n = 1 << 20;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    const size_t input_bytes = n * sizeof(float);
    const size_t partial_bytes = grid_size * sizeof(float);
    
    
}

