#pragma once

void vector_add_cuda(const float* d_a,
                     const float* d_b,
                     float* d_c,
                     int n);