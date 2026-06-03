#pragma once

// Launches elementwise vector addition on device memory.
// All pointers must refer to arrays with at least n float elements.
void vector_add_cuda(const float* d_a,
                     const float* d_b,
                     float* d_c,
                     int n);
