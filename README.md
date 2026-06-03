# CUDA Transformer Kernels

一个用于学习和验证 Transformer 常见算子的 CUDA/C++ 项目。仓库从最基础的 `vector add` 开始，逐步实现 reduction、row-wise softmax、RMSNorm 和 GEMM，并为每个 kernel 提供 CPU reference 校验程序，方便观察正确性、线程组织、shared memory 使用方式和不同实现之间的性能差异。

> 当前项目更偏向 CUDA kernel 学习、实验和 benchmark 骨架，不是完整的 Transformer 推理框架。

## Features

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| Vector Add | `kernels/vector_add.cu` | 最基础的一维 elementwise kernel，用于验证 CUDA 开发环境和 launch 配置。 |
| Reduction Sum | `kernels/reduction.cu` | 每个 block 使用 shared memory 规约出一个 partial sum，最终在 host 端合并。 |
| Reduction Max | `kernels/reduction.cu` | 每个 block 规约出一个 partial max，用于后续 softmax 等算子的基础练习。 |
| Softmax | `kernels/softmax.cu` | 对 row-major 矩阵逐行执行数值稳定 softmax，先减去行最大值再计算指数。 |
| RMSNorm | `kernels/rmsnorm.cu` | 对 row-major 矩阵逐行执行 RMSNorm，适合 Transformer block 中的归一化算子学习。 |
| GEMM Naive | `kernels/gemm.cu` | 直接矩阵乘法实现，每个线程计算一个输出元素。 |
| GEMM Tiled | `kernels/gemm.cu` | 使用 `16 x 16` shared-memory tile 的矩阵乘法实现。 |
| GEMM Compare | `benchmarks/benchmark_gemm_compare.cu` | 对 naive GEMM 和 tiled GEMM 做正确性校验与简单耗时对比。 |

## Project Structure

```text
.
|-- CMakeLists.txt
|-- README.md
|-- benchmarks/
|   |-- benchmark_vector_add.cu
|   |-- benchmark_reduction_sum.cu
|   |-- benchmark_reduction_max.cu
|   |-- benchmark_softmax.cu
|   |-- benchmark_rmsnorm.cu
|   |-- benchmark_gemm_naive.cu
|   |-- benchmark_gemm_tiled.cu
|   `-- benchmark_gemm_compare.cu
|-- include/
|   |-- cuda_utils.h
|   |-- benchmark_utils.h
|   |-- vector_add.h
|   |-- reduction.h
|   |-- softmax.h
|   |-- rmsnorm.h
|   `-- gemm.h
|-- kernels/
|   |-- vector_add.cu
|   |-- reduction.cu
|   |-- softmax.cu
|   |-- rmsnorm.cu
|   `-- gemm.cu
|-- docs/
`-- results/
```

## Requirements

- NVIDIA GPU
- CUDA Toolkit
- CMake `>= 3.18`
- C++17 compatible compiler

The default CUDA architecture in `CMakeLists.txt` is:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

This targets `sm_89`, which is suitable for Ada Lovelace GPUs such as RTX 40 series. If your GPU uses a different compute capability, update this value before building, for example:

| GPU generation | Common architecture value |
| --- | --- |
| Turing | `75` |
| Ampere | `80`, `86` |
| Ada Lovelace | `89` |
| Hopper | `90` |

## Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On Windows with a multi-config generator, the binaries may be placed under `build/Release/`. With Makefile or Ninja generators, they are usually emitted directly under `build/`.

## Run Correctness Checks and Benchmarks

Each executable initializes random inputs, computes a CPU reference result, launches the CUDA kernel, copies the output back to host memory, and prints a PASS/FAIL status with the maximum absolute error.

Linux, WSL, or Makefile-style build:

```bash
./build/benchmark_vector_add
./build/benchmark_reduction_sum
./build/benchmark_reduction_max
./build/benchmark_softmax
./build/benchmark_rmsnorm
./build/benchmark_gemm_naive
./build/benchmark_gemm_tiled
./build/benchmark_gemm_compare
```

Windows PowerShell with Release output:

```powershell
.\build\Release\benchmark_vector_add.exe
.\build\Release\benchmark_reduction_sum.exe
.\build\Release\benchmark_reduction_max.exe
.\build\Release\benchmark_softmax.exe
.\build\Release\benchmark_rmsnorm.exe
.\build\Release\benchmark_gemm_naive.exe
.\build\Release\benchmark_gemm_tiled.exe
.\build\Release\benchmark_gemm_compare.exe
```

Example output format:

```text
GEMM Benchmark Compare
Shape: M=512, N=512, K=512
Warmup: 10, Repeat: 50

Naive GEMM
  Time: ... ms
  Max abs error: ...
  Status: PASS

Tiled GEMM
  Time: ... ms
  Max abs error: ...
  Status: PASS

Speedup: ...x
```

## Implemented Kernel Notes

### Vector Add

- Uses a one-dimensional grid.
- Each thread handles one element.
- The last partial block is guarded by an `index < n` check.

### Reduction

- Uses dynamic shared memory via `extern __shared__`.
- Each block reduces its own slice into a partial result.
- The benchmark finishes the final aggregation on CPU, which keeps the kernel simple and makes the intermediate results easy to inspect.

### Softmax

- Each CUDA block handles one matrix row.
- The implementation first reduces the row maximum for numerical stability.
- It then computes `exp(x - max)`, reduces the row sum, and normalizes the output in place.

### RMSNorm

- Each CUDA block handles one matrix row.
- Threads stride over the hidden dimension.
- Shared memory is used to reduce the sum of squares before applying the per-channel weight.

### GEMM

- `gemm_naive_cuda` computes one `C[row, col]` element per thread by looping over `K`.
- `gemm_tiled_cuda` loads `A` and `B` into `16 x 16` shared-memory tiles to reduce global memory traffic.
- `benchmark_gemm_compare` validates both implementations against a CPU reference and reports the tiled speedup over the naive implementation.

## Development Notes

- `include/cuda_utils.h` provides `CUDA_CHECK` and `CUDA_KERNEL_CHECK` macros for CUDA runtime error checking.
- `include/benchmark_utils.h` provides a small `GpuTimer` wrapper based on CUDA events.
- All current benchmark inputs use deterministic random seeds, which makes correctness runs reproducible.
- Build artifacts are ignored through `.gitignore`; source code, docs, and result files are intended to be committed.

## Roadmap

Potential next steps:

- Add automated `ctest` targets for all correctness checks.
- Add CSV result export for benchmark timing.
- Add more optimized reductions using warp-level primitives.
- Compare custom GEMM against cuBLAS.
- Add LayerNorm, attention score softmax, and fused Transformer kernels.
- Add profiling notes from Nsight Compute or Nsight Systems.

## License

No license has been specified yet. Before publishing this repository publicly, consider adding a license such as MIT, Apache-2.0, or BSD-3-Clause.
