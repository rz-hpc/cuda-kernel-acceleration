# High-Performance CUDA Kernel Optimization & Distributed Systems Portfolio

A performance-optimized portfolio of CUDA kernels, numerical linear algebra solvers, parallel computing primitives, and distributed multi-GPU systems. This repository is engineered to explore GPU micro-architectural constraints and interconnect topologies through profiling-guided iteration.

---

## Background & Motivation

Across 8 years engineering iterative numerical solvers (CG, LU, domain decomposition) at Siemens EDA and 4 years optimizing high-throughput concurrent systems at Microsoft — including NUMA-aware memory layout tuning — I developed a deep intuition for hardware-aware algorithm design.

This self-directed CUDA project extends that foundation to GPU-native implementations and multi-node distributed setups, profiled end-to-end with NVIDIA Nsight Compute (`ncu`) and Nsight Systems (`nsys`). Each module targets a specific hardware constraint: memory coalescing, shared memory bank conflicts, warp divergence, L2 cache thrashing, or PCIe/MPI interconnect contention.

**Target focus:** Math library engineering (cuBLAS / cuSPARSE / cuSOLVER / cuBLASMp equivalent workflows), HPC kernel optimization, and GPU-native numerical methods focused on distributed systems.

---

## Hardware & Profiling Environments

This repository isolates different performance bounds by utilizing distinct hardware environments tailored to specific profiling goals:

1. **Single-GPU Micro-Architectural Profiling (NVIDIA Tesla T4, sm_75):** Used for all dense/sparse kernel optimizations, focusing on L1/L2 cache behavior, shared memory bank conflicts, and warp execution efficiency. Executed on Google Colab T4 instances (4 MB L2, 40 MB L1/SRAM).
2. **Multi-GPU Distributed Profiling (4× NVIDIA RTX PRO 4500, sm_89):** Dedicated exclusively to the distributed SUMMA vs. `cuBLASMp` benchmarking. This environment isolated host-routed PCIe (`NODE`/`PHB`) interconnect bottlenecks during MPI and NCCL stream overlap studies.

---

## Key Concepts Demonstrated

`distributed SUMMA GEMM` · `MPI + NCCL communicators` · `CUDA stream double-buffering` · `shared memory tiling` · `bank conflict analysis & padding` · `warp-level primitives (__shfl_down_sync)` · `Nsight Compute (ncu) & Nsight Systems (nsys)` · `occupancy optimization` · `memory coalescing` · `L1/L2 cache hierarchy` · `batched LU factorization` · `blocked Cholesky` · `Tall-Skinny QR (TSQR, Householder)` · `Krylov iterative solvers (CG)` · `FlashAttention block tiling` · `BLAS-level GEMM` · `CSR SpMV` · `Schur complement rank-1 update`.

---

## Repository Structure

```text
cuda-kernel-acceleration/
├── distributed-computing/    # Multi-GPU MPI + NCCL SUMMA GEMM vs. cuBLASMp
├── blas-primitives/          # GEMM (naive → tiled → register), Matrix Transpose
├── parallel-primitives/      # Prefix Sum: Blelloch, Brent-Kung, warp shuffle
├── sparse-linear-algebra/    # CSR SpMV, structured stencil variants
├── numerical-solvers/        # Batched LU (partial pivoting), Cholesky (blocked), TSQR
├── krylov-methods/           # 2D Conjugate Gradient for Poisson equations
├── dl-acceleration/          # Online Softmax, FlashAttention block tiling
├── asynchronous-streams/     # CUDA stream overlap experiments
└── benchmarks/               # Nsight Compute ncu reports and profiling summaries

```

---

## Featured Distributed Module: Multi-GPU SUMMA GEMM vs. cuBLASMp

Located in `distributed-computing/nccl-multi-gpu/`, this flagship module isolates interconnect latency from compute throughput by evaluating a custom Scalable Universal Matrix Multiplication Algorithm (SUMMA) against NVIDIA `cuBLASMp` on a 4-GPU workstation topology.

### Architectural Highlights

* **Process Grid Topology:** Dynamic P × Q MPI process grid with 2D block-cyclic local-to-global index mapping across physical GPUs.
* **Double-Buffered Overlap Engine:** Dual CUDA streams (`stream_compute` vs. `stream_comm`) utilize CUDA event synchronization to hide non-blocking `ncclBcast` transfers behind partial SGEMM tile computations.

### Comparative Profiling (`-np 4`, RTX PRO 4500)

| Rank | Implementation | Compute Time (ms) | MPI/Sync Overhead (ms) | Total Transfer (MB) | Execution Bottleneck |
| --- | --- | --- | --- | --- | --- |
| **0** | **cuBLASMp** | **7.31** | **1,604.74** | 263.45 | PCIe Host Bridge Bandwidth |
| **0** | **Custom SUMMA** | **26.66** | **774.66** | 263.35 | PCIe Host Bridge Bandwidth |
| **1** | **cuBLASMp** | **3.75** | **7,043.84** | 75.71 | Worker Spin-Wait Skew |
| **1** | **Custom SUMMA** | **4.08** | **6,392.87** | 75.60 | Worker Spin-Wait Skew |

**Key Insights:** Vendor-tuned `cuBLASMp` micro-kernels achieve a 2.6×–3.6× compute speedup over the custom tile GEMM on localized execution. However, on a topology relying entirely on host-routed PCIe bridges without direct NVLink fabrics, broadcast latency exceeds compute time by orders of magnitude. The double-buffered compute streams stall on network synchronization, forcing both custom SUMMA and `cuBLASMp` to hit the identical physical bandwidth floor imposed by the hardware.

---

## Single-GPU Micro-Architectural Case Studies (NVIDIA Tesla T4)

All numbers sourced directly from Nsight Compute (`ncu`) benchmark reports.

### 1. SGEMM: Naive vs. Tiled vs. Register-Tiled

* **Target:** Evaluate shared memory tiling and instruction-level parallelism on deliberately unaligned matrix dimensions (M=1001, K=503, N=1001) using a 63×63 grid.


* **Naive (16×16 blocks):** 3.5 ms latency.


* **Shared Memory Tiled (16×16 blocks):** 3.7 ms. **Analysis:** Tiling resulted in a ~6% performance regression because the L2 cache was already achieving a 99% hit rate; adding shared memory synchronization overhead penalized a bandwidth problem that did not exist at this problem size.


* **Register-Tiled (8×4 blocks, 4×2 output patch per thread):** 1.76 ms. **Analysis:** Achieved a ~2× speedup over the baseline. Despite shared memory bank conflicts remaining at ~50% of shared-store wavefronts, performance doubled because the total instruction count was reduced from 75M to 35M, amortizing memory operations across more work per thread. Unaligned dimensions forced vectorized float4 write-backs to successfully utilize scalar fallback paths at boundary guards.



### 2. Blocked Cholesky Factorization: Trailing Matrix Update

* **Target:** Resolve memory-bound bottlenecks in trailing submatrix updates for a 2048×2048 matrix using 32×32 thread blocks and 64 outer iterations.


* **Optimization:** Staged the pivot row and column directly into shared memory.


* **Results:** Execution time dropped from 5.8 ms to ~150 μs, representing a ~38× speedup. Instructions Per Cycle (IPC) scaled from 0.07 to ~0.9 (~13× improvement).


* **Bottleneck Shift:** Execution stalls successfully transitioned from a 97% global-memory dependency to a 68% shared-memory MIO queue bottleneck—proving the workload shifted from memory-bound to compute-bound.



### 3. Communication-Avoiding TSQR: Panel vs. Merge Kernels

* **Target:** Optimize Householder reflections for a tall-skinny matrix (M=512, N=4) distributed across 4 leaf panels of 128 rows each, utilizing a 3-level reduction tree.


* **Panel Kernel:** Implementing row-partitioned warp-shuffle reductions decreased latency from 67 μs to 30 μs (~2.2× speedup), while active threads per warp improved from 9 to 17.


* **Merge Kernel:** Execution time remained flat (17.06 μs to 17.38 μs) despite active threads per warp improving from 9 to 12. **Analysis:** The merge system size (8×4) is too small to yield performance benefits from 32-way warp cooperation, isolating the minimum threshold required for warp-shuffle efficacy.



### 4. Sparse CSR vs. 2D Spatial Tiling (2D Poisson Field Solver)

* **Target:** Minimize L2 cache thrashing and quantify the indirect memory access penalty on structured grids.
* **CSR Baseline:** 132 GB/s global memory bandwidth, but an 8.1% L2 cache hit rate (effectively uncached due to the T4's 4 MB L2 ceiling).
* **Architectural Fix:** Replaced CSR indirect indexing with 32×32 spatial blocks mapped to thread blocks in shared memory, yielding a **~4× throughput speedup** over the scalar CSR baseline by reusing data directly from SRAM.

---

## Compilation & Profiling Toolchain

### Single-GPU Profiling (Tesla T4)

Used for micro-architectural metric extraction.

```bash
# Example: numerical solver compilation
cd numerical-solvers
nvcc -O3 -arch=sm_75 LUFactorization.cu -o lu_solver

# Profile with Nsight Compute
ncu --set full ./lu_solver

```

### Multi-GPU Distributed Profiling (RTX PRO 4500)

Used for MPI process tracing, NCCL bandwidth limits, and stream concurrency evaluation.

```bash
# Compile Distributed SUMMA Kernel
cd distributed-computing/nccl-multi-gpu
nvcc -O3 -arch=native summa_gemm.cu -o summa_gemm -lcublas -lnccl -ccbin mpicxx

# Capture trace across all MPI processes with Nsight Systems
mpirun --allow-run-as-root -np 4 nsys profile \
  --trace=cuda,nvtx,mpi \
  -o trace_rank%q{OMPI_COMM_WORLD_RANK} \
  ./summa_gemm

```

---

## Roadmap

Work in progress — Balancing applied optimizations on accessible hardware with theoretical architectural studies for next-generation systems.

### Distributed Systems & Communication Overlap

* [ ] **Multi-Node & RDMA Architectural Study:** Theoretical extension of the current single-node SUMMA engine to multi-node MPI clusters, focusing on InfiniBand interconnect modeling and NCCL SHARP in-network reductions without immediate access to physical clusters.
* [ ] **Stream Priority & Pipeline Scheduling:** Refine asynchronous compute/comm overlap by leveraging CUDA stream priorities to prevent compute kernels from stalling critical-path NCCL broadcasts.

### Micro-Architecture & Kernel Optimization

* [ ] **cuDSS Refactorization Demo:** Implement repeated solves on matrices sharing a sparsity pattern (analyze-once / factorize-once / solve-many workflow for FEM and circuit simulators).
* [ ] **Mixed-Precision Tensor Core (WMMA) GEMM:** Optimize FP16/BF16 dense matrix multiplications via hardware tensor core instructions on accessible architectures (Turing sm_75).
* [ ] **Hopper (sm_90) Feature Exploration:** Theoretical review and literature study of Tensor Memory Accelerator (TMA) asynchronous block transfers and warpgroup-level MMA execution models.