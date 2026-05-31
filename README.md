# High-Performance CUDA Kernel Optimization Portfolio

A comprehensive library of CUDA kernels, numerical linear algebra solvers, and parallel computing primitives engineered to explore low-level GPU micro-architectural constraints. This repository documents a self-driven, profiling-guided study focusing on maximizing hardware compute utilization, mitigating memory hierarchy latency, eliminating shared memory bank conflicts, and managing execution synchronization.

**Hardware & Profiling Environment:** All source modules were compiled, executed, and analyzed within a cloud-hosted environment utilizing an **NVIDIA Tesla T4 GPU (Turing Architecture, Compute Capability 7.5)**. Performance data, cycle counts, and micro-architectural diagnostics listed below were extracted directly from execution profiles gathered via the **NVIDIA Nsight Compute (`ncu`)** toolkit.

---

## 🏛️ Repository Architectural Taxonomy

The repository is organized into distinct functional domains mapped directly to mathematical and hardware execution paradigms:

* **`blas-primitives/`**
    * **Matrix Multiplication (GEMM):** Implementations tracking the progressive transition from naive global memory layouts to block-level shared memory tiling, boundary guarding, and register tiling configurations.
    * **Matrix Transpose:** A diagnostic kernel used to analyze uncoalesced global memory write paths and evaluate the performance impact of shared memory staging.
* **`parallel-primitives/`**
    * **Prefix Sum (Scans):** Parallel prefix sum configurations mapping work-efficient tree-reduction architectures, including hierarchical multi-block scans, Brent-Kung padded layouts, Blelloch layouts, and register-level warp-shuffle (`__shfl_down_sync`) primitives.
* **`sparse-linear-algebra/`**
    * **Sparse Matrix-Vector Multiplication (SpMV):** Implementations evaluating sparse storage formats, focusing on Compressed Sparse Row (CSR) vector multiplication layouts to analyze memory coalescence and cache residency limitations across sparse stencils.
* **`numerical-solvers/`**
    * **Batched Direct Solvers:** Monolithic, block-level factorization systems featuring dense LU Decomposition with parallelized partial pivoting, shared memory row-swapping, and Schur complement rank-1 updates.
    * **Blocked Cholesky Factorization:** High-performance dense $A = LL^T$ decomposition. Features a cooperative, shared-memory tiled trailing submatrix update kernel optimized for implicit transposition layouts that bypasses global memory bandwidth constraints to achieve **~79.25% SM Compute Throughput**. Detailed Nsight hardware metrics analysis is archived in [`cholesky_profiling_report.md`](./numerical-solvers/cholesky_profiling_report.md).
    * **Linear Systems:** Basic utilities including backward substitution and lower triangle evaluations.
* **`krylov-methods/`**
    * **2D Conjugate Gradient (CG) Solver:** An iterative structured field solver optimized for 2D Poisson equations, demonstrating how vector operation fusion minimizes global memory round-trips.
* **`dl-acceleration/`**
    * **Deep Learning Operators:** Implementations focusing on foundational transformer acceleration blocks, including multi-pass Online Softmax reduction algorithms and fused FlashAttention block-tiling loops.

---

## 📊 Micro-Architectural Profiling Scorecard (NVIDIA T4 Baseline)

### 1. Register-Tiled Matrix Multiplication with Shared Memory Padding
* **Hardware Diagnostic Target:** Minimizing memory pipeline latency stalls and resolving shared memory bank serialization.
* **Nsight Compute Diagnostics:**
    * **Elapsed Duration:** 1,528,051 execution cycles (a ~100,000 cycle reduction / 6.5% speedup realized immediately upon shifting stride offsets).
    * **L1/TEX Cache Throughput:** 98.91%.
    * **DRAM Throughput:** 2.40%.
    * **Overall Memory Throughput:** 49.42%.
    * **Compute (SM) Throughput:** 40.10%.
* **Micro-Architectural Bottleneck Breakdown & Analysis:**
    * **The Pipeline Conflict:** The profiler isolated a localized bottleneck within the Load/Store Unit (LSU) pipeline, detecting an average **2.0 to 2.1-way bank conflict** across shared store operations, affecting up to **52.99% of all shared store wavefronts**. This was driven by thread allocation indexing patterns matching column steps (`tile_A[ty * 4 + i][k_inst]`).
    * **The Occupancy Floor:** Due to utilizing a tighter block footprint (4x4 execution configuration), the hardware was constrained to an **Average Active Threads Per Warp of 16.00**, anchoring the kernel to exactly **50% Theoretical Occupancy** and leaving exactly half of the available warp execution lanes unutilized during instruction issue slots.
    * **Pipeline Utilization:** Instruction analysis tracked the **Fused Multiply-Add (FMA)** pipeline as the primary execution engine, occupying **24.5% of active cycles**, ensuring robust floating-point throughput without hitting execution stalls on integer or transcendental pipelines.

### 2. Tiled Matrix Transpose Benchmarks
* **Hardware Diagnostic Target:** Eliminating uncoalesced global memory writes by establishing a shared memory staging pivot table.
* **Execution Profiling Breakdown:**
    * **Naive Baseline:** 0.228608 ms (1.0x Baseline) — Limited by uncoalesced, strided global memory writes across column indices separated by $N$ elements.
    * **Shared Memory Tiled (32x32):** 0.120672 ms (~1.9x speedup) — Restructures global access into coalesced horizontal reads and writes by utilizing `__syncthreads()` block memory barriers.
    * **Tiled + Bank Padded (32x33):** 0.073728 ms (~3.1x speedup) — Resolves internal shared memory bank conflicts by altering row-stride mapping parameters.

### 3. Sparse CSR vs. 2D Shared Memory Tiling (2D Poisson Field Solver)
* **Hardware Diagnostic Target:** Minimizing L2 cache thrashing and evaluating indirect memory access penalties on structured grids.
* **The Structural Problem:** Solving a 2D Poisson system on a $512 \times 512$ grid using a 5-point stencil via standard Compressed Sparse Row (CSR) format dictates a working memory set size of **~17.0 MB** per iteration (11 MB for the sparse matrix trio and 6 MB for the workspace vectors). 
* **Nsight Compute Diagnostics (CSR Baseline):**
    * **Achieved Global Memory Bandwidth:** 132 GB/s.
    * **L2 Cache Hit Rate:** 8.1%.
    * **L1/Texture Cache Hit Rate:** 49.96%.
    * **Micro-Architectural Bottleneck:** **L2 Cache Thrashing.** Because the 17.0 MB working set significantly exceeds the physical 4.0 MB L2 cache capacity of the target NVIDIA Tesla T4 hardware, the cache is continuously evicted, forcing repeated high-latency transactions to DRAM. While the L1/Texture cache caught immediate spatial thread neighbors (yielding ~50% hit rate), it lacked the capacity to maintain data reuse across global iterations.
* **The Architectural Shift (2D Spatial Tiling):**
    * Developed an optimized alternative layout bypassing the indirect indexing (`values[i] * p[col_indices[i]]`) of the CSR format. Instead, the grid was restructured into localized $32 \times 32$ spatial blocks mapped directly to thread blocks utilizing **Shared Memory Tiling**.
    * **Performance Gains:** Achieved a **~4× throughput speedup** over the scalar CSR baseline. 
    * **Micro-Architectural Driver:** By staging the spatial stencil tiles explicitly inside the fast L1/SRAM memory layer, data elements are loaded once from DRAM and reused concurrently by all 256 threads in the block. This drastically reduces the pressure on the GPU's memory controllers, simplifies coordinate tracking math for the hardware instruction schedulers, and optimizes the efficiency of the hardware prefetcher.

---

## 🛠️ Micro-Architectural Case Studies

### 1. Eliminating Shared Memory Bank Serialization via Structural Padding
During matrix transpose and register-tiled configurations, concurrent threads within a warp writing column-wise data to high-speed SRAM frequently mapped to identical hardware memory banks, resulting in 2-way serialization barriers.
* **The Solution:** Applied structural padding allocations by altering array stride dimensions (e.g., changing allocation boundaries from `[TILE_WIDTH][TILE_WIDTH]` to `[TILE_WIDTH][TILE_WIDTH + 1]` or `+ 2` for vectorized floating-point configurations). This padding dynamically shifts the index addresses, forcing sequential row elements to distribute evenly across separate physical memory banks, allowing the LSU pipeline to service memory requests in parallel.

### 2. Warp-Level Register Primitives for Parallel Scans
To maximize throughput across parallel prefix sums and bypass block-level synchronization overhead, implementations bypass standard shared memory staging arrays entirely where applicable.
* **The Solution:** Utilizing `__shfl_down_sync` warp-shuffle primitives allows threads to communicate and exchange data values directly within the processor register file. This drops instruction issue cycles by eliminating shared memory bank checks, array index pointer calculations, and explicit block-wide sync points (`__syncthreads()`).

### 3. Row Permutation Synchronization Boundaries in Batched LU Solvers
Implementing block-level dense direct solvers requires rigid management of thread execution barriers to maintain data integrity during parallel pivoting updates.
* **The Solution:** The kernel enforces distinct execution phases isolated by explicit synchronization barriers (`__syncthreads()`). Threads cooperatively calculate the absolute maximum element below the active diagonal, exchange rows within the shared tile footprint, and update global permutation trackers concurrently without introducing race conditions prior to computing the rank-1 Schur complement tail updates.

---

## 🚀 Compilation and Verification

### Compilation Layout via NVIDIA CUDA Compiler (`nvcc`)
All source code modules are written in native CUDA C++ (`.cu`) and compiled using standard library optimization levels:

```bash
# Example compilation layout for subfolders
cd numerical-solvers
nvcc -O3 -arch=sm_75 LUFactorization.cu -o lu_solver
./lu_solver