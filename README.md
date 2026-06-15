# High-Performance CUDA Kernel Optimization Portfolio

A performance-optimized portfolio of CUDA kernels, numerical linear algebra solvers, and
parallel computing primitives — engineered to explore GPU micro-architectural constraints
through profiling-guided iteration on an NVIDIA Tesla T4 (Turing, sm_75).

---

## Background & Motivation

Across 8 years engineering iterative numerical solvers (CG, LU, domain decomposition) and 4 years optimizing high-throughput concurrent systems — including lock-free producer-consumer pipelines for multi-billion row datasets and NUMA-aware memory layout tuning — I developed a deep intuition for hardware-aware algorithm design.

This self-directed CUDA project extends that foundation to GPU-native implementations,
profiled end-to-end with Nsight Compute. Each module targets a specific GPU
micro-architectural constraint: memory coalescing, shared memory bank conflicts, warp
divergence, occupancy ceilings, or L2 cache thrashing. The goal is not to replicate
tutorials — it is to develop the same instincts on GPU hardware that 12 years of CPU
systems work built on the CPU side.

**Target role focus:** Math library engineering (cuBLAS / cuSPARSE / cuSOLVER equivalent
workflows), HPC kernel optimization, and GPU-native numerical methods.

---

## Key Concepts Demonstrated

`shared memory tiling` · `bank conflict analysis & padding` · `warp-level primitives`
(`__shfl_down_sync`) · `Nsight Compute profiling (ncu)` · `occupancy optimization` ·
`memory coalescing` · `L1/L2 cache hierarchy` · `batched LU factorization` ·
`Krylov iterative solvers (CG)` · `FlashAttention block tiling` · `BLAS-level GEMM` ·
`CSR SpMV` · `Turing architecture (sm_75)` · `Google Colab T4 runtime`

---

## Performance Highlights (NVIDIA Tesla T4, Compute Capability 7.5)

All numbers sourced directly from Nsight Compute (`ncu`) benchmark reports in each subfolder.

| Kernel / Module | Baseline | Optimized | Speedup | Key Technique |
|---|---|---|---|---|
| Matrix Transpose | 0.2286 ms (naive) | 0.0737 ms (tiled + padded) | **~3.1×** | Shared mem staging + bank-conflict padding (32×33) |
| 2D Poisson Solver (SpMV vs Tiling) | CSR scalar (~8.1% L2 hit) | 32×32 spatial tiling | **~4×** | Eliminated L2 thrash via SMEM staging |
| Register-Tiled GEMM | — (baseline) | 1,528,051 cycles (−6.5%) | — | Stride offset fix, bank conflict resolution |
| Prefix Scan | Shared memory tree | Warp shuffle (`__shfl_down_sync`) | — | Zero `__syncthreads()`, register-only exchange |

> **Environment note:** All experiments run on Google Colab T4 runtime (4 MB L2, 40 MB
> L1/SRAM). Some multi-GPU or high-memory experiments are constrained by this environment.

---

## Repository Structure

```
cuda-kernel-acceleration/
├── blas-primitives/          # GEMM (naive → tiled → register), Matrix Transpose
├── parallel-primitives/      # Prefix Sum: Blelloch, Brent-Kung, warp shuffle
├── sparse-linear-algebra/    # CSR SpMV, structured stencil variants
├── numerical-solvers/        # Batched LU with partial pivoting, backward substitution
├── krylov-methods/           # 2D Conjugate Gradient for Poisson equations
├── dl-acceleration/          # Online Softmax, FlashAttention block tiling
└── asynchronous-streams/     # CUDA stream overlap experiments
```

---

## Micro-Architectural Profiling Scorecard

### 1. Register-Tiled GEMM with Shared Memory Padding

**Target:** Resolve shared memory bank serialization and minimize memory pipeline latency stalls.

**Nsight Compute Results:**
- Elapsed Duration: **1,528,051 cycles** (~6.5% reduction from stride offset fix alone)
- L1/TEX Cache Throughput: **98.91%**
- DRAM Throughput: **2.40%** (compute-bound, not memory-bound — expected for tiled GEMM)
- Compute (SM) Throughput: **40.10%**

**Bottleneck Analysis:**
- The profiler isolated a **2.0–2.1× bank conflict** rate across shared store operations,
  affecting ~52.99% of all shared store wavefronts. Root cause: column-step indexing
  patterns (`tile_A[ty * 4 + i][k_inst]`) mapping threads to the same physical banks.
- With a 4×4 block configuration, theoretical occupancy was anchored at **50%**
  (16 active threads/warp), leaving half the warp execution lanes unutilized.
- FMA pipeline occupied 24.5% of active cycles — healthy FP throughput with no
  execution stalls on integer or transcendental pipelines.

**CPU Parallel:** On CPU (Intel MKL), GEMM latency hides behind deep out-of-order
execution and large L3 caches. On T4, the 4 MB L2 ceiling and lack of speculative
prefetch make explicit tiling and padding non-negotiable — not a micro-optimization
but a correctness constraint for throughput.

---

### 2. Tiled Matrix Transpose

**Target:** Eliminate uncoalesced global memory writes by introducing a shared memory
staging pivot table.

| Configuration | Time | Speedup | Mechanism |
|---|---|---|---|
| Naive (baseline) | 0.2286 ms | 1.0× | Strided global writes, $N$-element column stride |
| Shared Memory Tiled (32×32) | 0.1207 ms | ~1.9× | Coalesced reads/writes via `__syncthreads()` barrier |
| Tiled + Bank Padded (32×33) | **0.0737 ms** | **~3.1×** | Bank conflict elimination via row-stride padding |

**CPU Parallel:** CPU transpose benefits from hardware prefetchers and large per-core
caches tolerating strided access. GPU warp execution is strictly sequential within a
wavefront on non-coalesced addresses — strided column writes serialize to 32 independent
memory transactions, which is why the 3.1× gap between naive and padded is so large.

---

### 3. Sparse CSR vs. 2D Spatial Tiling (2D Poisson Field Solver)

**Target:** Minimize L2 cache thrashing and quantify the indirect memory access penalty
on structured grids.

**The Structural Problem:** A 512×512 Poisson grid via 5-point CSR stencil requires a
~17 MB working set per iteration (11 MB sparse matrix trio + 6 MB workspace vectors).
The T4's 4 MB L2 is continuously evicted, forcing repeated high-latency DRAM traffic.

**CSR Baseline (Nsight Compute):**
- Global Memory Bandwidth: **132 GB/s**
- L2 Cache Hit Rate: **8.1%** — effectively uncached
- L1/TEX Cache Hit Rate: **49.96%** — catches immediate spatial neighbors, not inter-iteration reuse

**Architectural Fix — 2D Spatial Tiling:**
- Replaced CSR indirect indexing (`values[i] * p[col_indices[i]]`) with 32×32 spatial
  blocks mapped to thread blocks in shared memory.
- Data loaded once from DRAM, reused across all 256 threads in the block.
- **Result: ~4× throughput speedup** over scalar CSR baseline.

**CPU Parallel:** This is the GPU equivalent of the CPU lesson that MKL SpMV on structured
grids is often outperformed by stencil-aware cache-blocking. The principle transfers
directly — the difference is that the penalty for ignoring it on GPU is proportionally
larger due to the narrower cache hierarchy.

---

## Micro-Architectural Case Studies

### Bank Conflict Elimination via Structural Padding

During matrix transpose and GEMM kernels, warp threads writing column-wise to SRAM
frequently mapped to identical hardware banks, causing 2-way serialization.

**Solution:** Alter array stride dimensions (`[TILE_WIDTH][TILE_WIDTH + 1]` or `+ 2`
for vectorized FP configurations). This shifts index addresses so sequential row elements
distribute evenly across separate physical banks — from serialized to parallel LSU requests.

### Warp-Level Register Primitives for Prefix Scans

Standard shared memory prefix scans require two `__syncthreads()` barriers per level —
one after reduction, one after fan-out. This is avoidable inside a single warp.

**Solution:** `__shfl_down_sync` exchanges values directly in the register file, dropping
shared memory bank checks, array index pointer arithmetic, and explicit sync points
entirely. Particularly effective for reductions that fit within a 32-thread warp.

### Synchronization Boundaries in Batched LU Solvers

Dense batched direct solvers require strict thread barrier management to preserve data
integrity during parallel partial pivoting.

**Solution:** Three explicit execution phases separated by `__syncthreads()` barriers:
(1) cooperative max-element search below the diagonal, (2) shared tile row-swap,
(3) global permutation tracker update — all prior to computing rank-1 Schur complement
tail updates. Race-free pivoting at block level without global memory round-trips.

---

## Compilation

All modules are CUDA C++ (`.cu`), compiled with standard optimization flags:

```bash
# Example: numerical solver
cd numerical-solvers
nvcc -O3 -arch=sm_75 LUFactorization.cu -o lu_solver
./lu_solver

# Profile with Nsight Compute
ncu --set full ./lu_solver
```

---

## Roadmap

Work in progress — constrained by Colab T4 runtime, but scoped to what's learnable
within those limits:

- [ ] cuSPARSE API comparison benchmarks for SpMV formats (CSR vs BSR vs ELL)
- [ ] FP16/BF16 GEMM variants — explore whether tensor core access is achievable on T4
- [ ] Multi-stream overlap across solver iterations (asynchronous-streams module)
- [ ] Hopper architecture (sm_90) review: TMA, warpgroup MMA — architectural study
      even if not runnable on Colab
- [ ] Structured sparsity experiments for transformer inference acceleration

---

## Environment

| Component | Details |
|---|---|
| GPU | NVIDIA Tesla T4 (Turing, sm_75) |
| Runtime | Google Colab T4 instance |
| L2 Cache | 4 MB |
| CUDA Version | 12.x |
| Profiler | Nsight Compute (`ncu`) |
| Language | CUDA C++ (`.cu`) |
| Compiler | `nvcc -O3 -arch=sm_75` |
