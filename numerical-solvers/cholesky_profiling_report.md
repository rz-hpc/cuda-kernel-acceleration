# Blocked Cholesky — Profiling Report

## 1. Overview

Two separate profiling studies on different problem sizes and optimization axes.
Both use the same 3-kernel blocked Cholesky structure:
- Kernel 1: `diagnoal_factorization_kernel` — diagonal tile factorization
- Kernel 2: `column_update_kernel` — column panel update (TRSM-equivalent)
- Kernel 3: `trailing_submatrix_kernel` — rank-k trailing matrix update (SYRK-equivalent)

---

## 2. Study A: Trailing Submatrix Kernel Optimization (N=2048)

Compares naive global-memory access vs. shared-memory tiling on Kernel 3 only.
Kernels 1 and 2 are dependency-chain-bound at <1% throughput regardless of optimization.

### 2.1 Key Performance Metrics — Kernel 3 (Trailing Submatrix)

| Metric | Naive (`cholesky_native`) | Tiled (`cholesky_tiled`) | Delta |
|---|---|---|---|
| FP32 Roofline Achievement | ~1.0% | ~8.0%–9.0% | 8–9× improvement |
| SM Active Cycles | 3,357,342.88 | 80,183.40 | **41.8× reduction** |
| Duration (First/Heaviest Wave) | 5.82 ms | 151.74 µs | **38.3× speedup** |
| Compute (SM) Throughput | 6.43% | 79.25% | **12.3× increase** |
| Memory Throughput | 49.44% | 79.25% | Balanced execution achieved |
| DRAM Throughput | 1.18% | 11.39% | Low % reflects active cache hits |
| L1/TEX Cache Throughput | 98.88% | 87.86% | Tiled uses shared memory, reducing L1 pressure |
| IPC Active | 0.07 | ~0.92 | **13.1× improvement** |
| Warp Cycles / Issued Instruction | 486.28 | 33.44 | **14.5× faster** |
| Warp Cycles / Executed Instruction | 486.64 | 33.52 | Eliminates long scoreboards from VRAM reads |

### 2.2 Bottleneck Diagnosis

| | Naive | Tiled |
|---|---|---|
| Dominant stall | L1/LG memory latency (97.5% of stall cycles, 473.6 of 485.7 avg cycles/warp) | MIO shared memory queue (67.9% of stall cycles) |
| Interpretation | Kernel is stalling waiting on global memory — every access goes to DRAM | Bottleneck correctly shifted to shared memory pipeline — the expected regime for a tiled kernel |

### 2.3 Kernels 1 and 2 — Dependency-Chain-Bound (Unchanged by Optimization)

Both the diagonal factorization and column update kernels remain <1% of peak
throughput regardless of the tiling optimization applied to Kernel 3. This is
an inherent algorithmic constraint: both kernels operate on a sequential inner
loop gated by `__syncthreads()` at every step — adding more parallelism or
better memory access patterns does not help because the bottleneck is the
data dependency chain, not memory bandwidth or occupancy.

---

## 3. Study B: Custom Implementation vs. cuSolver `potrf` (N=1024)

### 3.1 End-to-End Timing

| Implementation | Total kernel time | Correctness |
|---|---|---|
| Custom (3-kernel) | 4.94 ms | max rel error 0.0 (tol 1e-4) ✅ |
| cuSolver `potrf` | 3.07 ms | max rel error 0.0 (tol 1e-4) ✅ |
| Gap | ~1.6× | — |

### 3.2 Per-Kernel Breakdown (Custom, from Nsight Compute CSV)

| Kernel | Instances | Duration sum (µs) | FFMA total | DRAM total (MB) | Regs | Block Lim Shared |
|---|---|---|---|---|---|---|
| `diagnoal_factorization_kernel` | 64 | 56.4 | 512,000 | 19.8 | 25 | 7 blocks/SM |
| `column_update_kernel` | 63 | 69.2 | 671,744 | 226.7 | 20 | 3 blocks/SM |
| `trailing_submatrix_kernel` | 63 | 175.2 | 22,211,243 | 98.1 | 29 | 3 blocks/SM |

### 3.3 Gap Analysis

cuSolver's `potrf` uses a **recursive hybrid strategy**:
- Small blocks → hand-tuned leaf kernels (`potrf_alg2_cta_upper`) — these exhibit
  the **identical dependency-chain signature** as the custom diagonal/column kernels
  (confirmed by Nsight Compute trace: <1% throughput, flat duration regardless of block position)
- Large sub-blocks → delegated to standard cuBLAS GEMM/TRSM
  (`volta_sgemm_128x64` visible in the cuSolver kernel trace)

The 1.6× gap is an **algorithmic structure difference** (recursive hybrid with
library reuse at scale vs. single-level blocking), not a kernel tuning deficiency.

---

## 4. Key Takeaways

1. The trailing submatrix kernel responds dramatically to shared-memory tiling
   (38.3× duration speedup, 12.3× SM throughput increase) — it was genuinely
   memory-bandwidth-limited in the naive form.
2. Diagonal and column kernels are dependency-chain-bound and do not benefit from
   the same optimization — they require algorithmic restructuring (e.g. warp-shuffle-
   based reduction, fewer sync barriers), not tiling.
3. The gap to cuSolver is explained by its recursive hybrid strategy, not by the
   trailing submatrix kernel's tuning quality.
4. Nsight Compute's stall analysis (L1/LG → MIO shift) confirmed the optimization
   addressed the right bottleneck.
