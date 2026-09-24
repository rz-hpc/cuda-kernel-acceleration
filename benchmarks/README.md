# Benchmarks — Roofline Analysis & Nsight Compute Reports

Nsight Compute profiling reports, roofline analysis scripts, and extracted
summary CSVs for all kernels in this repo. Every number in this folder is
directly traceable to a specific `ncu` profiling run.

---

## Hardware

**NVIDIA Tesla T4 (Turing, sm_75)** — all single-GPU kernel profiles.
- Peak fp32 throughput: **8,141 GFLOPs/s**
- Peak DRAM bandwidth: **320 GB/s**
- Roofline ridge point: **25.44 FLOP/Byte**
- L2 cache: 4 MB
- Shared memory per SM: 64 KB (48 KB static cap without dynamic allocation)
- Max resident warps per SM: 32
- Max resident threads per SM: 1,024 (Turing, lower than Volta's 2,048)

---

## Files

| File | Description |
|---|---|
| `generate_roofline.py` | Plots the roofline chart from extracted FLOP and DRAM data |
| `get_flop.py` | Computes arithmetic intensity (FLOP/Byte) from Nsight CSV exports |
| `ExtractNsightReportMetrics.py` | Converts full ncu CSV reports (MB-scale) to summary CSVs (KB-scale) for committing to the repo. Handles both ncu raw-metric-dump (Format A) and tabular export (Format B) automatically. |
| `roofline_analysis.png` | Roofline chart for the full kernel suite |
| `*_report.csv` | Full Nsight Compute exports — large files, not committed |
| `*_summary.csv` | Key metrics only, extracted by ExtractNsightReportMetrics.py — committed |

---

## Data Provenance Notes

**AI computation:** `get_flop.py` computes arithmetic intensity as
`(fadd + fmul + 2×ffma) / dram__bytes.sum`. This uses raw DRAM traffic only —
not the full L1/L2/DRAM hierarchy. For kernels with high cache reuse (GEMM),
this underestimates the true operational intensity since most data is served
from L1/L2. For memory-bound kernels (SpMV, CG, Cholesky), DRAM traffic
dominates and the metric is accurate.

**Hardcoded fallbacks:** For GEMM kernels where the CSV parse returned no
usable DRAM data, `generate_roofline.py` uses hardcoded constants
(`AI=80.0 FLOP/B`, `gflops=1.67` for Native MatMul and `8.63` for Tiled).
The `%opt` values for those two kernels derive from these constants. All other
kernels use directly parsed values.

**cuSPARSE and cuDSS FFMA counts:** These are 0 in the CSV exports because
cuSPARSE's sparse multiply and cuDSS's factorization/solve kernels use
non-FMA integer indexing before the actual arithmetic — the FFMA metric
does not capture their true compute. Duration and DRAM bytes are accurate.

---

## Per-Kernel Summary (from `*_summary.csv` files)

All durations are `gpu__time_duration.sum` in microseconds.
DRAM traffic is `dram__bytes.sum` in MB.
FFMA count is `sm__sass_thread_inst_executed_op_ffma_pred_on.sum`.
Register count is `launch__registers_per_thread`.

### GEMM (`matMulVecMem_summary.csv`)

| Kernel | Calls | Duration (µs) | FFMA | DRAM (MB) | Regs | Block Lim Regs |
|---|---|---|---|---|---|---|
| `multiplyKernelNative` | 1 | 4.0 | 500,000,000 | 12.5 | 52 | 4 blocks/SM |
| `multiplyKernelTiledVectorizedA` | 1 | 4.2 | 520,224,768 | 12.0 | 36 | 6 blocks/SM |

**Diagnosis:** Both kernels are in the compute-bound regime (AI >> ridge
point based on DRAM bytes alone). The register pressure difference
(52 → 36 regs/thread after bank-conflict padding) is the key change —
Block Limit Registers jumps from 4 to 6 blocks/SM, giving more occupancy
headroom. The real occupancy gains came from the separate `gemm_cuBLAS.cu`
experiment (16→64 threads/block), not from these two kernels directly.

### Cholesky (`cholesky_tiled_summary.csv`, N=1024, 64-block tiling)

| Kernel | Calls | Duration sum (µs) | FFMA total | DRAM total (MB) | Regs | Block Lim Shared |
|---|---|---|---|---|---|---|
| `diagnoal_factorization_kernel` | 64 | 56.4 | 512,000 (8,000/call) | 19.8 | 25 | 7 blocks/SM |
| `column_update_kernel` | 63 | 69.2 | 671,744 | 226.7 | 20 | 3 blocks/SM |
| `trailing_submatrix_kernel` | 63 | 175.2 | 22,211,243 | 98.1 | 29 | 3 blocks/SM |

**Diagnosis:** Diagonal and column-update kernels are dependency-chain-bound,
not throughput-bound — duration is flat regardless of which block position in
the matrix (confirmed by the 64 and 63 instances showing consistent per-call
timing despite varying sub-matrix sizes). Both have Block Limit Shared as the
binding constraint (3 blocks/SM). cuSolver's own leaf kernel
(`potrf_alg2_cta_upper`) shows the identical signature — this ceiling is
intrinsic to sequential small-tile Cholesky. The trailing-submatrix kernel
is the only genuinely throughput-limited kernel of the three: 22.2M FFMA
and 98.1 MB DRAM scale with the sub-matrix size, confirming memory-bound
SYRK-like behavior.

**vs. cuSolver `potrf`:** Custom 3-kernel implementation: 4.94ms total.
cuSolver: 3.07ms total (~1.6× gap). Gap traced to cuSolver using a recursive
hybrid that hands large sub-blocks to cuBLAS GEMM/TRSM (`volta_sgemm_128x64`
visible in its kernel trace) — a structural difference, not a tuning gap.
See `numerical-solvers/profile/blocked_cholesky_kernel_vs_cusolver_summary.csv`
for the 192-row per-kernel trace of both implementations side by side.

### Custom CSR SpMV (`spmv_csr_summary.csv`)

| Kernel | Calls | Duration (µs) | FFMA | DRAM (MB) | Regs | Block Lim Regs |
|---|---|---|---|---|---|---|
| `spmv_csr_scalar_kernel` | 1 | 137.2 | 995,145 | 19.6 | 55 | 4 blocks/SM |
| `spmv_csr_vector_kernel` | 1 | 78.4 | 995,145 | 13.0 | 55 | 9 blocks/SM |

**Diagnosis:** Identical FFMA count, same registers per thread — the only
differences are duration (1.75× faster for vector) and DRAM traffic (13.0 vs
19.6 MB, 33% reduction). Warp-level reduction in the vector kernel cuts
redundant global loads. Both are deep in the memory-bandwidth-limited regime.

### cuSPARSE CSR SpMV (`cuSPARSE_spmv_csr_summary.csv`)

| Kernel | Calls | Duration (µs) | FFMA | DRAM (MB) |
|---|---|---|---|---|
| `csrmv_v3_kernel` | 1 | 84.3 | 0 ‡ | 11.6 |
| `csr_partition_kernel` | 1 | 10.5 | 0 ‡ | 70.1 |
| `vector_scalar_multiply_kernel` | 1 | 3.4 | 0 ‡ | 4.0 |

**Total cuSPARSE pipeline:** ~98.2 µs vs custom vector at 78.4 µs —
custom vector kernel is faster on this problem. cuSPARSE's partition kernel
dominates secondary cost (70.1 MB DRAM for preprocessing). Note FFMA=0 for
all cuSPARSE kernels: the library uses non-FMA integer indexing, so this
metric does not capture compute correctly.

### 2D Conjugate Gradient (`conjugate_gradient_2d_summary.csv`)

| Kernel | Calls | Duration sum (µs) | FFMA total | DRAM total (MB) |
|---|---|---|---|---|
| `spmv_csr_vector_kernel` | 2 | 551.1 | 1,308,672 | 23.0 |
| `update_x_r_kernel` | 2 | 25.0 | 524,288 | 6.0 |
| `update_p_kernel` | 2 | 13.6 | 262,144 | 2.7 |
| `cub::DeviceReduceKernel` | 5 | 11.3 | 248,832 | 1.8 |
| `cub::DeviceReduceSingleTileKernel` | 5 | 5.2 | 0 | 22.9 |

**Diagnosis:** SpMV is the dominant bottleneck — 551.1 µs vs 25+13.6 µs for
the update kernels. Root cause: the 17 MB working set (512×512 sparse matrix
+ solution/residual vectors) exceeds T4's 4 MB L2 cache, producing 8.1% L2
hit rate and repeated DRAM fetches per iteration. The update kernels are
fast AXPY-like operations and not the bottleneck.

### cuDSS Sparse Direct Solver (`cuDSS_Refactorization_summary.csv`)

Key kernels (dominant by duration):

| Kernel | Calls | Duration sum (µs) | DRAM (MB) | Regs | Block Lim Shared |
|---|---|---|---|---|---|
| `factorize_v3_ker` | 4 | 44.7 | 72.0 | 64 | 5 blocks/SM |
| `offsets_ker` | 1 | 54.1 | 14.2 | 40 | 16 blocks/SM |
| `fwd_ker` (triangular fwd solve) | 4 | 18.6 | 27.5 | 64 | 15 blocks/SM |
| `bwd_ker` (triangular bwd solve) | 4 | 26.4 | 36.7 | 64 | 28 blocks/SM |
| `independent_ker` | 4 | 11.6 | 47.0 | 64 | 28 blocks/SM |

**Diagnosis:** `factorize_v3_ker` is shared-memory-bound (Block Limit Shared:
5 blocks/SM) and has 64 registers/thread — the highest register pressure in
the suite. All cuDSS kernels report FFMA=0 in the CSV (same reason as
cuSPARSE: non-FMA integer indexing dominates these sparse graph algorithms).
The full pipeline is memory-bandwidth-bound at small N — DRAM traffic is high
relative to the actual numerical work being done.

---

## Regenerating Summaries

To regenerate all `*_summary.csv` files from full `*_report.csv` exports:

```bash
# From repo root
python3 benchmarks/ExtractNsightReportMetrics.py
```

Commit only the `*_summary.csv` outputs — the full `*_report.csv` files are
too large to commit (1,313 columns per row). The summary files contain the
8 metrics needed for roofline analysis, resume bullets, and README content.
