# Numerical Solvers on the GPU

From-scratch CUDA implementations of the core dense linear-algebra factorizations
(LU, Cholesky, and QR), each checked for correctness against a reference and
profiled with Nsight Compute on an NVIDIA T4 (Turing, `sm_75`, Google Colab).
The Cholesky implementation is also benchmarked kernel by kernel against NVIDIA's
cuSolver `potrf`.

All three factorizations share one structure: a latency-bound panel step followed
by a BLAS3 trailing update that holds most of the flops. The projects here
explore how each piece behaves on a GPU.

---

## Highlights

| Project | Result |
|---|---|
| **Blocked Cholesky** | Shared-memory tiling of the trailing update made that kernel **11.5× faster** (127 ms → 11 ms, N=2048) and cut total kernel time **7.1×**. Compute (SM) throughput rose **12.6×** (6.4% → 80.2%, duration-weighted over all 63 trailing launches). |
| **Cholesky vs. cuSolver `potrf`** | Custom 3-kernel version within **1.6×** of cuSolver at N=1024 (4.94 ms vs. 3.07 ms of kernel time), with a launch-by-launch breakdown of both. |
| **Tall-Skinny QR (TSQR)** | Warp-shuffle column updates made the panel QR kernel **2.21×** faster (67 µs → 30 µs) and nearly doubled active lanes per warp (8.9 → 16.8). |
| **LU and linear solves** | LU with partial pivoting, triangular solves, and a batched solver (10,000 matrices of 32×32) with a cuBLAS `getrfBatched`/`getrsBatched` baseline. |

---

## Environment and method

- **Hardware:** NVIDIA T4 (Turing, compute capability 7.5), Google Colab.
- **Profiling:** Nsight Compute. It locks the SM clock near base (about 584 MHz in the
  TSQR logs), so results are best read as ratios between runs.
- **Correctness:** every project compares against a CPU reference or an algebraic
  check (see each section).
- **Layout:** row-major storage, 32×32 thread blocks for Cholesky, one warp per
  block for TSQR.

---

## 1. Blocked Cholesky

A right-looking blocked factorization `A = L Lᵀ` with three kernels per block step:

1. **Diagonal tile** (`diagnoal_factorization_kernel`): factorizes the diagonal block in shared memory.
2. **Column/panel update** (`column_update_kernel`): triangular solve for the panel below it.
3. **Trailing update** (`trailing_submatrix_kernel`): rank-32 update of the remaining matrix.

### Tiling the trailing update

The same factorization was profiled with a naive global-memory trailing kernel and
a shared-memory tiled one (N=2048: 64 block steps, trailing grid shrinking from
63×63 to 1×1). Compared launch for launch:

| | Naive | Tiled | Change |
|---|---|---|---|
| Trailing kernel, first (largest) launch | 5.82 ms | 495 µs | 11.7× faster |
| Trailing kernel, mid-run launch (step 29) | 1.72 ms | 151.7 µs | 11.3× faster |
| Trailing kernel, sum of 63 launches | 127.1 ms | 11.0 ms | **11.5×** |
| Total kernel time (3 kernels) | 135.1 ms | 19.0 ms | **7.1×** |
| Compute (SM) throughput, first launch | 6.43% | 82.38% | **12.8×** |
| Compute (SM) throughput, step 29 | 6.41% | 79.25% | **12.4×** |
| Compute (SM) throughput, duration-weighted over 63 launches | 6.38% | 80.15% | **12.6×** |
| SM active cycles, first launch | 3,357,343 | 271,261 | 12.4× fewer |
| Warp cycles per issued instruction, first launch | 486.3 | 33.3 | 14.6× fewer |
| IPC (active), first launch | 0.07 | 0.92 | 13.1× |

Nsight's stall analysis confirms the change addressed the right bottleneck: on the
naive kernel 97.5% of stall cycles wait on the L1 queue for global-memory
operations, and on the tiled kernel the dominant stall becomes the shared-memory
(MIO) queue at about 68%, which is the expected profile for a tiled kernel.

With the trailing update optimized, the diagonal and column kernels make up
42% of the remaining kernel time. Both are sequential dependency chains (a
loop with a `__syncthreads()` per column inside one tile), which makes them the
next optimization target.

### Comparison with cuSolver `potrf`

At N=1024 (32 block steps) the custom factorization and `cusolverDnSpotrf` were
profiled on the same matrix, and both match the CPU reference (tolerance 1e-4).

| | Custom (3 kernels) | cuSolver |
|---|---|---|
| Total kernel time | 4,942 µs | 3,066 µs |

Launch-by-launch (cuSolver roles inferred from kernel names):

| Stage | Custom | cuSolver |
|---|---|---|
| Diagonal / leaf | `portf_diag_kernel`: 32 × 56.5 µs = 1,807 µs | `potrf_alg2_cta_upper`: 32 × 37.9 µs = 1,212 µs |
| Panel solve | `column_update_kernel`: 31 × 51.1 µs = 1,586 µs | `potrf_alg2_upper_trsm`: 30 × 18.6 µs = 559 µs |
| Trailing update | `trailing_matrix_kernel`: 31 × 50.0 µs = 1,550 µs | `potrf_alg2_syrk_kernel`: 30 × 19.2 µs = 576 µs, plus 4 larger launches (`kernel_trsm_l_mul32` ×2, `volta_sgemm_128x64` ×2) = 713 µs |

The difference between the two implementations sits mainly in the per-step
diagonal and panel kernels (3,392 µs for the custom pair vs. 1,771 µs for
cuSolver's, which is 86% of the 1.87 ms gap). cuSolver's larger TRSM/SGEMM
launches account for 23% of its time.

---

## 2. Tall-Skinny QR (TSQR)

Householder QR of a tall, narrow matrix `A (M×N)` using a reduction tree:

- **Leaf level:** each block factorizes its own 128×4 panel (`panel_qr_baseline`).
- **Merge levels:** pairs of R factors are stacked into a 2N×N system and
  re-factorized (`tsqr_merge_kernel`); Householder vectors are saved to a separate
  workspace so Q can be rebuilt.
- **Configuration:** M=512, N=4, 128-row leaves, so 4 leaves and 2 merge levels.

**Optimization.** The trailing-column update moved from one thread per column to one
warp per column: the dot product is split across rows and reduced with
`__shfl_down_sync`.

| Kernel | Baseline | Optimized | Change |
|---|---|---|---|
| Panel QR duration | 67.14 µs | 30.37 µs | **2.21× faster** |
| Panel QR, active threads per warp | 8.90 | 16.76 | +88% |
| Merge level 1 duration | 17.06 µs | 17.38 µs | unchanged |
| Merge level 2 duration | 16.96 µs | 17.50 µs | unchanged |

The merge system is only 8×4, so there is little work to spread across a warp;
the warp-shuffle merge kernel is built for larger N. All TSQR kernels run at under
2% of peak compute throughput at this size, and Nsight reports the grid as too
small to fill the device. The profile shows a latency-bound, problem-size-limited
regime.

**Correctness.** `TSQR_Verification.cu` and `TSQR_Optimized.cu` rebuild Q on the
host from the leaf and merge reflectors and check `‖A − QR‖_F / ‖A‖_F` and
`‖QᵀQ − I‖_F` against a 1e-4 threshold, for the baseline and optimized kernels
respectively. `PanelQRVerification.cu` checks the panel kernel alone on a 4×3 matrix.

---

## 3. LU factorization and linear solves

Right-looking LU with partial pivoting, built up in stages and extended to batches.
Each elimination step is a short kernel sequence: pivot search, row swap, column
scaling (the multipliers of L), and a rank-1 update of the trailing matrix. L has an
implicit unit diagonal, and U's diagonal is stored in `A`.

| File | Description |
|---|---|
| `LUFactorization.cu` | In-place LU without pivoting (normalize + update kernels); 3×3 result checked against a hand-computed reference |
| `fullSolver.cu`, `fullsolverRef.cu` | `Ax = b` with LU plus forward and backward substitution; 3×3 residual check |
| `partialPivoting.cu` | Adds partial pivoting: device-side pivot search, whole-row swap with `b` permuted alongside; n=1024 random system verified by host residual in double precision |
| `LUUpdateTiling.cu` | Same solver with the rank-1 update staged through shared memory (16×16 blocks) |
| `lowerTriangle.cu`, `backwardSubsitution.cu` | Practice code for the forward and backward substitution steps; the working implementations are in the full solvers above |
| `batchedSolver.cu` | Pivoted LU and substitution over a batch: 10,000 matrices of 32×32, batch index in the grid, one launch per step for the whole batch; CUDA events time the LU, forward, and backward phases separately; per-matrix residual check |
| `cuBLAsLinearSolver.cu`, `cuBLAsLinearSolver_learning.cu` | The same batch solved with `cublasSgetrfBatched` + `cublasSgetrsBatched` as a library baseline (the second file adds CUDA-event timing) |

**Design notes**
- The pivot index stays on the device, so no host round trip is needed per step.
- Swapping whole rows keeps the already-computed L entries consistent with the
  permuted matrix, matching the LAPACK convention.
- The factorization permutes `b` as it goes, so each solve handles one right-hand side.
- The rank-1 update is BLAS2 work, so shared-memory staging reduces redundant loads
  but arithmetic intensity stays low. Blocked LU with a GEMM trailing update is
  the BLAS3 formulation, and the Cholesky work in section 1 shows that pattern.

---

## Repository contents

```
numerical-solvers/
├── LU / solvers      LUFactorization.cu, fullSolver.cu, fullsolverRef.cu,
│                     partialPivoting.cu, LUUpdateTiling.cu, lowerTriangle.cu,
│                     backwardSubsitution.cu, batchedSolver.cu,
│                     cuBLAsLinearSolver.cu, cuBLAsLinearSolver_learning.cu
├── Cholesky          diagonal_factorization.cu, columnUpdate.cu,
│                     TrailingSubmatrixSubtractNative.cu,
│                     TrailingSubmatrixSubtractTiling.cu,
│                     blocked_cholesky_cusolver.cu, cholesky_profiling_report.md
├── TSQR              PanelQRVerification.cu, panel_qr_baseline.cu, TSQR_Merge.cu,
│                     TSQR_Verification.cu, TSQR_Optimized.cu, tsqr_profiling_report.md
└── profile/          Nsight Compute logs and exports
                      (Cholesky naive/tiled traces, cuSolver comparison CSVs,
                       TSQR baseline/optimized traces)
```

## Build and profile

```bash
nvcc -O3 -arch=sm_75 <file>.cu -o <name>
nvcc -O3 -arch=sm_75 cuBLAsLinearSolver.cu -o cublas_solver -lcublas
nvcc -O3 -arch=sm_75 blocked_cholesky_cusolver.cu -o chol_vs_cusolver -lcublas -lcusolver
ncu --set full ./<name>
```

## Extensions

- Blocked LU with panel factorization, row-swap application, and a GEMM trailing update
- Block-level pivot-search reduction and per-kernel profiling for the LU pipeline
- Compact-WY accumulation for the QR trailing update
- Tile-size sweep and cuBLAS-based trailing update for the Cholesky comparison
