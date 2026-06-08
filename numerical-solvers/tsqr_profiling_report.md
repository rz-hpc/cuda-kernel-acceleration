# Performance Analysis: Baseline vs. Warp-Shuffle Optimized TSQR Kernels

This report presents a hardware-level performance comparison between the baseline shared-memory
Householder panel QR kernel and the warp-shuffle optimized variant, across both the leaf-level
panel factorization and the tree-reduction merge kernels of the Tall-Skinny QR (TSQR) factorization.

All profiling data was captured using NVIDIA Nsight Compute on a Turing-architecture GPU
(Compute Capability 7.5, SM clock ~584 MHz, DRAM clock ~4.97–4.99 GHz) at matrix dimensions
of $M = 512$, $N = 4$, partitioned into 4 leaf blocks of 128 rows each, with a 3-level
reduction tree.

---

## 1. Algorithm Overview

TSQR factorizes a tall-skinny matrix $A \in \mathbb{R}^{M \times N}$ ($M \gg N$) as $A = QR$
via a hierarchical tree reduction:

- **Level 0 (leaf):** Each block independently runs `panel_qr` on its $128 \times 4$ submatrix,
  producing a local $R_i$ and a packed set of Householder vectors stored in-place.
- **Levels 1–2 (merge):** Pairs of $R$ factors are stacked into a $2N \times N$ system and
  reduced by `tsqr_merge`, with Householder vectors written to a separate workspace buffer
  `d_V_merge` to prevent clobbering across levels.

The optimization target was the **trailing column update** inside both kernels — replacing the
column-per-thread pattern (one thread owns one column, serial inner loop) with a
column-per-warp pattern using `__shfl_down_sync` to parallelize the dot product reduction.

---

## 2. Key Performance Metrics

### 2.1 Panel QR Kernel (`panel_qr_baseline_kernel` vs. `qr_base_optimized_kernel`)

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| Duration | 67.14 µs | 30.37 µs | **2.21× faster** |
| Elapsed Cycles | 39,230 | 17,746 | 2.21× reduction |
| SM Active Cycles | 3,799.70 | 1,651.90 | 2.30× reduction |
| Compute (SM) Throughput | 1.32% | 0.94% | — |
| Memory Throughput | 1.32% | 0.94% | — |
| DRAM Throughput | 0.14% | 0.27% | — |
| L1/TEX Cache Throughput | 13.61% | 10.10% | — |
| Avg. Active Threads / Warp | 8.90 | **16.76** | **+88% warp utilization** |
| Avg. Not Predicated Off / Warp | 7.86 | 15.07 | +92% |
| Warp Cycles Per Issued Instruction | 6.01 | 5.68 | 5.5% reduction |
| Warp Cycles Per Executed Instruction | 6.02 | 5.69 | 5.5% reduction |
| Fixed Latency Stall (Est. Speedup) | 40.25% | 44.87% | — |

### 2.2 Merge Kernel (`tsqr_merge_kernel` vs. `tsqr_merge_optimized_kernel`)

Results shown for both tree levels. The merge system operates on $2N \times N = 8 \times 4$,
which fundamentally constrains how much warp parallelism can be exploited.

| Metric | Baseline L1 | Optimized L1 | Baseline L2 | Optimized L2 |
|---|---|---|---|---|
| Duration | 17.06 µs | 17.38 µs | 16.96 µs | 17.50 µs |
| Elapsed Cycles | 9,935 | 10,104 | 9,872 | 10,195 |
| SM Active Cycles | 437.00 | 445.45 | 217.03 | 225.72 |
| Compute (SM) Throughput | 0.17% | 0.16% | 0.08% | 0.08% |
| Avg. Active Threads / Warp | 8.81 | **12.48** | 8.81 | **12.48** |
| Avg. Not Predicated Off / Warp | 7.38 | 10.81 | 7.38 | 10.81 |
| Warp Cycles Per Issued Instruction | 7.72 | 7.67 | 7.68 | 7.76 |
| Fixed Latency Stall (Est. Speedup) | 35.88% | 37.04% | 36.08% | 36.62% |

---

## 3. Analysis

### 3.1 Panel QR: Meaningful Win from Warp Parallelism

The 2.21× speedup in the panel kernel is the headline result. The baseline kernel assigned one
thread per trailing column (`j = k + 1 + tx; j < N; j += blockDim.x`), meaning at most
$N - k - 1 \leq 3$ threads were active during the trailing update with $N = 4$. The remaining
29 threads sat idle.

The optimized kernel reassigns all 32 warp threads to cooperate on a single column at a time,
partitioning the dot product across rows (`i = k + tx; i < M; i += blockDim.x`), reducing via
`__shfl_down_sync`, then applying the rank-1 update in the same strided pattern. This is
confirmed by the active thread count jumping from **8.90 to 16.76 per warp** — a near-doubling
of warp lane utilization during the compute phase.

The SM active cycle reduction (3,800 → 1,652) shows that real work is finishing faster, not
just that idle cycles were reclassified. The slight increase in fixed-latency stall percentage
(40.25% → 44.87%) is expected: the warp shuffle instructions themselves introduce a small
pipeline dependency that becomes more visible as the kernel becomes less memory-latency-bound
and more instruction-latency-bound.

### 3.2 Merge Kernel: Warp Parallelism Hits the Problem-Size Floor

The merge kernel results are nearly flat — optimized and baseline are within ~2% of each other
on duration and active cycles. Active threads improved from 8.81 to 12.48 per warp, confirming
that the shuffle loop does engage more lanes, but the overall execution time is unchanged.

The reason is structural: the merge system is $2N \times N = 8 \times 4$ — only 8 rows and 3
trailing columns at most. With $M_{\text{merge}} = 8$ and BLOCK\_DIM = 32, at most 8 threads
contribute to any dot product even in the optimized kernel. The warp reduction sums 32 partial
values where 24 are zero. The compute payload (180 FLOPs, as derived in the arithmetic intensity
analysis) is simply too small for parallelism to overcome kernel launch overhead and
synchronization cost.

This is not a code quality issue — it is an architectural mismatch. The merge kernel is
occupancy-limited by problem size, not by memory bandwidth or instruction throughput.

### 3.3 Both Kernels Remain Firmly Memory-Bound and Occupancy-Limited

All kernels report `< 1%` of peak FP32 throughput and Nsight's grid-too-small warning on every
launch. This is consistent with the arithmetic intensity analysis (AI ≈ 0.99 FLOP/byte for panel
QR, ≈ 0.90 for merge) — both kernels sit far left of the H100/T4 ridge point, meaning
performance scales with memory bandwidth, not compute.

The practical ceiling for this workload at $M = 512$, $N = 4$ is set by:

1. **Grid size:** 4 leaf blocks, 2–1 merge blocks — far too few to fill the T4's 40 SMs.
2. **Problem arithmetic intensity:** Householder QR is inherently ~1 FLOP/byte without WY
   accumulation into a GEMM.
3. **Shared memory serialization:** Thread 0 computing norms serially (visible in the ~40%
   fixed-latency stall) is the next bottleneck after the warp utilization improvement.

---

## 4. Architectural Recommendations for Next Optimization Step

**Parallelize the norm computation (panel QR).**
Thread 0 serially accumulates $\sum x_i^2$ over all $M = 128$ rows at the start of each column
step. This is a pure parallel reduction and the dominant remaining bottleneck indicated by the
~40–45% fixed-latency stall fraction. Replacing it with a shared-memory tree reduction across
all 32 threads would eliminate the serial dependency entirely.

```cuda
// Replace thread-0 serial loop with parallel reduction:
__shared__ float s_reduce[BLOCK_DIM];
float partial = 0.0f;
for (int i = k + tx; i < M; i += blockDim.x)
    partial += tile_A[i][k] * tile_A[i][k];
s_reduce[tx] = partial;
__syncthreads();
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tx < s) s_reduce[tx] += s_reduce[tx + s];
    __syncthreads();
}
// s_reduce[0] now holds sum_squares
```

**Scale $N$ to make the merge kernel worthwhile.**
With $N = 4$, the $2N \times N$ merge system is too small for 32-thread cooperation. At $N = 16$
or $N = 32$, the merge kernel would have 32–64 active rows, fully utilizing the warp for both
the dot product and rank-1 update. The warp-shuffle merge kernel is the right design — it is
simply waiting for a problem large enough to justify it.

**WY accumulation for the trailing update.**
Accumulate the $b$ Householder reflectors in the panel into a compact $I - VTV^T$ form and apply
the trailing update as a DGEMM-equivalent operation. This is the standard approach in LAPACK's
`DLARFB` and is the only path to compute-bound performance (AI ~ 300 FLOP/byte vs. the current
~1 FLOP/byte).
