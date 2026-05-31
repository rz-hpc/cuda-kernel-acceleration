
Performance Analysis: Naive vs. Shared Memory Tiled Trailing Submatrix Update in Blocked Cholesky Factorization

This report presents an in-depth performance analysis comparing the naive global memory implementation and the optimized shared-memory tiled implementation of the trailing submatrix update kernel ($A_{22} = A_{22} - L_{21} L_{21}^T$) in a Blocked Cholesky solver.

The data used for this comparison was captured using NVIDIA Nsight Compute on a Turing-architecture GPU (Compute Capability 7.5, with an SM clock speed of ~585 MHz and DRAM clock speed of ~5.0 GHz) at a matrix dimension of $N = 2048$ with $32 \times 32$ thread blocks.

1. High-Level Summary of Executed Kernels

During the Cholesky solver execution, three kernels cooperate over 64 outer loop iterations (k_block). The workload distribution across these kernels is highly asymmetric:

Kernel 1 (Diagonal Factorization): Launches exactly 1 block globally per step. Because of the extremely small grid size, it achieves close to 0% of the device's FP32 peak.

Kernel 2 (Column Update): Launches $64 - k_{\text{block}} - 1$ blocks (averaging 32 blocks) on a 1D grid. Because it fails to fill the device's waves, it also hovers near 0% to 1% of the device's FP32 peak (occupying only a tiny fraction of the SMs).

Kernel 3 (Trailing Submatrix Update - Naive/Tiled): Launches a 2D grid of size up to $63 \times 63 = 3969$ blocks on the first iteration. Because it performs $O(N^3)$ operations, this is the primary compute engine of Cholesky factorization.

Below we compare the performance profile of the naive and tiled implementations of Kernel 3.

2. Key Performance Metrics Comparison

Metric

Naive Kernel (cholesky_native)

Tiled Kernel (cholesky_tiled)

Performance Delta / Analysis

FP32 Roofline Achievement

~1.0%

~8.0% - 9.0%

8x - 9x Speedup in raw compute peak utilization.

SM Active Cycles

3,357,342.88

80,183.40

41.8x reduction in active execution cycles.

Duration (First Wave)

5.82 ms

151.74 us

38.3x execution time speedup on the heaviest wave.

Compute (SM) Throughput

6.43%

79.25%

12.3x increase in SM execution resource efficiency.

Memory Throughput

49.44%

79.25%

Balanced execution profile achieved on tiled layout.

DRAM Throughput

1.18%

11.39%

Low DRAM bandwidth % reflects active cache hits.

L1/TEX Cache Throughput

98.88%

87.86%

Tiled utilizes dedicated Shared Memory paths instead of L1 data cache pressure.

IPC Active (Instructions/Cycle)

0.07

~0.92

13.1x improvement in instruction scheduling density.

Warp Cycles Per Issued Inst.

486.28 cycles

33.44 cycles

14.5x faster scheduling turnaround time.

Warp Cycles Per Executed Inst.

486.64 cycles

33.52 cycles

Eliminates long scoreboards from VRAM reads.

3. Deep-Dive Warp Stall & Bottleneck Analysis

The Nsight Compute full profile reveals two completely different hardware failure modes in these kernels, illustrating the journey from a memory-bound state to a compute-latency-bound state.

A. Naive Kernel Bottleneck: The Memory Wall

The naive trailing submatrix kernel sits on the steep, left-hand slope of the Roofline chart. Its instruction pipeline is completely paralyzed by memory stalls.

Dominant Stall Reason: L1 Instruction Queue Stalls / Local and Global (LG) Memory (97.5% of all stalls)

Cycles Stalled: 473.6 cycles out of the 485.7 average cycles between instruction issues.

Hardware Explanation: The inner loop fetches elements from global memory directly:

float val_row = d_A[global_row_idx * n + (panel_start_col + dot_idx)];
float val_col = d_A[global_col_idx * n + (panel_start_col + dot_idx)];


Every thread makes active DRAM transactions. The warp scheduler is forced to stall because the L1/LG instruction queue becomes completely saturated. This limits instruction throughput to a mere 0.07 IPC.

B. Tiled Kernel Bottleneck: Pipeline Stalls and Constant Broadcasts

By loading block tiles into __shared__ memory, the tiled kernel moves off the global memory highway.

Dominant Stall Reason: MIO (Memory Input/Output) / Shared Memory Queue Stalls (67.9% of all stalls)

Cycles Stalled: 22.7 cycles out of the 33.4 average cycles between instruction issues.

Hardware Explanation: 1. By loading the panel block into shared memory tiles (tile_row and tile_col), VRAM DRAM traffic is eliminated.
2. In the compute phase, the loop executes:
sum += tile_row[ty][k] * tile_col[tx][k];
Because tx is fixed across threads in the same column lane of the warp, every thread in the warp requests the exact same shared memory bank address at step k. The hardware intercepts this and triggers an instantaneous Constant Broadcast, leading to zero shared memory bank conflicts.
3. However, because the math loop is extremely tight, issuing memory reads from shared tiles on every iteration saturates the hardware's local Memory Input/Output (MIO) pipeline queue. The warp scheduler is no longer waiting on global VRAM latency (Long Scoreboard stalls drop to 0%), but must wait for the shared memory address execution pipelines to clear the queue.

4. Architectural Recommendations for Next Optimization Milestone

To push the tiled kernel beyond 9% FP32 peak performance and eliminate the MIO pipeline stalls, the following compiler and structural optimizations are recommended:

Loop Unrolling (#pragma unroll):
Adding #pragma unroll above the dot product accumulation loop forces the compiler to expand the 32 iterations. This removes branch arithmetic overhead and allows the compiler to interleave memory requests with independent Fused Multiply-Add (FFMA) pipelines, hiding shared memory latency.

Vectorized Loads (Vectorization):
Instead of fetching single float values, cast shared memory accesses to float4 (utilizing LDG.E.128 instructions). This reduces the absolute number of memory instructions dispatched to the MIO queue by 75%, immediately preventing pipeline queue saturation.

Thread Tiling:
Configure each thread to compute a small $2 \times 2$ or $4 \times 4$ sub-matrix of outputs rather than a single element. This increases register-level data reuse, drastically reducing the demand on shared memory read bandwidth.

