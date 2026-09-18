# High-Performance CUDA Kernel & Distributed GPU Numerics Portfolio

A hands-on engineering portfolio built across two connected tracks: single-GPU CUDA
kernel optimization, profiled end-to-end with Nsight Compute and roofline analysis
against the T4's hardware ceilings; and distributed multi-GPU/multi-node numerical
computing, from hand-derived 2D block-cyclic layouts and MPI process grids up through
a verified 4-GPU SUMMA GEMM benchmarked directly against NVIDIA's cuBLASMp.

---

## Background & Motivation

Across 8 years engineering iterative numerical solvers (CG, LU, domain decomposition)
at Siemens EDA and 4 years optimizing high-throughput concurrent systems and diagnosing
production distributed-systems failures at Microsoft — including lock-free pipelines,
NUMA-aware memory layout, and on-call ownership of Analysis Services' multi-node Azure
deployment — I built a deep intuition for hardware-aware algorithm design and for how
distributed systems fail under partial outages and network latency.

This project extends that foundation in two directions: (1) single-GPU CUDA kernel
engineering, diagnosed and iterated against specific micro-architectural bottlenecks
(bank conflicts, occupancy ceilings, roofline positioning); and (2) distributed
multi-GPU and multi-node numerical computing — building a distributed GEMM (SUMMA) from
scratch on MPI process grids and NCCL collectives, then benchmarking it against
NVIDIA's cuBLASMp on real hardware. The goal throughout is not to replicate tutorials —
it is to develop the same instincts on distributed GPU hardware that years of CPU
systems work built on the CPU side.

**Target role focus:** Distributed multi-GPU/multi-node numerical computing (PBLAS /
ScaLAPACK-style 2D block-cyclic distribution, MPI, NCCL, cuBLASMp), math library
engineering (cuBLAS / cuSPARSE / cuSOLVER workflows), HPC kernel optimization.

---

## Key Concepts Demonstrated

**Single-GPU kernel engineering:** `shared memory tiling` · `bank conflict analysis &
padding` · `warp-level primitives (__shfl_down_sync)` · `Nsight Compute profiling` ·
`occupancy Block Limit analysis` · `roofline / arithmetic intensity analysis` ·
`memory coalescing` · `L1/L2 cache hierarchy` · `batched LU factorization` ·
`blocked Cholesky (LL^T)` · `Tall-Skinny QR (TSQR, Householder)` ·
`Krylov iterative solvers (CG)` · `FlashAttention block tiling` · `BLAS-level GEMM` ·
`CSR SpMV` · `cuSPARSE comparison` · `cuDSS sparse direct solver` ·
`Schur complement rank-1 update` · `FP16 trait dispatch and type-safe accumulation`

**Distributed multi-GPU / multi-node computing:** `MPI Cartesian process grids` ·
`2D block-cyclic distribution (PBLAS/ScaLAPACK-style)` · `global_to_local /
local_to_global index mapping` · `MPI_Comm_split row/column sub-communicators` ·
`NCCL Broadcast collectives` · `distributed SUMMA GEMM` · `CUDA stream double-buffering
for communication/compute overlap` · `GPU interconnect topology analysis (nvidia-smi topo)` ·
`MPI_Comm_split_type node-local rank binding` · `cuBLASMp (NVIDIA distributed BLAS)` ·
`Nsight Systems distributed timeline profiling` · `Frobenius-norm distributed correctness verification`

---

## Distributed Computing (Primary Focus)

Built bottom-up: CPU-only MPI mechanics → single-node multi-GPU SUMMA → multi-node
infrastructure → cuBLASMp benchmark comparison.

### Phase 1 — MPI Fundamentals (`distributed-computing/mpi-fundamentals/`) ✅ Complete
- 2D process grid via `MPI_Cart_create` + `MPI_Dims_create` (auto-factors any rank
  count into a balanced P×Q grid — verified across 2×2 and 3×2 grids).
- Hand-derived 2D block-cyclic index mapping (`global_to_local` / `local_to_global`):
  the same distribution model PBLAS/ScaLAPACK use — named explicitly as a qualification
  in NVIDIA's Dense Linear Algebra math-libs job postings.
- Full scatter → compute → gather round-trip verified end-to-end.

### Phase 2 — Single-Node Multi-GPU SUMMA (`distributed-computing/nccl-multi-gpu/`) ✅ Complete
- From-scratch distributed GEMM: MPI process grid + NCCL row/column panel broadcasts +
  CUDA stream double-buffering for communication/compute overlap.
- Generalized to arbitrary M/N/K via `get_local_matrix_dim`, real random test matrices,
  and Frobenius-norm-ratio verification (`||C_dist - C_ref|| / ||C_ref||`).
- Verified correct on real 2-GPU hardware (RunPod, 2× RTX 2000 Ada).
- **Topology finding:** `nvidia-smi topo -m` showed `SYS`-only connectivity (cross-NUMA
  socket, no NVLink, no shared PCIe switch) — ~1.4 GB/s broadcast throughput, no
  measurable communication/compute overlap despite correct double-buffering. The same
  failure mode as CPU-side cross-NUMA cache traffic: topology sets a ceiling no software
  fix can raise.

### Phase 3 — Multi-Node Infrastructure ✅ Logic verified, real-hardware run pending
- Fixed a device-binding bug invisible to single-node testing: `cudaSetDevice` must use
  a node-local rank (`MPI_Comm_split_type, MPI_COMM_TYPE_SHARED`), not the global rank.
  Single-node testing cannot expose this bug — local and global ranks always coincide
  when there is only one machine.
- Added `ncclAllReduce` cross-node proof (verified against expected sum).
- Logic verified on Colab; 2-node Cluster run is the next step.

### Phase 4 — cuBLASMp Benchmark (`summa_cublasmp.cu`) ✅ Comparison complete on 4-GPU hardware

**Hardware:** 4× NVIDIA RTX PRO 4500 (Blackwell), single node, `NODE`/`PHB` topology,
same NUMA domain, no NVLink. CUDA 13.0, Driver 580.126.20.

**Correctness** — both implementations independently agree to ~6.86e-07 Frobenius norm
ratio, mutually validating each other:

| Config | Custom SUMMA | cuBLASMp |
|---|---|---|
| `-np 1` (1×1 grid) | 6.86e-07 PASS | 0 (exact — local dispatch) PASS |
| `-np 2` (2×1 grid) | 6.94e-07 PASS | 6.87e-07 PASS |
| `-np 4` (2×2 grid) | 6.86e-07 PASS | 6.86e-07 PASS |

**End-to-end wall-clock performance:**

| Config | Custom SUMMA | cuBLASMp | Custom advantage |
|---|---|---|---|
| `-np 1` | 100.8 ms / 0.728 TFLOPS | 249.9 ms / 0.294 TFLOPS | **2.48×** |
| `-np 2` | 93.2 ms / 0.788 TFLOPS | 110.2 ms / 0.666 TFLOPS | **1.18×** |
| `-np 4` | 105.2 ms / 0.698 TFLOPS | 152.3 ms / 0.482 TFLOPS | **1.45×** |

**Kernel-level analysis (Nsight Systems, `-np 4`, rank 0):**

| Metric | Custom SUMMA | cuBLASMp |
|---|---|---|
| NCCL broadcast total (rank 0) | 23.56 ms / 10 calls | 4.21 ms / 10 calls |
| GEMM kernel total (rank 0) | 3.10 ms / 6 calls | 3.09 ms / 6 calls |
| Broadcast % of GPU time (rank 0) | 88.4% | 57.7% |
| Broadcast rank imbalance | **7.2× spread** (3.3–23.6 ms) | **1.5× spread** (3.0–4.5 ms) |
| GEMM kernel variant | `cutlass_simt_sgemm_256x128` (all ranks) | Two different variants per rank — cuBLASMp's heuristic adapts to local sub-matrix shape |

**Key findings:**
1. Custom SUMMA is faster end-to-end at every rank count, but its per-rank NCCL
   broadcast cost is far less balanced (7.2× spread vs. cuBLASMp's 1.5×) —
   production libraries specifically optimize collective scheduling to reduce this.
2. cuBLASMp's internal kernel selection is adaptive per-rank (two different CUTLASS
   tile shapes in the same 4-rank run), responding to uneven sub-matrix dimensions
   from the 2×2 block-cyclic split.
3. The `-np 1` gap (2.48×) correlates with cuBLASMp's trace showing `Using local
   Matmul` at that grid size — a structurally different code path from its generic
   distributed algorithm.

**API/environment findings from cuBLASMp integration:**
- Current cuBLASMp (≥0.8.0) uses NCCL symmetric memory, not NVSHMEM — older docs and
  samples referencing NVSHMEM are stale.
- `SPLIT_P2P` and other explicit pipelined algorithms return
  `CUBLASMP_STATUS_NOT_SUPPORTED` on hardware without NVLink — `DEFAULT` falls back
  cleanly to `no_overlap`.
- Matrix descriptors require the same row-major/column-major transpose convention as
  plain cuBLAS calls (B-first argument order with transposed dimensions).

---

## Roofline Analysis (NVIDIA Tesla T4, sm_75)

**Hardware ceilings:** 8,100 GFLOPs/s (fp32), 320 GB/s DRAM bandwidth,
ridge point at **25.31 FLOP/Byte**.

**Data provenance note:** All values come from `generate_roofline.py` and `get_flop.py`,
which parse Nsight Compute CSV reports. Both scripts compute AI as
`(fadd + fmul + 2×ffma) / dram__bytes.sum` — raw DRAM traffic only, not the full
memory hierarchy. For kernels where the CSV parse returned no usable DRAM data (notably
the GEMM kernels `multiplyKernelNative` and `multiplyKernelTiledVectorizedA`), the
script falls back to hardcoded constants (`AI=80.0 FLOP/B`, `gflops=1.67` and `8.63`
respectively) rather than parsed measurements. The `%opt` values for those two kernels
derive from those hardcoded constants, not from Nsight data directly. All other kernels
(Cholesky, SpMV, CG, cuDSS) have AI values parsed directly from their CSV reports.

| Kernel | AI (FLOP/B) | GFLOPs/s | %opt | Source | Bound | Key finding |
|---|---|---|---|---|---|---|
| Native MatMul | 80.0 † | 1.67 † | 0.02% | hardcoded fallback | Compute (barely) | Well above ridge but near-zero utilization — naive global memory, no tiling |
| Tiled Vectorized MatMul | 86.6 † | 8.63 † | 0.11% | hardcoded fallback | Compute | 5.5× better than native; still <0.2% — register-limited occupancy per Nsight |
| Cholesky Diagonal | 0.81 | 1.50 | 0.58% | parsed CSV | Memory (sequential) | `__syncthreads()`-gated serial loop; dependency-chain-bound, not resource-bound |
| Cholesky Column Update | 5.93 | 42.0 | 2.21% | parsed CSV | Memory | Duration flat regardless of grid size — same dependency-chain signature |
| Cholesky Trailing Submatrix | 6.68 | 135.2 | 6.32% | parsed CSV | Memory | Largest AI of the three; duration and DRAM throughput scale with grid — SYRK-like, genuinely memory-bound |
| Custom CSR SpMV (scalar) | 0.10 | 0.0145 | 0.04% | parsed CSV | Memory | Indirect indexing thrashes L2; 137ms for 19.6 MB |
| Custom CSR SpMV (vector) | 0.25 | 0.0419 | 0.05% | parsed CSV | Memory | 1.75× faster than scalar, both far from the roofline |
| cuSPARSE CSR SpMV | 0.29 | 0.0397 | 0.04% | parsed CSV | Memory | Comparable to custom vector — structured sparsity on this problem offers no advantage over well-written CSR |
| CG SpMV bottleneck | 1.58 | 0.0661 | 0.01% | parsed CSV | Memory | 551ms for 23 MB — cache thrashing from 17 MB working set exceeding T4's 4 MB L2 |
| CG Vector Update X/R | 0.17 | 0.0419 | 0.08% | parsed CSV | Memory | Simple AXPY-like update; ~25ms |
| CG Vector Update P | 0.19 | 0.0386 | 0.06% | parsed CSV | Memory | Same shape; ~14ms |
| cuDSS Analysis (N=100) | 0.018 ‡ | 0.00015 ‡ | 0.00% | estimated ‡ | Memory | Graph preprocessing — cuDSS kernels returned 0 FLOPs in CSV; values estimated from memory-bound profile |
| cuDSS Factorize (N=100) | 0.065 ‡ | 0.0028 ‡ | 0.01% | estimated ‡ | Memory | `factorize_v3_ker` ~110ms per call, 3.27 MB DRAM |
| cuDSS Solve (N=100) | 0.048 ‡ | 0.0014 ‡ | 0.01% | estimated ‡ | Memory | fwd+bwd triangular solve kernels |

† Hardcoded fallback constant in `generate_roofline.py` — CSV parse returned no data for these kernels.
‡ Estimated in script (`generate_roofline.py` lines 182–197) — cuDSS kernels showed 0 FLOPs in CSV;
  comment in script explicitly notes "estimated based on memory-bound operation profile."

**Roofline takeaway:** All kernels sit well below the peak ceiling. GEMM kernels are
above the ridge point (compute-bound regime) but achieve <0.2% of peak — low occupancy
prevents them from reaching theoretical throughput. All sparse/iterative/factorization
kernels are deep in the memory-bandwidth-limited regime. This makes different
optimization strategies necessary for different kernel families — bank-conflict padding
for GEMM, spatial tiling for SpMV, algorithmic restructuring for Cholesky's serial
diagonal steps — rather than any single fix applying broadly.

![Roofline Analysis](benchmarks/roofline_analysis.png)

---

## Performance Highlights (NVIDIA Tesla T4, sm_75, unless noted)

| Kernel / Module | Baseline | Optimized | Speedup | Key technique |
|---|---|---|---|---|
| Matrix Transpose | 0.2286 ms (naive) | 0.0737 ms (tiled + padded) | **~3.1×** | Shared mem staging + bank-conflict padding (32×33) |
| 2D Poisson SpMV vs Tiling | CSR scalar (~8.1% L2 hit) | 32×32 spatial tiling | **~4×** | Eliminated L2 thrash via SMEM staging |
| GEMM: bank conflicts → occupancy → vs. cuBLAS | 16 thr/block: 2.13ms (f32) / 24.54ms (f64) | 64 thr/block: 1.30ms (f32) / 11.85ms (f64) | **1.6×/2.1×** | Padding + `THREAD_TILE_SIZE` retune; f64 beats cuBLAS's `volta_dgemm_128x64_nn` |
| Blocked Cholesky vs. cuSolver `potrf` | — | 4.94ms vs 3.07ms (cuSolver) | ~1.6× gap (architectural) | Traced to recursive-hybrid vs. single-level-blocking algorithm, not kernel tuning |
| Custom SUMMA GEMM vs. cuBLASMp (4-GPU, Blackwell) | — | 105ms / 0.698 TFLOPS | **1.45× faster** than cuBLASMp | Hand-rolled MPI+NCCL SUMMA vs. vendor library; see Distributed section |

---

## Micro-Architectural Case Studies

### Bank Conflict Elimination via Structural Padding
Column-wise shared memory writes mapped warp threads to identical hardware banks
(2.0–2.1× serialization confirmed by Nsight). Fix: `[TILE_WIDTH][TILE_WIDTH + 1]`
stride padding redistributes access across banks without changing the algorithm.

### Occupancy-Limited GEMM: The Block Limit Story
At 16 threads/block (half a warp), the kernel was occupancy-limited before register
pressure or shared memory even mattered. After increasing to 64 threads/block, Nsight's
Block Limit breakdown showed shared memory as the new binding constraint (7 blocks/SM ×
8.25 KB = 57.75 KB of 64 KB; predicted ceiling 43.75%, measured 40.64%) — two
different bottleneck classes, each requiring a different fix, diagnosed by Block Limit
analysis rather than trial-and-error.

### Cholesky Dependency Chain vs. Throughput Bound
Both the diagonal factorization and column update kernels appear slow but are
dependency-chain-bound, not throughput-bound: duration is flat regardless of grid size
and throughput sits below 1% of peak. cuSolver's own equivalent leaf kernel
(`potrf_alg2_cta_upper`) shows the identical signature — this ceiling is intrinsic to
sequential small-tile Cholesky, not something left on the table by poor implementation.

### FP16 Trait Dispatch: Infrastructure Complete, WMMA Kernel Pending
`gemm_traits_template_FINAL.cu` establishes a complete compile-time dispatch
infrastructure for FP16: `gemm_traits<half>` specialization routing to fp32
accumulation (`compute_t = float`), `if constexpr` branching on `use_tensor_cores`,
and `static_assert`s formally verifying the routing at compile time. The actual WMMA
API (`nvcuda::wmma::fragment`) execution path is not yet implemented — the branch
compiles but is empty. This is the correct scaffold for adding a real tensor-core path
without touching the calling code or verification layer.

### Distributed SUMMA Root-Cause: Topology Before Code
When communication/compute overlap didn't materialize despite correct double-buffering,
the diagnostic path was `nvidia-smi topo -m`, not further code inspection. `SYS`
topology (GPUs on different NUMA sockets, no shared PCIe switch) produced host-staged
cross-socket transfers at ~1.4 GB/s rather than the direct GPU-to-GPU path the
architecture assumed. Same instinct as CPU-side NUMA cache locality — topology sets a
ceiling no software fix can raise.

### Node-Local Device Binding: Invisible Bug on Single-Node Hardware
`cudaSetDevice(world_rank)` works silently on any single-node test regardless of rank
count, because local and global ranks always coincide when there is only one machine.
On a real multi-node cluster, rank 2 on node B would call `cudaSetDevice(2)` but only
have devices 0 and 1 — immediate crash with no indication of the root cause.
`MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)` is the fix; single-node testing
structurally cannot expose the bug regardless of how thoroughly it is run.

---

## Repository Structure

```
cuda-kernel-acceleration/
├── blas-primitives/           # GEMM (naive → shared-mem-tiled → register-tiled →
│                              # occupancy-tuned → vs. cuBLAS), FP16 trait dispatch,
│                              # Matrix Transpose
├── parallel-primitives/       # Prefix Sum: Blelloch, Brent-Kung, warp shuffle
├── sparse-linear-algebra/     # CSR SpMV, cuSPARSE comparison, structured stencil
├── numerical-solvers/         # Batched LU, blocked Cholesky (vs. cuSolver),
│                              # TSQR, Conjugate Gradient, cuDSS refactorization
├── krylov-methods/            # 2D Conjugate Gradient for Poisson equations
├── dl-acceleration/           # Online Softmax, FlashAttention block tiling
├── asynchronous-streams/      # CUDA stream overlap experiments
├── distributed-computing/
│   ├── mpi-fundamentals/      # MPI process grids, block-cyclic layout, scatter/gather
│   └── nccl-multi-gpu/        # SUMMA GEMM (MPI+NCCL), cuBLASMp benchmark,
│                              # multi-node template; Nsight profiles in profile/
└── benchmarks/                # Nsight Compute CSV reports, roofline_analysis.png,
                               # get_flop.py, generate_roofline.py
```

---

## Compilation

```bash
# Single-GPU CUDA kernel
nvcc -O3 -arch=sm_75 numerical-solvers/LUFactorization.cu -o lu_solver
ncu --set full ./lu_solver

# FP16 trait dispatch kernel (gemm_traits_template)
nvcc -O3 -arch=sm_75 -std=c++17 blas-primitives/gemm_traits_template_FINAL.cu \
    -o gemm_traits -lcublas

# MPI + NCCL distributed GEMM (single-node multi-GPU)
cd distributed-computing/nccl-multi-gpu
nvcc summa_gemm.cu -o summa_gemm -O3 -arch=native -lcublas -lnccl -ccbin mpicxx
mpirun --allow-run-as-root -np 4 ./summa_gemm

# cuBLASMp benchmark (see file header for full env/install steps)
nvcc -arch=sm_75 summa_cublasmp.cu -o summa_cublasmp \
    -lcublas -lnccl -lcublasmp -lcudart -ccbin mpicxx
mpirun --allow-run-as-root -np 4 \
    -x LD_LIBRARY_PATH=$(pwd)/cublasmp_env/lib:$LD_LIBRARY_PATH \
    ./summa_cublasmp

# Roofline analysis (from repo root)
python3 benchmarks/get_flop.py
python3 benchmarks/generate_roofline.py
```

---

## Roadmap

- [x] GEMM: bank-conflict elimination → occupancy tuning → vs. cuBLAS
- [x] Blocked Cholesky vs. cuSolver `potrf`
- [x] Roofline / arithmetic intensity analysis across the full suite
- [x] cuDSS sparse direct solver integration and profiling
- [x] MPI 2D process grid + block-cyclic layout, scatter/gather verified
- [x] Distributed SUMMA GEMM on real 2-GPU hardware (topology finding documented)
- [x] Distributed SUMMA generalized to arbitrary M/N/K, Frobenius-norm verification
- [x] Multi-node device-binding fix (`MPI_COMM_TYPE_SHARED`) + AllReduce cross-node proof
- [x] cuBLASMp integration: 4-GPU correctness + end-to-end + kernel-level comparison
- [x] FP16 trait dispatch infrastructure: `gemm_traits<half>`, `if constexpr` routing,
      `static_assert` compile-time verification — **WMMA execution kernel not yet written**
- [ ] **WMMA tensor-core kernel** for FP16 path — scaffold is in place, fill the branch
- [ ] **Real 2-node Cluster run** — verify Phase 3 device-binding fix on actual separate machines
- [ ] **Multi-node bandwidth cliff measurement** — 1-node-4-GPU vs. 2-node-2-GPU, same problem
- [ ] Distributed blocked Cholesky, extending trailing-matrix update across GPUs
- [ ] Overlap validation on topology with NVLink or PIX interconnect (to confirm overlap
      architecture works when hardware doesn't impose a SYS-topology ceiling)
- [ ] cuSPARSE format comparison (CSR vs. BSR vs. ELL)
- [ ] Hopper (sm_90) architectural study: TMA, warpgroup MMA

---

## Environment

| Component | Details |
|---|---|
| Single-GPU dev | NVIDIA Tesla T4 (Turing, sm_75), Google Colab |
| Multi-GPU dev | RunPod Secure Cloud (2× RTX 2000 Ada; 4× RTX PRO 4500 Blackwell for cuBLASMp comparison) |
| Multi-node (pending) | RunPod Clusters |
| CUDA | 12.x–13.x depending on pod |
| Profilers | Nsight Compute (`ncu`), Nsight Systems (`nsys`) |
| Language | CUDA C++ (`.cu`), C++ with MPI (`.cpp`) |
| Compilers | `nvcc -O3 -arch=native`, `mpicxx` (OpenMPI) |
| Distributed libraries | OpenMPI, NCCL, cuBLASMp (conda-forge/nvidia channel) |
