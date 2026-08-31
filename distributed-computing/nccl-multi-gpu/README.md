# Distributed SUMMA GEMM — NCCL Multi-GPU

A from-scratch distributed GEMM (SUMMA — Scalable Universal Matrix Multiplication
Algorithm) across multiple GPUs, using MPI for process-grid topology and NCCL for
the broadcast-based communication pattern, with CUDA stream double-buffering for
communication/compute overlap. Built on top of the block-cyclic layout math and
scatter/gather work from `../mpi-fundamentals/`.

Verified correct on real 2-GPU hardware (RunPod, 2x RTX 2000 Ada). Overlap
architecture is implemented and synchronization-correct, but did not produce
measurable overlap on this specific pod — root-caused to GPU interconnect
topology, not a code defect. See **Analysis** below.

---

## Pod Setup

Container images ship with held/locked package versions that clash with the
host driver — expect to need `--allow-change-held-packages` most sessions.

**1. OpenMPI:**
```bash
apt-get update
apt-get install -y --allow-change-held-packages openmpi-bin libopenmpi-dev
```

**2. Align NCCL to the host's actual CUDA driver version** — check what's
available before installing, don't assume a version number carries over
from a previous session:
```bash
apt-cache policy libnccl2
apt-get install -y --allow-change-held-packages \
    libnccl2=2.26.2-1+cuda12.8 \
    libnccl-dev=2.26.2-1+cuda12.8
```

**3. Nsight Systems, if profiling** (not always bundled in CUDA devel images):
```bash
which nsys   # check first
apt-get install -y cuda-nsight-systems-12-8   # adjust version suffix to match this pod's CUDA
```

**4. Diagnostics worth running immediately after boot, before writing or
running any code** — cheap, and answers questions that are expensive to
debug into later:
```bash
nvidia-smi topo -m    # GPU interconnect: PIX/NV# = real P2P, SYS = host-routed (slow)
nvcc --version          # confirm actual CUDA version on this specific pod
```

---

## Compile

```bash
nvcc summa_gemm.cu -o summa_gemm -O3 -arch=native -lcublas -lnccl -ccbin mpicxx
```

`-arch=native` auto-detects whatever GPU is actually present on the pod —
**deliberately not** a hardcoded `-gencode arch=compute_89,code=sm_89`
(Ada-specific). GPU availability on rented pods varies session to session
(this project has seen Ada, Ampere A4000/A5000/A40 availability shift day to
day); a hardcoded architecture flag would silently target the wrong
generation the moment the hardware changes. `mpicxx` as the host compiler
resolves MPI include/lib paths automatically — no manual `-I`/`-L` needed.

---

## Run

```bash
mpirun --allow-run-as-root -np <N> ./summa_gemm
```

`<N>` should match the pod's actual GPU count. The code is portable across
grid shapes via `MPI_Dims_create` (auto-factors `N` processes into a
balanced P×Q grid) + `MPI_Comm_size` (reads the real communicator size
rather than assuming a fixed `2`) — no source changes needed between a
2-GPU and a 4-GPU pod, just the `-np` value.

**Correctness check is built in**: each rank fills synthetic values
depending on both `rank_row` and `rank_col` (not just one axis — a
single-axis version can't detect a wrong-root broadcast, since same-row
ranks would already share identical values regardless of which one NCCL
actually picked as root). Each rank independently computes its expected
`C` value on the host and compares against the real post-SUMMA GPU result.

---

## Profiling

```bash
mpirun --allow-run-as-root -np <N> nsys profile -o summa_profile_rank%q{OMPI_COMM_WORLD_RANK} ./summa_gemm
```

Generate readable summaries from the resulting `.nsys-rep` files:
```bash
nsys stats --report cuda_gpu_trace summa_profile_rank0.nsys-rep > gpu_trace_rank0.txt
nsys stats --report cuda_api_sum summa_profile_rank0.nsys-rep > api_summary_rank0.txt
nsys stats --report cuda_gpu_kern_sum summa_profile_rank0.nsys-rep > kernel_summary_rank0.txt
nsys stats --report gpu_gaps summa_profile_rank0.nsys-rep > gpu_gaps_rank0.txt
nvidia-smi topo -m > gpu_topology.txt
```

**Before terminating any pod**, pull everything off first — container disk
is wiped on termination:
```bash
# from LOCAL machine, one scp per file (avoid wildcard globbing with zsh,
# it expands locally before reaching the remote host and fails):
scp -P <port> root@<pod-ip>:/path/to/file.ext .
```
If SSH prompts for a password and none was set: many RunPod images disable
password auth entirely. Generate a local key (`ssh-keygen -t ed25519`), add
the public key to the pod via its **web terminal** (not scp, which needs
the key to already work):
```bash
mkdir -p ~/.ssh && echo "<paste public key>" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```
Retry `scp` from local — should authenticate without a password.

Confirm every file landed locally with a non-zero size matching the pod's
`ls -la` before terminating. Decline to keep an attached network volume
unless something genuinely persistent was written to `/workspace`
specifically — work done in other paths (e.g. a cloned repo elsewhere)
lives on the container disk and is gone at termination regardless of
volume choice.

---

## Results

**Environment**: RunPod Secure Cloud, 2x NVIDIA RTX 2000 Ada Generation,
`-np 2`, `Nb=1024` (1024×1024 tiles), `K_blocks=4`.

**Correctness — PASS on both ranks:**

| Rank | Grid position | Expected | Max abs error | Relative error | Verified |
|---|---|---|---|---|---|
| 0 | (row=0, col=0) | 73,865,216.00 | 8.0 | 1.08e-07 | 1 |
| 1 | (row=1, col=0) | 140,015,616.00 | 16.0 | 1.14e-07 | 1 |

Both hand-verified against the synthetic-value formula independently (see
correctness-check derivation in project history) — errors are consistent
with fp32 rounding noise at this magnitude, not a logic bug.

**Communication/compute overlap — not observed:**

| Op | Stream | Start (ns) | End (ns) | Gap before next op |
|---|---|---|---|---|
| Broadcast | 20 | 3,091,067,473 | 3,093,946,351 | — |
| sgemm | 19 | 3,136,489,335 | 3,136,841,523 | 42.5ms after prior broadcast |
| Broadcast | 20 | 3,136,836,691 | 3,139,780,496 | starts 4.8µs before prior sgemm ends |
| sgemm | 19 | 3,139,814,351 | 3,140,174,411 | 33.9µs gap |
| Broadcast | 20 | 3,140,167,819 | 3,143,429,221 | starts 6.6µs before prior sgemm ends |
| sgemm | 19 | 3,143,460,772 | 3,143,789,408 | 31.6µs gap |
| Broadcast | 20 | 3,143,780,992 | 3,146,715,294 | starts 8.4µs before prior sgemm ends |
| sgemm | 19 | 3,146,722,525 | 3,147,047,322 | 7.2µs gap |

The 5-8µs apparent "overlaps" are launch-scheduling noise against
operations lasting 300-3000µs — not meaningful concurrent execution.
Broadcast and compute ran essentially back-to-back despite correct
double-buffering, separate streams, and per-buffer event synchronization.

**Broadcast throughput**: each broadcast moves one 1024×1024 float tile
(4.19 MB) in ~2.9-3.3ms → **~1.4 GB/s effective throughput** — far below
PCIe Gen4 x16's 25+ GB/s, and consistent with a host-staged transfer path
rather than direct GPU-to-GPU DMA.

**Topology** (`nvidia-smi topo -m`):
```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      SYS     0-31            0
GPU1    SYS      X      32-63           1
```

---

## Analysis: why no overlap, and the NUMA connection

### The root cause

`SYS` is the worst case on `nvidia-smi`'s topology legend — it means the
only path between GPU0 and GPU1 traverses PCIe *and* the host's
socket-to-socket interconnect (QPI/UPI on this Intel-class host). Confirmed
by the CPU/NUMA affinity columns: GPU0 sits on CPU cores 0-31 (NUMA node
0), GPU1 on cores 32-63 (NUMA node 1) — **the two GPUs are attached to
different CPU sockets entirely**, with no NVLink and no shared PCIe switch
bridging them directly.

Every byte moved between GPU0 and GPU1 has to: leave GPU0 over PCIe to its
local CPU socket, cross the inter-socket interconnect to the other socket,
then continue over PCIe to GPU1 (and the reverse for any response/sync
traffic) — versus a direct `PIX` (single PCIe bridge) or `NV#` (NVLink)
path, which would stay on one, much faster hop. This fully explains both
findings: the ~1.4 GB/s throughput (consistent with cross-socket
host-staged transfer, not raw PCIe bandwidth) and the lack of overlap (the
communication itself is so slow and structurally expensive that there's
comparatively little useful compute-side work to hide behind it, and what
compute exists finishes before the next broadcast is even close to ready).

**This is a provisioning outcome, not a code defect.** The synchronization
logic, correctness, and double-buffering architecture are all confirmed
sound — verified independently by the correctness check above. Given a pod
with `PIX` or `NV#` topology between GPUs, the same code should show real
overlap; that's a natural follow-up experiment when topology allows it.

### The NUMA connection — same failure mode, one layer up the stack

This is architecturally the same problem as CPU-side NUMA, which I worked
directly at Microsoft: on a multi-socket server, each CPU socket has its
own local memory bank, and a thread accessing memory attached to a
*different* socket pays a real latency and bandwidth penalty crossing the
inter-socket interconnect — the same QPI/UPI-class link, incidentally,
that's now the bottleneck here. The Analysis Services work involved tuning
memory layout specifically to avoid cross-thread cache "ping-ponging"
across that boundary — keeping a thread's working set local to the NUMA
node it was actually scheduled on, rather than assuming uniform memory
access latency across the whole machine.

The diagnostic instinct transfers directly, even though the mechanism is
different: **don't assume a fast path exists — check the topology, then
design around what's actually there.** On the CPU side that meant thread
affinity and NUMA-aware allocation; on the GPU side here it would mean
either (a) requesting a pod with GPUs on the same PCIe switch or NVLink
bridge at provisioning time, since topology isn't something you can fix
after the fact in software, or (b) if stuck with a `SYS`-topology pod,
minimizing cross-GPU communication volume deliberately rather than assuming
NCCL's broadcast is "free" the way it would be on a well-connected node.

This is also, concretely, part of why cuBLASMp/cuSolverMp exist rather than
everyone hand-rolling SUMMA: production distributed math libraries build
topology-aware communication scheduling internally — detecting `SYS` vs.
`NV#` paths and adjusting ring/tree communication patterns accordingly —
exactly the class of problem surfaced here by hand. Worth revisiting this
comparison directly once Phase 3/4 gets to benchmarking against those
libraries: does cuBLASMp handle this same `SYS`-topology pod any better,
or does it hit the identical physical ceiling? That's a genuinely open,
interesting question this finding sets up.

---

## Known scope limits (not yet done)

- `d_A_local`/`d_B_local` currently hold synthetic per-rank values, not
  data sourced from a real matrix via `global_to_local`/`local_to_global`
  (Phase 1). Current results verify the *distributed communication and
  compute pipeline*, not an end-to-end real-matrix SUMMA multiply — that
  integration is a separate follow-on step.
- Only tested at `-np 2` (`dims={2,1}`) — one broadcast axis (column, size
  2) was genuinely exercised; the other (row, size 1) was trivial. A 4-GPU
  run would be the first to exercise both axes for real.
- Overlap has not been observed on any topology yet — worth retesting on a
  pod with `PIX`/`NV#` interconnect if one becomes available, to confirm
  the architecture actually delivers overlap when the hardware allows it.
