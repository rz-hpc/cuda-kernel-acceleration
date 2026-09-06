
/*
Compile and Run:
# 0. (Optional) suppress compiler warning about deprecated architectures on Colab  -Wno-deprecated-gpu-targets -arch=sm_75
!nvcc -Wno-deprecated-gpu-targets -arch=sm_75 multinode_mpi_nccl_template.cu -o multinode_template -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lnccl

# 1. Compile directly using nvcc, linking MPI and NCCL libraries
#!nvcc multinode_mpi_nccl_template.cu -o multinode_template -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lnccl

# 2. Run a clean 1:1 rank-to-GPU test (no oversubscription needed for -np 1)
!mpirun --allow-run-as-root -np 1 ./multinode_template

# Example output on Colab:
#[af040d134073] Global Rank: 0/1 | Local Rank: 0/1 -> Bound to GPU 0

#[Status] Multi-node infrastructure initialized successfully. Ready for compute workloads.


##### On real multi-node env
# 2. Run across the multi-node RunPod cluster using a hostfile and network flags
#mpirun --hostfile hostfile \
#    --allow-run-as-root \
#    -np 4 \
#    -mca btl_tcp_if_include eth0 \
#    -x NCCL_SOCKET_IFNAME=eth0 \
#    -x NCCL_DEBUG=INFO \
#    ./multinode_template

*/

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h> // for gethostname

int main(int argc, char** argv) {
    // Phase 1: MPI Initialization & World Identity
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    // Phase 2: Local Node Isolation & GPU Binding
    // Group ranks sharing the same physical node (shared memory)
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm);

    int local_rank, local_size;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    // Safe device binding using local_rank (NOT world_rank!)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    int assigned_device = local_rank % num_gpus;
    cudaSetDevice(assigned_device);

    printf("[%s] Global Rank: %d/%d | Local Rank: %d/%d -> Bound to GPU %d\n", 
           hostname, world_rank, world_size, local_rank, local_size, assigned_device);

    // Phase 3: Cross-Node NCCL Initialization
    // The entire multi-node job has only one shared ncclUniqueId
    ncclUniqueId nccl_id;
    // Rank 0 acts as the coordinator to generate the unique ID
    if (world_rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }

    // Broadcast the unique ID across all network nodes using MPI
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);


    // handshake function where every GPU introduces itself to NCCL
    // Connect local GPU into the global NCCL communication mesh
    ncclComm_t nccl_comm;
    ncclCommInitRank(&nccl_comm, 
                      world_size, // total number of ranks (how many gpus are expected)
                      nccl_id, // the shared group ID so NCCL knows which cluster group to join
                      world_rank // the specific global identity, who is within that group
                      );

    // Phase 4: Core Workload / Algorithm Execution
    // TODO: Distributed logic here
    // (e.g., MPI Scatter/Gather for matrix chunks, NCCL broadcasts for SUMMA panels)

    if (world_rank == 0) {
        printf("\n[Status] Multi-node infrastructure initialized successfully. Ready for compute workloads.\n\n");
    }

    // Synchronize all nodes before exiting
    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 5: Clean Teardown
    ncclCommDestroy(nccl_comm);
    MPI_Comm_free(&local_comm);
    MPI_Finalize();

    return 0;
}
