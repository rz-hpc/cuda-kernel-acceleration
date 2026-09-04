
// Compile and run:
// !mpicxx multinode_mpi_practice.cpp -o multinode_mpi_practice
// !mpirun --allow-run-as-root -np 1 ./multinode_mpi_practice

// On Google Colab, output:
// [1f8b44ba91da] world_rank=0/1  local_rank=0/1

    // ---- What to expect, and why this test can't fully prove itself on Colab ----
    //
    // SINGLE NODE (Colab, or any -np N run here): every rank shares the
    // same machine, so there's only ONE group. local_rank ends up
    // numerically identical to world_rank for every rank -- e.g. at
    // -np 4: world_rank/local_rank pairs are 0/0, 1/1, 2/2, 3/3.
    // This is indistinguishable from just using world_rank directly --
    // which is exactly why the ORIGINAL cudaSetDevice(world_rank) bug
    // was invisible on single-node testing. The bug only exists in a
    // regime this environment structurally cannot produce.
    //
    // MULTI-NODE (2 nodes x 2 ranks each, real Cluster test): now there
    // are TWO groups, one per node. local_rank RESETS to 0 on each node.
    // Expected output, unordered since ranks print independently:
    //   [node-A] world_rank=0/4  local_rank=0/2
    //   [node-A] world_rank=1/4  local_rank=1/2
    //   [node-B] world_rank=2/4  local_rank=0/2   <- local_rank REPEATS
    //   [node-B] world_rank=3/4  local_rank=1/2   <- local_rank REPEATS
    //
    // That repetition (local_rank hitting 0 twice, on two different
    // hostnames) is the actual proof the split worked correctly -- it's
    // the one signature this test can show on the real cluster that it
    // fundamentally cannot show here.


#include <mpi.h>
#include <cstdio>
#include <unistd.h> // for gethostname() -- which physical machine each rank actually runs on

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // world_rank: this rank's identity across the ENTIRE job, regardless
    // of how many physical machines it's spread across. Always unique
    // 0..(total_ranks-1), whether your're on 1 node or 10.
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // MPI_Comm_split_type with MPI_COMM_TYPE_SHARED groups ranks that
    // share memory -- in practice, this means "ranks on the same
    // physical node." Every rank calls this identically; MPI figures out
    // the grouping and hands each rank back a new communicator scoped to
    // just its own node's ranks.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, 
                        MPI_COMM_TYPE_SHARED, 
                        world_rank, 
                        MPI_INFO_NULL,
                        &local_comm);

    // local_rank: this rank's identity WITHIN its own node's group only.
    // local_size: how many ranks are sharing that node with it.
    int local_rank, local_size;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    printf("[%s] world_rank=%d/%d  local_rank=%d/%d\n",
           hostname, world_rank, world_size, local_rank, local_size);

    MPI_Comm_free(&local_comm);
    MPI_Finalize();

    return 0;
}
