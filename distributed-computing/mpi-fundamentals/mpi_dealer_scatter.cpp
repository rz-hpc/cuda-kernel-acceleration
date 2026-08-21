
///////////////////////////////////////////////////////////
// Compile and run as:
//!mpicxx mpi_dealer_scatter.cpp -o mpi_dealer_scatter
//!mpirun --allow-run-as-root --oversubscribe -n 4 ./mpi_dealer_scatter
//
// Output:
// 
//Rank 0 (Grid 0, 0) Local Matrix:
//    1     2     5     6 
//    9    10    13    14 
//   33    34    37    38 
//   41    42    45    46 
//
//Rank 1 (Grid 0, 1) Local Matrix:
//    3     4     7     8 
//   11    12    15    16 
//   35    36    39    40 
//   43    44    47    48 
//
//Rank 2 (Grid 1, 0) Local Matrix:
//   17    18    21    22 
//   25    26    29    30 
//   49    50    53    54 
//   57    58    61    62 
//
//Rank 3 (Grid 1, 1) Local Matrix:
//   19    20    23    24 
//   27    28    31    32 
//   51    52    55    56 
//   59    60    63    64 
///////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <iomanip>
#include <mpi.h>

struct LocalCoord {
    int rank_coord;
    int local_idx;
};

inline LocalCoord global_to_local(int global_idx, int nb, int p_dim) {
    int block_idx = global_idx / nb; // global block id
    int owner_coord = block_idx % p_dim; // this block's process id
    int local_block = block_idx / p_dim; // this block's local block id in the process
    int offset = global_idx % nb; // element th inside the block tile

    return {owner_coord, local_block * nb + offset};
}

struct Local2DCoord {
    int rank_row, rank_col;
    int local_row, local_col;
};

inline Local2DCoord global_to_local_2d(int g_row, int g_col, int nb, int P_r, int P_c) {
    LocalCoord r = global_to_local(g_row, nb, P_r);
    LocalCoord c = global_to_local(g_col, nb, P_c);

    return {r.rank_coord, c.rank_coord, r.local_idx, c.local_idx};
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 1. Setup 2D Cartesian Process Grid (2x2)
    int dims[2] = {2, 2};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);
    int rank_row = coords[0], rank_col = coords[1];

    // Matrix Parameters
    const int M = 8, N = 8;
    const int nb = 2;
    const int P_r = 2, P_c = 2;

    // Each process owns a 4x4 local array
    const int local_M = M / P_r;
    const int local_N = N / P_c;
    std::vector<float> local_matrix(local_M * local_N, 0.0f);

    // 2. Rank 0 Allocates and Scatters the Global Matrix
    if (world_rank == 0) {
        std::vector<float> global_matrix(M * N);
        for (int i = 0; i < M * N; i++) {
            global_matrix[i] = static_cast<float>(i + 1); // 1.0 to 64.0
        }

        // Distribute elements using index mapping
        for (int r = 0; r < M; r++) {
            for (int c = 0; c < N; c++) {
                float val = global_matrix[r * N + c];
                Local2DCoord target = global_to_local_2d(r, c, nb, P_r, P_c);

                int target_coords[2] = {target.rank_row, target.rank_col};
                int target_rank;
                MPI_Cart_rank(cart_comm, target_coords, &target_rank);

                int local_offset = target.local_row * local_N + target.local_col;

                if (target_rank == 0) {
                    local_matrix[local_offset] = val;
                }
                else {
                    // Send value and target local offset to owner rank
                    // payload: pack the metadata (destination index and the matrix value)
                    // here cast to int just for the prototype, HPC MPI requires float or double
                    int payload[2] = {local_offset, static_cast<int>(val)};
                    MPI_Send(payload, // memory address of data to send
                            2,  // number of elements to send
                            MPI_INT, // datatype
                            target_rank, // rank id of the reciever process
                            0, // tag/message identifier
                            cart_comm); // communicator universe
                }
            }
        }
    }
    else {
        // Receiver ranks collect their assigned elements
        int elements_to_receive = local_M * local_N;
        for (int i = 0; i < elements_to_receive; i++) {
            int payload[2];
            MPI_Recv(payload, // memory address to write the incoming data to
                      2, // max number of elements to receive
                      MPI_INT, // expected MPI data type
                      0, // message source (message from rank 0)
                      0, // tag/identifier to match sender's tag exactly
                      cart_comm, // communicator universe
                      MPI_STATUS_IGNORE);// metadata struct
            local_matrix[payload[0]] = static_cast<float>(payload[1]);
        }
    }

    // 3. Print Local Sub-matrices
    MPI_Barrier(cart_comm);
    for (int r = 0; r < world_size; r++) {
        if (world_rank == r) {
            std::cout << "\nRank " << world_rank << " (Grid " 
            << rank_row << ", " << rank_col << ") Local Matrix:\n";
            for (int i = 0; i < local_M; i++) {
                for (int j = 0; j < local_N; j++) {
                    std::cout << std::setw(5) << local_matrix[i * local_N + j] << " ";
                }
                std::cout << "\n";
            }
        }
        MPI_Barrier(cart_comm);
    }

    MPI_Finalize();

    return 0;
}

