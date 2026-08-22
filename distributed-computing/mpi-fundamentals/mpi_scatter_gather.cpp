
// Compile and Run with:
// !mpicxx mpi_scatter_gather.cpp -o mpi_scatter_gather
// !mpirun --allow-run-as-root --oversubscribe -n 4 ./mpi_scatter_gather

// Ouput:
// Ressembled and Computed Global Matrix on Rank 0:
//    10    20    30    40    50    60    70    80 
//    90   100   110   120   130   140   150   160 
//   170   180   190   200   210   220   230   240 
//   250   260   270   280   290   300   310   320 
//   330   340   350   360   370   380   390   400 
//   410   420   430   440   450   460   470   480 
//   490   500   510   520   530   540   550   560 
//   570   580   590   600   610   620   630   640 


#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

// Process is the actual physical OS-level entity running on the CPU
// Rank is the integer ID assigned to the process inside a communicator
struct LocalCoord {
    int rank_coord;
    int local_idx;
};

inline LocalCoord global_to_local(int global_idx, int nb, int p_dim) {
    int block_idx = global_idx / nb;
    int owner_coord = block_idx % p_dim;
    int local_block = block_idx / p_dim;
    int offset = global_idx % nb;

    return {owner_coord, local_block * nb + offset};
}

inline int local_to_global(int local_idx, int rank_coord, int nb, int p_dim) {
    int local_block = local_idx / nb;
    int offset = local_idx % nb;
    int global_block = local_block * p_dim + rank_coord;

    return global_block * nb + offset;
}

struct Local2DCoord {
    int rank_row, rank_col;
    int local_row, local_col;
};

struct Global2DCoord {
    int g_row, g_col;
};

inline Local2DCoord global_to_local_2d(int g_row, int g_col, int nb, int P_r, int P_c) {
    LocalCoord r = global_to_local(g_row, nb, P_r);
    LocalCoord c = global_to_local(g_col, nb, P_c);

    return {r.rank_coord, c.rank_coord, r.local_idx, c.local_idx};
}

inline Global2DCoord local_to_global_2d(int l_row, int l_col, 
                                        int rank_row, int rank_col,
                                        int nb, int P_r, int P_c) {
    int g_row = local_to_global(l_row, rank_row, nb, P_r);
    int g_col = local_to_global(l_col, rank_col, nb, P_c);
                                      
    return {g_row, g_col};
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int dims[2] = {2, 2};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 
                    2, // number of dimensions in the Cartesian grid
                    dims, // array specifying the number of processes in each dimension
                    periods, // logical array indicating whether the grid is periodic (true) or not in each dimension
                    1, // process ranks reordered flag
                    &cart_comm);// output communicator with the Cartesian topology

    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);
    int rank_row = coords[0], rank_col = coords[1];

    const int M = 8, N = 8, nb = 2, P_r = 2, P_c = 2;
    const int local_M = M / P_r, local_N = N / P_c;

    std::vector<float> local_matrix(local_M * local_N, 0.0f);

    // Phase 1: Scatter (Rank 0 -> All Ranks)
    if (world_rank == 0) {
        std::vector<float> global_matrix(M * N);
        for (int i = 0; i < M * N; i++) {
            global_matrix[i] = static_cast<float>(i + 1);
        }

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
        int elements_to_receive = local_M * local_N;
        for (int i = 0; i < elements_to_receive; i++) {
            int payload[2]; // destination index (local_offset), value
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

    // Phase 2: Compute (All Ranks)
    // Multiply every element by 10 to prove local processing happened
    for (int i = 0; i < local_M * local_N; i++) {
        local_matrix[i] *= 10.0f;
    }
    MPI_Barrier(cart_comm); // Ensure everyone finishes computing

    // Phase 3: Gather (All Ranks -> Rank 0)
    if (world_rank != 0) {
        // Workers calcualte where their elements belong in the global matrix and send them to Rank 0
        for (int lr = 0; lr < local_M; lr++) {
            for (int lc = 0; lc < local_N; lc++) {
                Global2DCoord g = local_to_global_2d(lr, lc, rank_row, rank_col, nb, P_r, P_c);
                int global_offset = g.g_row * N + g.g_col;
                float val = local_matrix[lr * local_N + lc];

                int payload[2] = {global_offset, static_cast<int>(val)};
                MPI_Send(payload, 2, MPI_INT, 0, 1, cart_comm); // Tag 1 for gather
            }
        }
    }
    else {
        std::vector<float> final_global_matrix(M * N, 0.0f);

        // 1. Rank 0 maps its own local memory back to global memory
        for (int lr = 0; lr < local_M; lr++) {
            for (int lc = 0; lc < local_N; lc++) {
                Global2DCoord g = local_to_global_2d(lr, lc, rank_row, rank_col, nb, P_r, P_c);
                int global_offset = g.g_row * N + g.g_col;
                final_global_matrix[global_offset] = local_matrix[lr * local_N + lc];
            }
        }

        // 2. Rank 0 catches messages from all other ranks (using MPI_ANY_SOURCE)
        int incoming_elements = (M * N) - (local_M * local_N);
        for (int i = 0; i < incoming_elements; i++) {
            int payload[2];
            MPI_Recv(payload, 2, MPI_INT, MPI_ANY_SOURCE, 1, cart_comm, MPI_STATUS_IGNORE);
            final_global_matrix[payload[0]] = static_cast<float>(payload[1]);
        }

        // Print final result
        std::cout << "\nRessembled and Computed Global Matrix on Rank 0:\n";
        for (int r = 0; r < M; r++) {
            for (int c = 0; c < N; c++) {
                std::cout << std::setw(5) << final_global_matrix[r * N + c] << " ";
            }
            std::cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}

