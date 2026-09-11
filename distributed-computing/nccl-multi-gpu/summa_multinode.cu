
/*
// Compile and Run:
!nvcc -Wno-deprecated-gpu-targets summa_multinode.cu -o summa_multinode -lcublas -lnccl -ccbin mpicxx -arch=native

!NCCL_DEBUG=WARN mpirun --allow-run-as-root -np 1 ./summa_multinode

# Run on multi-node RunPod:
# mpirun --hostfile hostfile --allow-run-as-root -np <TOTAL_GPUS> \
#     -mca btl_tcp_if_include eth0 \
#     -x NCCL_SOCKET_IFNAME=eth0 \
#     -x NCCL_DEBUG=INFO \
#     ./summa_multinode

// Example output on Colab:
NCCL version 2.25.1+cuda12.8
Initializing multi-node SUMMA across 1x1 Grid.
Frobenius norm ratio ||C_dist - C_ref|| / ||C_ref||: 9.65322e-07
VERIFICATION: PASS

// Timing -np 1:
Custom SUMMA Execution Time: 36.5726 ms (2.00697 TFLOPS)

// on Google Colab, can't run with --oversubscribe -np 4, error:
06e811e0e6df:8700:8700 [0] init.cc:720 NCCL WARN Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 40
free(): double free detected in tcache 2

*/

#include <mpi.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unistd.h> // for hostname
#include <cstring>
#include <random>

// Error Checking Macros
#define CHECK_CUDA(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        printf("CUDA Error %s: %d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_NCCL(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        printf("CUDA Error %s: %d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(cmd) do { \
    cublasStatus_t s = cmd; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS error %s: %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// MPI Index Mapping
struct LocalCoord {
    int rank_coord;
    int local_idx;
};

struct Local2DCoord {
    int rank_row, rank_col;
    int local_row, local_col;
};

struct Global2DCoord {
    int g_row, g_col;
};

// Example: nb = 2, p_dim = 3
// global index 0 1 2 3 4 5 6 7 8 9
// global block 0 0 1 1 2 2 3 3 4 4
// owner rank   0 0 1 1 2 2 0 0 1 1
// offset       0 1 0 1 0 1 0 1 0 1
// local block  0 0 0 0 0 0 1 1 1 1
// local index  0 1 0 1 0 1 2 3 2 3
// local index = lobal_block * nb + offset
// global block = local block * p_dim + owner_rank 

__host__ __device__ inline LocalCoord global_to_local(int global_idx, int nb, int p_dim) {
    int block_idx = global_idx / nb;
    int offset = global_idx % nb;
    int owner_rank = block_idx % p_dim;
    int local_block = block_idx / p_dim;

    return {owner_rank, local_block * nb + offset};
}

__host__ __device__ inline int local_to_global(int local_idx, int rank_coord, int nb, int p_dim) {
    int local_block = local_idx / nb;
    int offset = local_idx % nb;
    int global_block = local_block * p_dim + rank_coord;

    return global_block * nb + offset;
}

__host__ __device__ inline Local2DCoord global_to_local_2d(int g_row, int g_col, int nb, int P_r, int P_c) {
    LocalCoord r = global_to_local(g_row, nb, P_r);
    LocalCoord c = global_to_local(g_col, nb, P_c);

    return {r.rank_coord, c.rank_coord, r.local_idx, c.local_idx};
}

__host__ __device__ inline Global2DCoord local_to_global_2d(int l_row, int l_col, int rank_row, int rank_col, int nb, int P_r, int P_c) {
    int g_row = local_to_global(l_row, rank_row, nb, P_r);
    int g_col = local_to_global(l_col, rank_col, nb, P_c);

    return {g_row, g_col};
}

// Structure to track local matrix dimensions and allocation strides
struct LocalMatrixDim {
    int local_rows;
    int local_cols;
    int alloc_rows;
    int alloc_cols;
    int num_blocks_row;
    int num_blocks_col;
};

// Compute local allocation and valid dimensions under 2D block-cyclic layout
inline LocalMatrixDim get_local_matrix_dim(int G_M, int G_N, 
                                          int nb, int P_r, int P_c,
                                          int rank_row, int rank_col) {
    int total_blocks_m = (G_M + nb - 1) / nb;
    int total_blocks_n = (G_N + nb - 1) / nb;

    // total_blocks_m % P_r is the remainder:
    // extra blocks to hand out after every rank gets total_blocks_m / P_r
    // For example, if total_blocks_m = 10 and P_r = 4
    // Every rank gets 10/4 = 2 blocks
    // then there are 10 % 4 = 2 extra blocks to hand out (round robin)
    // rank_row 0: 0 < 2 --> rank_row 0 gets 3 blocks (2 base + 1 extra)
    // rank_row 1: 1 < 2 --> rank_row 1 gets 3 blocks (2 base + 1 extra)
    // rank_row 2: 2 == 2 --> rank_row 2 gets 2 base blocks
    // rank_row 3: 3 > 2 --> rank_row 3 gets 2 base blocks
    int blocks_r = total_blocks_m / P_r + (rank_row < (total_blocks_m % P_r) ? 1 : 0);
    int blocks_c = total_blocks_n / P_c + (rank_col < (total_blocks_n % P_c) ? 1 : 0);

    int valid_r = 0;
    for (int b = rank_row; b < total_blocks_m; b += P_r) {
        int rows_in_block = std::min(nb, G_M - b * nb);
        valid_r += rows_in_block;
    }

    int valid_c = 0;
    for (int b = rank_col; b < total_blocks_n; b += P_c) {
        int cols_in_block = std::min(nb, G_N - b * nb);
        valid_c += cols_in_block;
    }

    return {
        valid_r,
        valid_c,
        blocks_r * nb,
        blocks_c * nb,
        blocks_r,
        blocks_c
    };
}

// Distributed Data Ingestion and Extration
void distribute_matrix(const float* global_mat, float* local_mat,
                      int G_Rows, int G_Cols, int nb,
                      int P_r, int P_c, int rank_row, int rank_col,
                      LocalMatrixDim dim, MPI_Comm cart_comm, 
                      int world_rank, int world_size) {
    std::memset(local_mat, 0, dim.alloc_rows * dim.alloc_cols * sizeof(float));

    // Rank 0 coordinator node translates every process's 1D rank id into 2d grid coordinates
    if (world_rank == 0) {
        for (int p = 0; p < world_size; p++) { // every single process
            int p_coords[2]; // 2D position
            // take linear rank p and look up its position inside the 2d Cartesian grid cart_comm
            MPI_Cart_coords(cart_comm, p, 2, p_coords);
            int pr = p_coords[0];
            int pc = p_coords[1];

            LocalMatrixDim p_dim = get_local_matrix_dim(G_Rows, G_Cols, nb, P_r, P_c, pr, pc);
            std::vector<float> pack_buf(p_dim.alloc_rows * p_dim.alloc_cols, 0.0f);

            for (int lr = 0; lr < p_dim.local_rows; lr++) {
                for (int lc = 0; lc < p_dim.local_cols; lc++) {
                    Global2DCoord g = local_to_global_2d(lr, lc, pr, pc, nb, P_r, P_c);
                    if (g.g_row < G_Rows && g.g_col < G_Cols) {
                        pack_buf[lr * p_dim.alloc_cols + lc] = global_mat[g.g_row * G_Cols + g.g_col];
                    }
                }
            }

            if (p == 0) {
                std::memcpy(local_mat, pack_buf.data(), pack_buf.size() * sizeof(float));
            }
            else {
                MPI_Send(pack_buf.data(), pack_buf.size(), MPI_FLOAT, p, 0, cart_comm);
            }
        }
    }
    else { // other ranks
        MPI_Recv(local_mat, dim.alloc_rows * dim.alloc_cols, MPI_FLOAT, 0, 0, cart_comm, MPI_STATUS_IGNORE);
    }
}

void gather_matrix(float* global_mat, const float* local_mat,
                  int G_Rows, int G_Cols, int nb,
                  int P_r, int P_c, int rank_row, int rank_col,
                  LocalMatrixDim dim, MPI_Comm cart_comm,
                  int world_rank, int world_size) {
    if (world_rank == 0) {
        for (int p = 0; p < world_size; p++) {
            int p_coords[2];
            MPI_Cart_coords(cart_comm, p, 2, p_coords);
            int pr = p_coords[0];
            int pc = p_coords[1];

            LocalMatrixDim p_dim = get_local_matrix_dim(G_Rows, G_Cols, nb, P_r, P_c, pr, pc);
            std::vector<float> recv_buf(p_dim.alloc_rows * p_dim.alloc_cols, 0.0f);

            if (p == 0) {
                std::memcpy(recv_buf.data(), local_mat, p_dim.alloc_rows * p_dim.alloc_cols * sizeof(float));
            }
            else {
                MPI_Recv(recv_buf.data(), p_dim.alloc_rows * p_dim.alloc_cols, MPI_FLOAT, p, 1, cart_comm, MPI_STATUS_IGNORE);
            }

            for (int lr = 0; lr < p_dim.local_rows; lr++) {
                for (int lc = 0; lc < p_dim.local_cols; lc++) {
                    Global2DCoord g = local_to_global_2d(lr, lc, pr, pc, nb, P_r, P_c);
                    if (g.g_row < G_Rows && g.g_col < G_Cols) {
                        global_mat[g.g_row * G_Cols + g.g_col] = recv_buf[lr * p_dim.alloc_cols + lc];
                    }
                }
            }
        }
    }
    else {
        MPI_Send(local_mat, dim.alloc_rows * dim.alloc_cols, MPI_FLOAT, 0, 1, cart_comm);
    }
}

// Main Routine

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        world_rank,
                        MPI_INFO_NULL,
                        &local_comm);

    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);

    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    CHECK_CUDA(cudaSetDevice(local_rank % num_gpus));

    // Define the shape (number of rows and columns) in 2D progress grid
    // MPI_Dims_create
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    int P_r = dims[0];
    int P_c = dims[1];

    // whether Cartesian grid wraps around like torus (ring topology) in each dimension
    // 0 -- false, non-periodic, linear boundaries
    // 1 -- true, periodic, shifting data off the edge wraps it around to the other side
    // SUMMA does standard point-to-point broadcasts down rows and columns, no edge wrapping
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    // the exact 2D coordinates (row, col) of the current process
    // MPI_Cart_coords
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);
    int rank_row = coords[0];
    int rank_col = coords[1];

    // Split global 2D Cartesian grid into smaller independent sub-communicators
    // MPI_Comm_split(parent_comm, color, key, &new_comm)
    // color: tell MPI which group a process belongs to
    // key: tell MPI how to order/rank process inside the new sub-communicator
    // Processes with smaller key values get lower ranks within the new group
    MPI_Comm row_comm, col_comm;
    // within each row group, processes are ordered based on their column position
    MPI_Comm_split(cart_comm, rank_row, rank_col, &row_comm);
    MPI_Comm_split(cart_comm, rank_col, rank_row, &col_comm);

    int row_comm_rank, col_comm_rank;
    MPI_Comm_rank(row_comm, &row_comm_rank);
    MPI_Comm_rank(col_comm, &col_comm_rank);

    ncclUniqueId row_id, col_id;
    if (row_comm_rank == 0) ncclGetUniqueId(&row_id);
    if (col_comm_rank == 0) ncclGetUniqueId(&col_id);

    MPI_Bcast(&row_id, sizeof(row_id), MPI_BYTE, 0, row_comm);
    MPI_Bcast(&col_id, sizeof(col_id), MPI_BYTE, 0, col_comm);

    int row_comm_size, col_comm_size;
    MPI_Comm_size(row_comm, &row_comm_size);
    MPI_Comm_size(col_comm, &col_comm_size);

    ncclComm_t nccl_row_comm, nccl_col_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_row_comm, row_comm_size, row_id, row_comm_rank));
    CHECK_NCCL(ncclCommInitRank(&nccl_col_comm, col_comm_size, col_id, col_comm_rank));

    const int Global_M = 3500;
    const int Global_N = 2048;
    const int Global_K = 5120;
    const int Nb = 1024;
    const int K_blocks = (Global_K + Nb - 1) / Nb;

    LocalMatrixDim dim_C = get_local_matrix_dim(Global_M, Global_N, Nb, P_r, P_c, rank_row, rank_col);
    LocalMatrixDim dim_A = get_local_matrix_dim(Global_M, Global_K, Nb, P_r, P_c, rank_row, rank_col);
    LocalMatrixDim dim_B = get_local_matrix_dim(Global_K, Global_N, Nb, P_r, P_c, rank_row, rank_col);

    // Host allocation for local partitioned data
    std::vector<float> h_A_local(dim_A.alloc_rows * dim_A.alloc_cols, 0.0f);
    std::vector<float> h_B_local(dim_B.alloc_rows * dim_B.alloc_cols, 0.0f);
    std::vector<float> h_C_local(dim_C.alloc_rows * dim_C.alloc_cols, 0.0f);

    std::vector<float> global_A, global_B, global_C_verify;

    if (world_rank == 0) {
        printf("Initializing multi-node SUMMA across %dx%d Grid.\n", P_r, P_c);
        global_A.resize(Global_M * Global_K);
        global_B.resize(Global_K * Global_N);
        global_C_verify.resize(Global_M * Global_N, 0.0f);

        // Generate arbitrary matrix data on root node
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (auto& val : global_A) val = dis(gen);
        for (auto& val : global_B) val = dis(gen);
    }

    // Scatter arbitrary data from Rank 0 to local buffers
    distribute_matrix(global_A.data(), h_A_local.data(), Global_M, Global_K, Nb, P_r, P_c, rank_row, rank_col, dim_A, cart_comm, world_rank, world_size);
    distribute_matrix(global_B.data(), h_B_local.data(), Global_K, Global_N, Nb, P_r, P_c, rank_row, rank_col, dim_B, cart_comm, world_rank, world_size);

    float *d_C_local, *d_A_local, *d_B_local;
    CHECK_CUDA(cudaMalloc(&d_C_local, dim_C.alloc_rows * dim_C.alloc_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A_local, dim_A.alloc_rows * dim_A.alloc_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_local, dim_B.alloc_rows * dim_B.alloc_cols * sizeof(float)));
    
    // Copy local assigned data to GPU
    CHECK_CUDA(cudaMemcpy(d_A_local, h_A_local.data(), dim_A.alloc_rows * dim_A.alloc_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_local, h_B_local.data(), dim_B.alloc_rows * dim_B.alloc_cols * sizeof(float), cudaMemcpyHostToDevice));    
    CHECK_CUDA(cudaMemset(d_C_local, 0, dim_C.alloc_rows * dim_C.alloc_cols * sizeof(float)));

    // Double buffering (overlapped computation and communication)
    float *d_A_recv[2], *d_B_recv[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMalloc(&d_A_recv[i], dim_A.alloc_rows * Nb * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_recv[i], Nb * dim_B.alloc_cols * sizeof(float)));
    }

    cudaStream_t compute_stream, comm_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&comm_stream));
    cudaEvent_t compute_done[2], comm_done[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaEventCreate(&compute_done[i]));
        CHECK_CUDA(cudaEventCreate(&comm_done[i]));
    }

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));

    float alpha = 1.0f, beta = 1.0f;

    // SUMMA 
    
    // For step k, the process column for broadcasting panel A is k % P_c
    // The process row responsible for broadcasting panel B is k % P_r
    int current_buf = 0;
    int root_A = 0 % P_c; // column 0
    int root_B = 0 % P_r; // row 0
    int current_kb_0 = std::min(Nb, Global_K - 0 * Nb);

    // Matrix A requires a 2D strided Copy
    // because A is partitioned along K dimension horizontally
    // To extra a vertical strip of width current_kb from d_A_local,
    // have to skip across the full width of local allocation dim_A.alloc_cols
    if (rank_col == root_A) {
        int local_k_A = 0 / P_c;
        float* src_A = d_A_local + (local_k_A * Nb);
        CHECK_CUDA(cudaMemcpy2DAsync(d_A_recv[0], current_kb_0 * sizeof(float),
                                    src_A, dim_A.alloc_cols * sizeof(float),
                                    current_kb_0 * sizeof(float), dim_A.alloc_rows,
                                    cudaMemcpyDeviceToDevice, comm_stream));
    }
    // Matrix B uses a float contiguous Copy
    // because B is partitioned along K vertically
    // the entire horizontal block-row of height current_kb sits contigously in mempry
    if (rank_row == root_B) {
        int local_k_B = 0 / P_r;
        float* src_B = d_B_local + (local_k_B * Nb * dim_B.alloc_cols);
        CHECK_CUDA(cudaMemcpyAsync(d_B_recv[0], src_B, current_kb_0 * dim_B.alloc_cols * sizeof(float), cudaMemcpyDeviceToDevice, comm_stream));
    }

    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclBroadcast((const void*)d_A_recv[0], (void*)d_A_recv[0], dim_A.alloc_rows * Nb, ncclFloat, root_A, nccl_row_comm, comm_stream));
    CHECK_NCCL(ncclBroadcast((const void*)d_B_recv[0], (void*)d_B_recv[0], Nb * dim_B.alloc_cols, ncclFloat, root_B, nccl_col_comm, comm_stream));    
    CHECK_NCCL(ncclGroupEnd());
    CHECK_CUDA(cudaEventRecord(comm_done[0], comm_stream));

    // --- START TIMING ---
    MPI_Barrier(cart_comm);
    CHECK_CUDA(cudaDeviceSynchronize());
    double start_time = MPI_Wtime();

    for (int k = 0; k < K_blocks; k++) {
        int next_buf = (current_buf + 1) % 2;
        int current_kb = std::min(Nb, Global_K - k * Nb);

        CHECK_CUDA(cudaStreamWaitEvent(compute_stream, comm_done[current_buf], 0));

        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                dim_C.alloc_cols, dim_C.alloc_rows, current_kb,
                                &alpha, d_B_recv[current_buf], dim_B.alloc_cols,
                                d_A_recv[current_buf], Nb,
                                &beta, d_C_local, dim_C.alloc_cols));
        CHECK_CUDA(cudaEventRecord(compute_done[current_buf], compute_stream));

        if (k + 1 < K_blocks) {
            int next_k = k + 1;
            int next_kb = std::min(Nb, Global_K - next_k * Nb);
            int next_root_A = next_k % P_c;
            int next_root_B = next_k % P_r;

            CHECK_CUDA(cudaStreamWaitEvent(comm_stream, compute_done[next_buf], 0));

            if (rank_col == next_root_A) {
                int local_k_A = next_k / P_c;
                float* src_A = d_A_local + (local_k_A * Nb);
                CHECK_CUDA(cudaMemcpy2DAsync(d_A_recv[next_buf], Nb * sizeof(float),
                                            src_A, dim_A.alloc_cols * sizeof(float),
                                            next_kb * sizeof(float), dim_A.alloc_rows,
                                            cudaMemcpyDeviceToDevice, comm_stream));
            }
            if (rank_row == next_root_B) {
                int local_k_B = next_k / P_r;
                float* src_B = d_B_local + (local_k_B * Nb * dim_B.alloc_cols);
                CHECK_CUDA(cudaMemcpyAsync(d_B_recv[next_buf], src_B, next_kb * dim_B.alloc_cols * sizeof(float), cudaMemcpyDeviceToDevice, comm_stream));
            }

            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclBroadcast((const void*)d_A_recv[next_buf], (void*)d_A_recv[next_buf], dim_A.alloc_rows * Nb, ncclFloat, next_root_A, nccl_row_comm, comm_stream));
            CHECK_NCCL(ncclBroadcast((const void*)d_B_recv[next_buf], (void*)d_B_recv[next_buf], Nb * dim_B.alloc_cols, ncclFloat, next_root_B, nccl_col_comm, comm_stream));
            CHECK_NCCL(ncclGroupEnd());
            CHECK_CUDA(cudaEventRecord(comm_done[next_buf], comm_stream));
        }
        current_buf = next_buf;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // --- END TIMING ---
    double end_time = MPI_Wtime();
    if (world_rank == 0) {
        double elapsed_ms = (end_time - start_time) * 1000.0;
        double tflops = (2.0 * (double)Global_M * (double)Global_N * (double)Global_K) / (elapsed_ms * 1e9);
        std::cout << "Custom SUMMA Execution Time: " << elapsed_ms << " ms (" << tflops << " TFLOPS)\n";
    }

    CHECK_CUDA(cudaMemcpy(h_C_local.data(), d_C_local, dim_C.alloc_rows * dim_C.alloc_cols * sizeof(float), cudaMemcpyDeviceToHost));

    // Gather distributed blocks back into global C matrix on Rank 0
    gather_matrix(global_C_verify.data(), h_C_local.data(), Global_M, Global_N, Nb, P_r, P_c, rank_row, rank_col, dim_C, cart_comm, world_rank, world_size);

    if (world_rank == 0) {
        float *d_A_ref, *d_B_ref, *d_C_ref;
        CHECK_CUDA(cudaMalloc(&d_A_ref, Global_M * Global_K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_ref, Global_K * Global_N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_ref, Global_M * Global_N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A_ref, global_A.data(), Global_M * Global_K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B_ref, global_B.data(), Global_K * Global_N * sizeof(float), cudaMemcpyHostToDevice));    
        CHECK_CUDA(cudaMemset(d_C_ref, 0, Global_M * Global_N * sizeof(float)));   

        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                Global_N, Global_M, Global_K,
                                &alpha, d_B_ref, Global_N, d_A_ref, Global_K,
                                &beta, d_C_ref, Global_N));

        std::vector<float> h_C_ref(Global_M * Global_N);
        CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C_ref, Global_M * Global_N * sizeof(float), cudaMemcpyDeviceToHost));

        /*
        // Use standard combined absolute + relative tolerance (numpy.allclose uses)
        // passes when the difference is small in either absolute or relative terms
        // so near-zero references stop causing false failures
        const float atol = 1e-3f;
        const float rtol = 1e-3f;
        float max_err = 0.0f;
        float max_rel_err = 0.0f;
        bool all_pass = true;
        for (int i = 0; i < Global_M * Global_N; i++) {
            float abs_diff = std::fabs(global_C_verify[i] - h_C_ref[i]);
            max_err = std::max(max_err, abs_diff);
            float rel_diff = abs_diff / (std::fabs(h_C_ref[i]) + 1e-8f);
            max_rel_err = std::max(max_rel_err, rel_diff);
            if (abs_diff > atol + rtol * std::fabs(h_C_ref[i])) all_pass = false;
        }
        std::cout << "Max Absolute Error: " << max_err << "\n";
        std::cout << "Max Relative Error: " << max_rel_err << " (may be inflated by near-zero entries)\n";
        std::cout << (all_pass ? "VERIFICATION: PASS" : "VERIFICATION: FAIL") << std::endl;
        */

        // The Frobenius-norm ratio verification (single global norm ratio)
        double norm_diff_sq = 0.0, norm_ref_sq = 0.0;
        for (int i = 0; i < Global_M * Global_N; i++) {
            double diff = global_C_verify[i] - h_C_ref[i];
            norm_diff_sq += diff * diff;
            norm_ref_sq += (double)h_C_ref[i] * h_C_ref[i];
        }
        double norm_ratio = std::sqrt(norm_diff_sq) / std::sqrt(norm_ref_sq);

        std::cout << "Frobenius norm ratio ||C_dist - C_ref|| / ||C_ref||: " << norm_ratio << "\n";
        // Typical accepted threshold scales with K and machine epsilon:
        double threshold = std::sqrt((double)Global_K) * 1e-6;  // ~sqrt(K)*eps, generous margin
        std::cout << (norm_ratio < threshold ? "VERIFICATION: PASS" : "VERIFICATION: FAIL") << std::endl;


        cudaFree(d_A_ref);
        cudaFree(d_B_ref);
        cudaFree(d_C_ref);
    }

    // Clean up
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaStreamDestroy(comm_stream));
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaEventDestroy(compute_done[i]));
        CHECK_CUDA(cudaEventDestroy(comm_done[i]));
        CHECK_CUDA(cudaFree(d_A_recv[i]));
        CHECK_CUDA(cudaFree(d_B_recv[i]));
    }
    CHECK_CUDA(cudaFree(d_C_local));
    CHECK_CUDA(cudaFree(d_A_local));
    CHECK_CUDA(cudaFree(d_B_local));
    CHECK_NCCL(ncclCommDestroy(nccl_row_comm));
    CHECK_NCCL(ncclCommDestroy(nccl_col_comm));

    MPI_Comm_free(&local_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
