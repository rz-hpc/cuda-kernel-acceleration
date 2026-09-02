
// !apt-get update
// !apt-get install -y libopenmpi-dev openmpi-bin
// Compile and run with
// !nvcc summa_gemm.cu -o summa_gemm -lcublas -lnccl -ccbin mpicxx
// !./summa_gemm
//
// 
// The run needs to be multi-gpu environment
// Because of Google Colab hardware limitation (only one T4 GPU), the output
// 
// nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
// [773eafce20e3:04260] *** An error occurred in MPI_Cart_create
// [773eafce20e3:04260] *** reported by process [317652993,0]
// [773eafce20e3:04260] *** on communicator MPI_COMM_WORLD
// [773eafce20e3:04260] *** MPI_ERR_ARG: invalid argument of some other kind
// [773eafce20e3:04260] *** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
// [773eafce20e3:04260] ***    and potentially your MPI job)

// With NCCL_DEBUG = WARN and -np 1 (1 core on Colab)
// !nvcc summa_gemm.cu -o summa_gemm -lcublas -lnccl -ccbin mpicxx -arch=native
// !NCCL_DEBUG=WARN mpirun --allow-run-as-root -np 1 ./summa_gemm

// Output:
// NCCL version 2.25.1+cuda12.8
// [Rank 0 (row=0,col=0)] expected=50927616.00 max_abs_err=4.000000 rel_err=7.85e-08 verified=1
// SUMMA correctness verified across ALL ranks: PASS

// With NCCL_DEBUG = WARN and -np 2 on 1 core Colab
// !nvcc summa_gemm.cu -o summa_gemm -lcublas -lnccl -ccbin mpicxx -arch=native
// !mpirun --allow-run-as-root -x NCCL_DEBUG=WARN -np 2 ./summa_gemm

// Output:
// There are not enough slots available in the system to satisfy the 2
// slots that were requested by the application: ...

// cuBLAS Baseline Verification Output:
// NCCL version 2.25.1+cuda12.8
// Baseline C[0] = 2.28981e+10, Distributed C[0] = 2.28981e+10
// [Rank 0] cuBLAS Baseline Verification completed.
// Max Absolute Error across 1024x1024 matrix: 36864
// Max Relative Error: 1.40157e-06
// DISTRIBUTED VERIFICATION: PASS

#include <mpi.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Error checking macro (essential fro production-grade CUDA/NCCL)
#define CHECK_CUDA(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_NCCL(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(cmd) do { \
    cublasStatus_t s = cmd; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS error %s:%d\n", __FILE__, __LINE__); \
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

__host__ __device__ inline LocalCoord global_to_local(int global_idx, int nb, int p_dim) {
    int block_idx = global_idx / nb;
    int owner_coord = block_idx % p_dim;
    int local_block = block_idx / p_dim;
    int offset = global_idx % nb;

    return {owner_coord, local_block * nb + offset};
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

// structure to track local matrix dimensions and allocation strides
struct LocalMatrixDim {
    int local_rows; // Exact valid rows owned by rank
    int local_cols; // Exact valid cols owned by rank
    int alloc_rows; // Padded rows (aligned to multiple of Nb)
    int alloc_cols; // Padded cols (aligned to multiple of Nb)
    int num_blocks_row; // Number of Nb x Nb block tiles vertically
    int num_blocks_col; // Number of Nb x Nb block titles horizontally
};

// Compute local allocation and valid dimensions under 2D block-cyclic layout
inline LocalMatrixDim get_local_matrix_dim(int G_M, int G_N, int nb, int P_r, int P_c, int rank_row, int rank_col) {
    int total_blocks_m = (G_M + nb - 1) / nb;
    int total_blocks_n = (G_N + nb - 1) / nb;

    // Count how many full/partial blocks this rank owns
    int blocks_r = total_blocks_m / P_r + (rank_row < (total_blocks_m % P_r) ? 1 : 0);
    int blocks_c = total_blocks_n / P_c + (rank_col < (total_blocks_n % P_c) ? 1 : 0);

    // Calculate exact non-padded valid element bounds
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
        blocks_r * nb, // Allocating full padded tiles simplifies memory strides
        blocks_c * nb,
        blocks_r,
        blocks_c
    };
}

// Convert local element coordinate back to global coordinate in 2D block-cyclic layout
__host__ __device__ inline Global2DCoord local_to_global_2d_arbitrary(
                                              int l_row, int l_col,
                                              int rank_row, int rank_col,
                                              int nb, int P_r, int P_c) {
    int local_block_r = l_row / nb;
    int offset_r = l_row % nb;
    int global_block_r = local_block_r * P_r + rank_row;
    int g_row = global_block_r * nb + offset_r;

    int local_block_c = l_col / nb;
    int offset_c = l_col % nb;
    int global_block_c = local_block_c * P_c + rank_col;
    int g_col = global_block_c * nb + offset_c;

    return {g_row, g_col};
}

__global__ void fill_kernel(float* ptr, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = value;
    }
}

// Block-Cyclic Matrix Initialization Kernels for Arbitrary (M, N, K)
__global__ void init_matrix_A_kernel(float* ptr, int alloc_cols, int G_M, int G_K, int Nb, int P_r, int P_c, int rank_row, int rank_col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Strided grid loop over local memory
    for (int i = idx; i < alloc_cols * G_M; i += total_threads) {
        int l_row = i / alloc_cols;
        int l_col = i % alloc_cols;

        Global2DCoord g = local_to_global_2d_arbitrary(l_row, l_col, rank_row, rank_col, Nb, P_r, P_c);
        if (g.g_row < G_M && g.g_col < G_K) {
            ptr[l_row * alloc_cols + l_col] = static_cast<float>(g.g_row + g.g_col);
        } else {
            ptr[l_row * alloc_cols + l_col] = 0.0f; // Padding
        }
    }
}

__global__ void init_matrix_B_kernel(float* ptr, int alloc_cols, int G_K, int G_N, int Nb, int P_r, int P_c, int rank_row, int rank_col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = idx; i < alloc_cols * G_K; i += total_threads) {
        int l_row = i / alloc_cols;
        int l_col = i % alloc_cols;

        Global2DCoord g = local_to_global_2d_arbitrary(l_row, l_col, rank_row, rank_col, Nb, P_r, P_c);
        if (g.g_row < G_K && g.g_col < G_N) {
            ptr[l_row * alloc_cols + l_col] = static_cast<float>(g.g_row - g.g_col);
        } else {
            ptr[l_row * alloc_cols + l_col] = 0.0f; // Padding
        }
    }
}

// Clean baseline initialization kernel for row-major reference matrices
__global__ void init_baseline_matrix_kernel(float* ptr, int M, int N, bool is_matrix_A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int row = idx / N; // Row-major: row index is idx / N, col index is idx % N
        int col = idx % N;
        if (is_matrix_A) {
            ptr[idx] = static_cast<float>(row + col);
        } else {
            ptr[idx] = static_cast<float>(row - col);
        }
    }
}

// Struct to prevent precision loss during MPI Gather
struct FloatPayload {
    int global_offset;
    float val;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 1. Set GPU Device maaping (assumes 1 process per GPU on a node)
    int local_rank = world_rank; // In real deployments, parse local rank carefully
    CHECK_CUDA(cudaSetDevice(local_rank));

    // 2. Setup 2D MPI Grid
    // MPI_Dims_create finds a valid P x Q factorization of world_size to fill dims
    // For 4 cores, 2 x 2, for 6 cores 3 x 2
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    int P_r = dims[0];
    int P_c = dims[1];

    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);
    int rank_row = coords[0];
    int rank_col = coords[1];

    // 3. Create Row and Column Sub-Communicators for NCCL Broadcasts
    MPI_Comm row_comm, col_comm;
    // Split by row: everyone with the same rank_row joins the same row_comm
    MPI_Comm_split(cart_comm, rank_row, rank_col, &row_comm);
    // Split by col: everyone with the same rank_col joins the same col_comm
    MPI_Comm_split(cart_comm, rank_col, rank_row, &col_comm);

    int row_comm_rank, col_comm_rank;
    MPI_Comm_rank(row_comm, &row_comm_rank); // Effectively equal to rank_col
    MPI_Comm_rank(col_comm, &col_comm_rank); // Effectively equal to rank_row

    // 4. Initial NCCL Communicators
    ncclUniqueId row_id, col_id;
    if (row_comm_rank == 0) ncclGetUniqueId(&row_id);
    if (col_comm_rank == 0) ncclGetUniqueId(&col_id);

    MPI_Bcast(&row_id, sizeof(row_id), MPI_BYTE, 0, row_comm);
    MPI_Bcast(&col_id, sizeof(col_id), MPI_BYTE, 0, col_comm);

    int row_comm_size, col_comm_size; // the actual communicator size
    MPI_Comm_size(row_comm, &row_comm_size);
    MPI_Comm_size(col_comm, &col_comm_size);

    ncclComm_t nccl_row_comm, nccl_col_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_row_comm, row_comm_size, row_id, row_comm_rank));
    CHECK_NCCL(ncclCommInitRank(&nccl_col_comm, col_comm_size, col_id, col_comm_rank));

    // 5. Matrix and Block Size Setup (e.g., K_blocks total)

    // Inputs: arbitrary M, N, K and Block Size Nb
    const int Global_M = 3500;
    const int Global_N = 2048;
    const int Global_K = 5120;
    const int Nb = 1024; // Tile size

    const int K_blocks = (Global_K + Nb - 1) / Nb;

    // Calculate dynamic local storage needs per rank
    LocalMatrixDim dim_C = get_local_matrix_dim(Global_M, Global_N, Nb, P_r, P_c, rank_row, rank_col);

    // Matrix A panel storage: Row ranks split M, Column ranks split K panel iterations
    LocalMatrixDim dim_A = get_local_matrix_dim(Global_M, Global_K, Nb, P_r, P_c, rank_row, rank_col);

    // Matrix B panel storage: Row ranks split K panel iterations, Column ranks split N
    LocalMatrixDim dim_B = get_local_matrix_dim(Global_K, Global_N, Nb, P_r, P_c, rank_row, rank_col);

    size_t bytes_C = dim_C.alloc_rows * dim_C.alloc_cols * sizeof(float);
    size_t bytes_A = dim_A.alloc_rows * dim_A.alloc_cols * sizeof(float);
    size_t bytes_B = dim_B.alloc_rows * dim_B.alloc_cols * sizeof(float);

    // Device Memory Allocations (Zero-initialized to handle padding seamlessly)
    float *d_C_local, *d_A_local, *d_B_local;
    CHECK_CUDA(cudaMalloc(&d_C_local, bytes_C));
    CHECK_CUDA(cudaMalloc(&d_A_local, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B_local, bytes_B));
    CHECK_CUDA((cudaMemset(d_C_local, 0, bytes_C)));
    CHECK_CUDA((cudaMemset(d_A_local, 0, bytes_A)));
    CHECK_CUDA((cudaMemset(d_B_local, 0, bytes_B)));

    // Matrix A and B Initial
    int threadsPerBlock = 256;
    // int blocksPerGrid = (Nb * Nb + threadsPerBlock - 1) / threadsPerBlock;
    
    // Total elements allocated in local 2D buffers for Matrix A and B
    int total_elements_A = dim_A.alloc_rows * dim_A.alloc_cols;
    int total_elements_B = dim_B.alloc_rows * dim_B.alloc_cols;
    int blocks_A = (total_elements_A + threadsPerBlock - 1) / threadsPerBlock;
    int blocks_B = (total_elements_B + threadsPerBlock - 1) / threadsPerBlock;

    init_matrix_A_kernel<<<blocks_A, threadsPerBlock>>>(d_A_local, dim_A.alloc_cols, Global_M, Global_K, Nb, P_r, P_c, rank_row, rank_col);
    init_matrix_B_kernel<<<blocks_B, threadsPerBlock>>>(d_B_local, dim_B.alloc_cols, Global_K, Global_N, Nb, P_r, P_c, rank_row, rank_col);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 6. Double Buffering Setup (The Overlap architecture)
    
    // Panel A size: local rows x panel width (Nb)
    size_t bytes_A_panel = dim_A.alloc_rows * Nb * sizeof(float);
    // Panel B size: panel height (Nb) x local cols
    size_t bytes_B_panel = Nb * dim_B.alloc_cols * sizeof(float);
    
    float *d_A_recv[2], *d_B_recv[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMalloc(&d_A_recv[i], bytes_A_panel));
        CHECK_CUDA(cudaMalloc(&d_B_recv[i], bytes_B_panel));
        CHECK_CUDA(cudaMemset(d_A_recv[i], 0, bytes_A_panel));
        CHECK_CUDA(cudaMemset(d_B_recv[i], 0, bytes_B_panel));
    }

    cudaStream_t compute_stream, comm_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&comm_stream));

    // Decalre per-buffer event arrays (instead of single events)
    cudaEvent_t compute_done[2], comm_done[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaEventCreate(&compute_done[i]));
        CHECK_CUDA(cudaEventCreate(&comm_done[i]));
    }

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));

    float alpha = 1.0f, beta = 1.0f; // C = alpha * A * B + beta * c

    ///////////////////////////////////////////
    // SUMMA EXECUTION
    ///////////////////////////////////////////
    int current_buf = 0;

    // A. Prime the pump (Load Buffer 0)
    int root_A = 0 % P_c;
    int root_B = 0 % P_r;
    int current_kb_0 = std::min(Nb, Global_K - 0 * Nb);

    if (rank_col == root_A) {
      int local_k_A = 0 / P_c;
      float* src_A = d_A_local + (local_k_A * Nb);
      // CHECK_CUDA(cudaMemcpyAsync(d_A_recv[0], d_A_local + (0 * Nb * Nb), tile_bytes, cudaMemcpyDeviceToDevice, comm_stream));
      CHECK_CUDA(cudaMemcpy2DAsync(d_A_recv[0], current_kb_0 * sizeof(float), // dpitch = width of panel
                                 src_A, dim_A.alloc_cols * sizeof(float),   // spitch = stride of local matrix A
                                 current_kb_0 * sizeof(float),              // width = panel width in bytes
                                 dim_A.alloc_rows,                          // height = total local rows
                                 cudaMemcpyDeviceToDevice, comm_stream));
    }
    if (rank_row == root_B) {
        int local_k_B = 0 / P_r;
        float* src_B = d_B_local + (local_k_B * Nb * dim_B.alloc_cols);
        // CHECK_CUDA(cudaMemcpyAsync(d_B_recv[0], d_B_local + (0 * Nb * Nb), tile_bytes, cudaMemcpyDeviceToDevice, comm_stream));
        CHECK_CUDA(cudaMemcpyAsync(d_B_recv[0], src_B, current_kb_0 * dim_B.alloc_cols * sizeof(float), cudaMemcpyDeviceToDevice, comm_stream));
    }

    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclBroadcast((const void*)d_A_recv[0], // pointer to data being sent (read, only this rank)
                              (void*)d_A_recv[0], // pointer to where data is saved (write, on all ranks)
                              Nb * Nb, // number of elements
                              ncclFloat, // type of data
                              root_A, // the rank ID inside this specific communicator that holds source data
                              nccl_row_comm, // the specific ncclComm_t universe (row or col comm)
                              comm_stream)); // cuda stream executing the transfer
    CHECK_NCCL(ncclBroadcast((const void*)d_B_recv[0],
                              (void*)d_B_recv[0],
                              Nb * Nb,
                              ncclFloat,
                              root_B,
                              nccl_col_comm, 
                              comm_stream));
    CHECK_NCCL(ncclGroupEnd());
    CHECK_CUDA(cudaEventRecord(comm_done[0], comm_stream));

    // B. The Main Loop
    for (int k = 0; k < K_blocks; k++) {
        int next_buf = (current_buf + 1) % 2;
        int current_kb = std::min(Nb, Global_K - k * Nb);

        // 1. Compute on current buffer (waits for comm_done)
        CHECK_CUDA(cudaStreamWaitEvent(compute_stream, comm_done[current_buf], 0));

        // Row-major GEMM via cuBLAS Column-Major API (C = A * B -> C^T = B^T * A^T)
        // M_cublas = dim_C.alloc_cols
        // N_cublas = dim_C.alloc_rows
        // K_cublas = current_kb
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         dim_C.alloc_cols, dim_C.alloc_rows, current_kb,
                         &alpha,
                         d_B_recv[current_buf], dim_B.alloc_cols, // lda = dim_B.alloc_cols (must be >= dim_C.alloc_cols)
                         d_A_recv[current_buf], current_kb,       // ldb = current_kb        (must be >= current_kb)
                         &beta,
                         d_C_local, dim_C.alloc_cols));           // ldc = dim_C.alloc_cols (must be >= dim_C.alloc_cols)
        CHECK_CUDA(cudaEventRecord(compute_done[current_buf], compute_stream));

        // 2. Commnicate next buffer (waits for compute_done on the next buffer)
        if (k + 1 < K_blocks) {
            int next_k = k + 1;
            int next_kb = std::min(Nb, Global_K - next_k * Nb);
            int next_root_A = next_k % P_c;
            int next_root_B = next_k % P_r;

            // communicate into next_buf, has to wait the next_buf compute from 2 iterations ago to have finished reading it
            CHECK_CUDA(cudaStreamWaitEvent(comm_stream, compute_done[next_buf], 0));

            // Copy next tile from local storage to send buffer if we own it
            if (rank_col == next_root_A) {
                // CHECK_CUDA(cudaMemcpyAsync(d_A_recv[next_buf], d_A_local + (next_k * Nb * Nb), tile_bytes, cudaMemcpyDeviceToDevice, comm_stream));
                int local_k_A = next_k / P_c;
                float* src_A = d_A_local + (local_k_A * Nb);
                CHECK_CUDA(cudaMemcpy2DAsync(d_A_recv[next_buf], Nb * sizeof(float),
                                             src_A, dim_A.alloc_cols * sizeof(float),
                                             next_kb * sizeof(float), dim_A.alloc_rows,
                                             cudaMemcpyDeviceToDevice, comm_stream));
            }
            if (rank_row == next_root_B) {
                // CHECK_CUDA(cudaMemcpyAsync(d_B_recv[next_buf], d_B_local + (next_k * Nb * Nb), tile_bytes, cudaMemcpyDeviceToDevice, comm_stream));
                int local_k_B = next_k / P_r;
                float* src_B = d_B_local + (local_k_B * Nb * dim_B.alloc_cols);
                CHECK_CUDA(cudaMemcpyAsync(d_B_recv[next_buf], src_B, next_kb * dim_B.alloc_cols * sizeof(float), cudaMemcpyDeviceToDevice, comm_stream));
            }

            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclBroadcast((const void*)d_A_recv[next_buf], (void*)d_A_recv[next_buf], Nb * Nb, ncclFloat, next_root_A, nccl_row_comm, comm_stream));
            CHECK_NCCL(ncclBroadcast((const void*)d_B_recv[next_buf], (void*)d_B_recv[next_buf], Nb * Nb, ncclFloat, next_root_B, nccl_col_comm, comm_stream));
            CHECK_NCCL(ncclGroupEnd());

            CHECK_CUDA(cudaEventRecord(comm_done[next_buf], comm_stream));
        }

        current_buf = next_buf;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Gather and cuBLAS Ground Truth Verification on Rank 0
    std::vector<float> h_C_local(dim_C.alloc_rows * dim_C.alloc_cols);
    CHECK_CUDA(cudaMemcpy(h_C_local.data(), d_C_local, dim_C.alloc_rows * dim_C.alloc_cols * sizeof(float), cudaMemcpyDeviceToHost));

    if (world_rank != 0) {
        for (int lr = 0; lr < dim_C.alloc_rows; lr++) {
            for (int lc = 0; lc < dim_C.alloc_cols; lc++) {
                Global2DCoord g = local_to_global_2d_arbitrary(lr, lc, rank_row, rank_col, Nb, P_r, P_c);
                if (g.g_row < Global_M && g.g_col < Global_N) {
                    float val = h_C_local[lr * dim_C.alloc_cols + lc];
                    FloatPayload p = {g.g_row * Global_N + g.g_col, val};
                    MPI_Send(&p, sizeof(FloatPayload), MPI_BYTE, 0, 1, cart_comm);
                }
            }
        }
    }
    else {
        std::vector<float> final_C_dist(Global_M * Global_N, 0.0f);

        // Self-map Rank 0 using row-major global offset
        for (int lr = 0; lr < dim_C.alloc_rows; lr++) {
            for (int lc = 0; lc < dim_C.alloc_cols; lc++) {
                Global2DCoord g = local_to_global_2d_arbitrary(lr, lc, rank_row, rank_col, Nb, P_r, P_c);
                if (g.g_row < Global_M && g.g_col < Global_N) {
                    final_C_dist[g.g_row * Global_N + g.g_col] = h_C_local[lr * dim_C.alloc_cols + lc];
                }
            }
        }

        // Receive payloads from Workers
        int total_valid_elements = Global_M * Global_N;
        int rank0_valid_elements = dim_C.local_rows * dim_C.local_cols;
        int incoming_elements = total_valid_elements - rank0_valid_elements;

        for (int i = 0; i < incoming_elements; i++) {
            FloatPayload p;
            MPI_Recv(&p, sizeof(FloatPayload), MPI_BYTE, MPI_ANY_SOURCE, 1, cart_comm, MPI_STATUS_IGNORE);
            final_C_dist[p.global_offset] = p.val;
        }

        // Run baseline GEMM to Verify
        float *d_A_ref, *d_B_ref, *d_C_ref;
        CHECK_CUDA(cudaMalloc(&d_A_ref, Global_M * Global_K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_ref, Global_K * Global_N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_ref, Global_M * Global_N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_C_ref, 0, Global_M * Global_N * sizeof(float)));

        int blocks_A_ref = (Global_M * Global_K + threadsPerBlock - 1) / threadsPerBlock;
        int blocks_B_ref = (Global_K * Global_N + threadsPerBlock - 1) / threadsPerBlock;
        //int blocks_C = (Global_M * Global_N + threadsPerBlock - 1) / threadsPerBlock;

        // Use clean baseline initialization kernels
        init_baseline_matrix_kernel<<<blocks_A_ref, threadsPerBlock>>>(d_A_ref, Global_M, Global_K, true);
        init_baseline_matrix_kernel<<<blocks_B_ref, threadsPerBlock>>>(d_B_ref, Global_K, Global_N, false);

        // Baseline cuBLAS Row-Major GEMM call (B_ref first, A_ref second)
        // Dimensions: N = Global_N, M = Global_M, K = Global_K
        // Leading Dimensions: ldb = Global_N, lda = Global_K, ldc = Global_N
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                Global_N, Global_M, Global_K,
                                &alpha, 
                                d_B_ref, Global_N, // ldb = Global_N
                                d_A_ref, Global_K, // lda = Global_K
                                &beta, 
                                d_C_ref, Global_N)); // ldc = Global_N

        std::vector<float> h_C_ref(Global_M * Global_N);
        CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C_ref, Global_M * Global_N * sizeof(float), cudaMemcpyDeviceToHost));   
    
        std::cout << "Baseline C[0] = " << h_C_ref[0] << ", Distributed C[0] = " << final_C_dist[0] << "\n";

        float max_err = 0.0f;
        float max_rel_err = 0.0f;
        for (int i = 0; i < Global_M * Global_N; i++) {
            float abs_diff = std::fabs(final_C_dist[i] - h_C_ref[i]);
            max_err = std::max(max_err, abs_diff);
            float rel_diff = abs_diff / (std::fabs(h_C_ref[i]) + 1e-5f);
            max_rel_err = std::max(max_rel_err, rel_diff);
        }

        std::cout << "[Rank 0] cuBLAS Baseline Verification completed.\n";
        std::cout << "Max Absolute Error across " << Global_M << "x" << Global_N << " matrix: " << max_err << "\n";
        std::cout << "Max Relative Error: " << max_rel_err << "\n";
        
        // Use a reasonable relative tolerance for FP32 large-scale accumulations
        std::cout << (max_rel_err < 1e-3f ? "DISTRIBUTED VERIFICATION: PASS" : "DISTRIBUTED VERIFICATION: FAIL") << std::endl;
        cudaFree(d_A_ref); cudaFree(d_B_ref); cudaFree(d_C_ref);
    }

    // Cleanup (reverse order teardown)

    // A. Destry cuBLAS / CUDA streams & events
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaStreamDestroy(comm_stream));
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaEventDestroy(compute_done[i]));
        CHECK_CUDA(cudaEventDestroy(comm_done[i]));
    }

    // B. Free GPU memory
    CHECK_CUDA(cudaFree(d_C_local));
    CHECK_CUDA(cudaFree(d_A_local));
    CHECK_CUDA(cudaFree(d_B_local));
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaFree(d_A_recv[i]));
        CHECK_CUDA(cudaFree(d_B_recv[i]));
    }

    // C. Destroy NCCL Communicators
    CHECK_NCCL(ncclCommDestroy(nccl_row_comm));
    CHECK_NCCL(ncclCommDestroy(nccl_col_comm));

    // D. Free MPI Sub-Communicators
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);

    // E. Finalize MPI
    MPI_Finalize();

    return 0;
}
