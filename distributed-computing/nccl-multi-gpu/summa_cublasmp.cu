
/*
Run and Compile:
# (Run once) 0. Create local environment with cuBLASMp and NCCL (no NVSHMEM needed!)
!curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba && \
 ./bin/micromamba create -y -p ./cublasmp_env -c nvidia -c conda-forge \
 libcublasmp-dev libcublasmp nccl cuda-cudart

# 1. Compile
!./cublasmp_env/bin/nvcc -Wno-deprecated-gpu-targets -arch=sm_75 summa_cublasmp.cu -o summa_cublasmp \
 -I./cublasmp_env/include \
 -I./cublasmp_env/include/cublasMp \
 -L./cublasmp_env/lib \
 -lcublas -lnccl -lcublasmp -lcudart \
 -ccbin mpicxx

# 2. Run
!mpirun --allow-run-as-root --oversubscribe -np 1 \
 -x LD_LIBRARY_PATH=$(pwd)/cublasmp_env/lib:$LD_LIBRARY_PATH \
 -x NCCL_DEBUG=WARN \
 -x NCCL_MULTI_RANK_GPU_ENABLE=1 \
 -x CUBLASMP_LOG_LEVEL=3 \
 ./summa_cublasmp

 # Example output on Google Colab T4:
 # the exact 0 is too clean, real validation needs more than -np 1:
Initializing cuBLASMp multi-node GEMM across 1x1 Grid.
NCCL version 2.30.7+cuda13.3
[2026-09-10 18:38:24][cublasMp][3313][Trace][cublasMpMatmul] Using local Matmul
Frobenius norm ratio ||C_dist - C_ref|| / ||C_ref||: 0
VERIFICATION: PASS

# Google Colab T4 with -np 4:
Initializing cuBLASMp multi-node GEMM across 2x2 Grid.
NCCL version 2.30.7+cuda13.3
[2026-09-10 18:47:40][cublasMp][5920][Trace][cublasMpMatmul] Using generic Matmul
[2026-09-10 18:47:40][cublasMp][5920][Trace][cublasMpMatmul] Matmul NN
[2026-09-10 18:47:40][cublasMp][5921][Trace][cublasMpMatmul] Using generic Matmul
[2026-09-10 18:47:40][cublasMp][5922][Trace][cublasMpMatmul] Using generic Matmul
[2026-09-10 18:47:40][cublasMp][5921][Trace][cublasMpMatmul] Matmul NN
[2026-09-10 18:47:40][cublasMp][5923][Trace][cublasMpMatmul] Using generic Matmul
[2026-09-10 18:47:40][cublasMp][5923][Trace][cublasMpMatmul] Matmul NN
[2026-09-10 18:47:40][cublasMp][5922][Trace][cublasMpMatmul] Matmul NN
Frobenius norm ratio ||C_dist - C_ref|| / ||C_ref||: 9.54703e-07
VERIFICATION: PASS

Note:
Official doc https://docs.nvidia.com/cuda/cublasmp/index.html
Official examples https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLASMp/matmul_ag.cu
*/


#include <mpi.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <cublasmp.h>
//#include <nvshmem.h>
//#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <cstring>
#include <random>

// Error Cehcking Macros
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
        printf("NCCL Error %s: %d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(cmd) do { \
    cublasStatus_t s = cmd; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS Error %s: %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLASMP(cmd) do { \
    cublasMpStatus_t s = cmd; \
    if (s != CUBLASMP_STATUS_SUCCESS) { \
        printf("CUBLASMP Error %s: %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// MPI Index Mapping and Data Distribution Logic
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
    int global_block = global_idx / nb;
    int offset = global_idx % nb;
    int local_block = global_block / p_dim;
    int owner_coord = global_block % p_dim;

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

// Structure to track local matrix dimensions and allocation strides
struct LocalMatrixDim {
    int local_rows;
    int local_cols;
    int alloc_rows;
    int alloc_cols;
    int num_blocks_row;
    int num_blocks_col;
};

// Compute local allocation and valid dimensions under 2D block-cyclic local_block_ryout
inline LocalMatrixDim get_local_matrix_dim(int G_M, int G_N,
                                          int nb, int P_r, int P_c,
                                          int rank_row, int rank_col) {
    int total_blocks_m = (G_M + nb - 1) / nb;
    int total_blocks_n = (G_N + nb - 1) / nb;

    // Round robin to distribute blocks to rank row and col
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

// Distributed Data Ingestion and Extraction
void distribute_matrix(const float* global_mat, float* local_mat,
                      int G_Rows, int G_Cols, int nb,
                      int P_r, int P_c, int rank_row, int rank_col,
                      LocalMatrixDim dim, MPI_Comm cart_comm,
                      int world_rank, int world_size) {
      std::memset(local_mat, 0, dim.alloc_rows * dim.alloc_cols * sizeof(float));

      // Rank 0 coordinator node translates every process's 1D rank id into 2d grid coordinates
      if (world_rank == 0) {
          // for every single process
          for (int p = 0; p < world_size; p++) {
              int p_coords[2];
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

// Main Execution

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

    // Shape (numbers of rows and columns)
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    int P_r = dims[0];
    int P_c = dims[1];

    // Cartesian grid wraps around or no
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    // 2D coordinates
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);
    int rank_row = coords[0];
    int rank_col = coords[1];

    const int Global_M = 3500;
    const int Global_N = 2048;
    const int Global_K = 5120;
    const int Nb = 1024;

    LocalMatrixDim dim_C = get_local_matrix_dim(Global_M, Global_N, Nb, P_r, P_c, rank_row, rank_col);
    LocalMatrixDim dim_A = get_local_matrix_dim(Global_M, Global_K, Nb, P_r, P_c, rank_row, rank_col);
    LocalMatrixDim dim_B = get_local_matrix_dim(Global_K, Global_N, Nb, P_r, P_c, rank_row, rank_col);

    std::vector<float> h_A_local(dim_A.alloc_rows * dim_A.alloc_cols, 0.0f);
    std::vector<float> h_B_local(dim_B.alloc_rows * dim_B.alloc_cols, 0.0f);
    std::vector<float> h_C_local(dim_C.alloc_rows * dim_C.alloc_cols, 0.0f);
    std::vector<float> global_A, global_B, global_C_verify;

    if (world_rank == 0) {
        printf("Initializing cuBLASMp multi-node GEMM across %dx%d Grid.\n", P_r, P_c);
        global_A.resize(Global_M * Global_K);
        global_B.resize(Global_K * Global_N);
        global_C_verify.resize(Global_M * Global_N, 0.0f);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (auto& val : global_A) val = dis(gen);
        for (auto& val : global_B) val = dis(gen);
    }

    distribute_matrix(global_A.data(), h_A_local.data(), Global_M, Global_K, Nb, P_r, P_c, rank_row, rank_col, dim_A, cart_comm, world_rank, world_size);
    distribute_matrix(global_B.data(), h_B_local.data(), Global_K, Global_N, Nb, P_r, P_c, rank_row, rank_col, dim_B, cart_comm, world_rank, world_size);

    float *d_C_local, *d_A_local, *d_B_local;
    CHECK_CUDA(cudaMalloc((void**)&d_C_local, dim_C.alloc_rows * dim_C.alloc_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_A_local, dim_A.alloc_rows * dim_A.alloc_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B_local, dim_B.alloc_rows * dim_B.alloc_cols * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A_local, h_A_local.data(), dim_A.alloc_rows * dim_A.alloc_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_local, h_B_local.data(), dim_B.alloc_rows * dim_B.alloc_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C_local, 0, dim_C.alloc_rows * dim_C.alloc_cols * sizeof(float)));

    cudaStream_t compute_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));

    // cuBLASMp Exectution Setup

    // 1. Establish a single global NCCL communicator spanning the entire 2D grid
    ncclUniqueId global_id;
    if (world_rank == 0) {
        ncclGetUniqueId(&global_id);
    }
    MPI_Bcast(&global_id, sizeof(global_id), MPI_BYTE, 0, cart_comm);

    ncclComm_t nccl_global_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_global_comm, world_size, global_id, world_rank));

    // 2. Initialize cuBLASMp library handle
    // Requires passing the compute stream according to modern cuBLASMp v0.4+ signatures
    cublasMpHandle_t mp_handle;
    CHECK_CUBLASMP(cublasMpCreate(&mp_handle, compute_stream));

    // 3. Define 2D Process Grid
    // By using CUBLASMP_GRID_LAYOUT_COL_MAJOR on a transposed (P_c x P_r) grid,
    // the internal rank mappings exactly align with Row-Major MPI_Cart_create grid
    cublasMpGrid_t grid;
    CHECK_CUBLASMP(cublasMpGridCreate(P_c, P_r, CUBLASMP_GRID_LAYOUT_COL_MAJOR, nccl_global_comm, &grid));

    // 4. Matrix Descriptors
    // Transpose all matrix dimensions globally to map Row-Major arrays to the library's Column-Major expectation.
    // llc (local leading dimension) is set to alloc_cols
    cublasMpMatrixDescriptor_t descA, descB, descC;
    CHECK_CUBLASMP(cublasMpMatrixDescriptorCreate(Global_K, Global_M, Nb, Nb, 0, 0, dim_A.alloc_cols, CUDA_R_32F, grid, &descA));
    CHECK_CUBLASMP(cublasMpMatrixDescriptorCreate(Global_N, Global_K, Nb, Nb, 0, 0, dim_B.alloc_cols, CUDA_R_32F, grid, &descB));
    CHECK_CUBLASMP(cublasMpMatrixDescriptorCreate(Global_N, Global_M, Nb, Nb, 0, 0, dim_C.alloc_cols, CUDA_R_32F, grid, &descC));
   
    // C = alpha * A x B + beta * C
    float alpha = 1.0f, beta = 1.0f;

    // 5. Distributed Matmul Execution
    // Computing: C_colmaj(N x M) = B_colmaj(N x K) * A_colmaj(K x M)
    
    // Create and configure the matmul Descriptor
    cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;

    cublasMpMatmulDescriptor_t matmulDesc;
    CHECK_CUBLASMP(cublasMpMatmulDescriptorCreate(&matmulDesc, CUBLAS_COMPUTE_32F));

    CHECK_CUBLASMP(cublasMpMatmulDescriptorSetAttribute(matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLASMP(cublasMpMatmulDescriptorSetAttribute(matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &transB, sizeof(transB)));

    // Query workspace requirements
    size_t workspaceInBytesOnDevice = 0, workspaceInBytesOnHost = 0;
    CHECK_CUBLASMP(cublasMpMatmul_bufferSize(
        mp_handle, 
        matmulDesc, 
        Global_N, Global_M, Global_K,
        &alpha, 
        d_B_local, 1, 1, descB, 
        d_A_local, 1, 1, descA, 
        &beta, 
        d_C_local, 1, 1, descC,
        d_C_local, 1, 1, descC,
        &workspaceInBytesOnDevice, 
        &workspaceInBytesOnHost
    ));

    //// Allocate Workspaces (NVSHMEM for distributed device workspace)
    //void* d_work = nvshmem_malloc(workspaceInBytesOnDevice);
    //std::vector<int8_t> h_work(workspaceInBytesOnHost);

    void* d_work = nullptr;
    CHECK_CUDA(cudaMalloc(&d_work, workspaceInBytesOnDevice));

    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    // Execute Distributed GEMM
    CHECK_CUBLASMP(cublasMpMatmul(
        mp_handle, 
        matmulDesc, 
        Global_N, Global_M, Global_K,
        &alpha, 
        d_B_local, 1, 1, descB, 
        d_A_local, 1, 1, descA, 
        &beta,
        d_C_local, 1, 1, descC,
        d_C_local, 1, 1, descC,
        d_work, 
        workspaceInBytesOnDevice, 
        h_work.data(), 
        workspaceInBytesOnHost
    ));

    CHECK_CUDA(cudaStreamSynchronize(compute_stream));

    // 6. Cleanup Library Resources
    cudaFree(d_work);
    CHECK_CUBLASMP(cublasMpMatmulDescriptorDestroy(matmulDesc));
    CHECK_CUBLASMP(cublasMpMatrixDescriptorDestroy(descA));
    CHECK_CUBLASMP(cublasMpMatrixDescriptorDestroy(descB));
    CHECK_CUBLASMP(cublasMpMatrixDescriptorDestroy(descC));
    CHECK_CUBLASMP(cublasMpGridDestroy(grid));
    CHECK_CUBLASMP(cublasMpDestroy(mp_handle));
    CHECK_NCCL(ncclCommDestroy(nccl_global_comm));
    
    // End cuBLASMp

    CHECK_CUDA(cudaMemcpy(h_C_local.data(), d_C_local, dim_C.alloc_rows * dim_C.alloc_cols * sizeof(float), cudaMemcpyDeviceToHost));

    gather_matrix(global_C_verify.data(), h_C_local.data(), Global_M, Global_N, Nb, P_r, P_c, rank_row, rank_col, dim_C, cart_comm, world_rank, world_size);

    if (world_rank == 0) {
        cublasHandle_t cublas_handle;
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));

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

        double norm_diff_sq = 0.0, norm_ref_sq = 0.0;
        for (int i = 0; i < Global_M * Global_N; i++) {
            double diff = global_C_verify[i] - h_C_ref[i];
            norm_diff_sq += diff * diff;
            norm_ref_sq += (double)h_C_ref[i] * h_C_ref[i];
        }
        double norm_ratio = std::sqrt(norm_diff_sq) / std::sqrt(norm_ref_sq);

        std::cout << "Frobenius norm ratio ||C_dist - C_ref|| / ||C_ref||: " << norm_ratio << "\n";
        double threshold = std::sqrt((double)Global_K) * 1e-6; 
        std::cout << (norm_ratio < threshold ? "VERIFICATION: PASS" : "VERIFICATION: FAIL") << std::endl;

        cudaFree(d_A_ref);
        cudaFree(d_B_ref);
        cudaFree(d_C_ref);
        CHECK_CUBLAS(cublasDestroy(cublas_handle));
    }

    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaFree(d_C_local));
    CHECK_CUDA(cudaFree(d_A_local));
    CHECK_CUDA(cudaFree(d_B_local));

    MPI_Comm_free(&local_comm);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    
    return 0;
}
