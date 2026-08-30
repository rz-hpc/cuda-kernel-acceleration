
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

#include <mpi.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

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

__global__ void fill_kernel(float* ptr, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = value;
    }
}

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
    const int Nb = 1024; // 1024 x 1024 tiles
    const int K_blocks = 4; // Total iterations
    size_t tile_bytes = Nb * Nb * sizeof(float);

    // Local device memory for Matrix C and our owned piecies of A and B
    float *d_C_local, *d_A_local, *d_B_local;
    CHECK_CUDA(cudaMalloc(&d_C_local, tile_bytes));
    CHECK_CUDA(cudaMalloc(&d_A_local, tile_bytes * K_blocks)); // Assuming owns a strip
    CHECK_CUDA(cudaMalloc(&d_B_local, tile_bytes * K_blocks));
    CHECK_CUDA((cudaMemset(d_C_local, 0, tile_bytes)));

    int threadsPerBlock = 256; 
    int blocksPerGrid = (Nb * Nb + threadsPerBlock - 1) / threadsPerBlock;
    for (int k = 0; k < K_blocks; k++) {
        float val = (rank_row + 1) * 100.0f + (rank_col + 1) * 10.0f + k;
        fill_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A_local + k * Nb * Nb, val, Nb * Nb);
        fill_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B_local + k * Nb * Nb, val, Nb * Nb);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 6. Double Buffering Setup (The Overlap architecture)
    float *d_A_recv[2], *d_B_recv[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMalloc(&d_A_recv[i], tile_bytes));
        CHECK_CUDA(cudaMalloc(&d_B_recv[i], tile_bytes));
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
    int root_A = 0 % row_comm_size; // Which process column owns A's block 0
    int root_B = 0 % col_comm_size; // Which process row owns B's block 0

    if (rank_col == root_A)
      CHECK_CUDA(cudaMemcpyAsync(d_A_recv[0], d_A_local + (0 * Nb * Nb), tile_bytes, cudaMemcpyDeviceToDevice, comm_stream));
    if (rank_row == root_B)
      CHECK_CUDA(cudaMemcpyAsync(d_B_recv[0], d_B_local + (0 * Nb * Nb), tile_bytes, cudaMemcpyDeviceToDevice, comm_stream));

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

        // 1. Compute on current buffer (waits for comm_done)
        CHECK_CUDA(cudaStreamWaitEvent(compute_stream, comm_done[current_buf], 0));

        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    Nb, Nb, Nb,
                    &alpha,
                    d_B_recv[current_buf], Nb, // cublas is column-major, A and B a swapped
                    d_A_recv[current_buf], Nb,
                    &beta,
                    d_C_local, Nb));
        CHECK_CUDA(cudaEventRecord(compute_done[current_buf], compute_stream));

        // 2. Commnicate next buffer (waits for compute_done on the next buffer)
        if (k + 1 < K_blocks) {
            int next_k = k + 1;
            int next_root_A = next_k % row_comm_size;
            int next_root_B = next_k % col_comm_size;

            // communicate into next_buf, has to wait the next_buf compute from 2 iterations ago to have finished reading it
            CHECK_CUDA(cudaStreamWaitEvent(comm_stream, compute_done[next_buf], 0));

            // Copy next tile from local storage to send buffer if we own it
            if (rank_col == next_root_A)
                CHECK_CUDA(cudaMemcpyAsync(d_A_recv[next_buf], d_A_local + (next_k * Nb * Nb), tile_bytes, cudaMemcpyDeviceToDevice, comm_stream));
            if (rank_row == next_root_B)
                CHECK_CUDA(cudaMemcpyAsync(d_B_recv[next_buf], d_B_local + (next_k * Nb * Nb), tile_bytes, cudaMemcpyDeviceToDevice, comm_stream));

            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclBroadcast((const void*)d_A_recv[next_buf], (void*)d_A_recv[next_buf], Nb * Nb, ncclFloat, next_root_A, nccl_row_comm, comm_stream));
            CHECK_NCCL(ncclBroadcast((const void*)d_B_recv[next_buf], (void*)d_B_recv[next_buf], Nb * Nb, ncclFloat, next_root_B, nccl_col_comm, comm_stream));
            CHECK_NCCL(ncclGroupEnd());

            CHECK_CUDA(cudaEventRecord(comm_done[next_buf], comm_stream));
        }

        current_buf = next_buf;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Verification: does the distributed communication pattern work correctly
    // Note: the SUMMA computation is not verified in this experiment
    std::vector<float> h_C(Nb * Nb);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C_local, tile_bytes, cudaMemcpyDeviceToHost));

    float expected = 0.0f;
    for (int k = 0; k < K_blocks; k++) {
        int owner_col = k % row_comm_size; // which col-position owns A's k-th block
        int owner_row = k % col_comm_size; // which row-position owns B's k-th block
        float val_A = (rank_row + 1) * 100.0f + (owner_col + 1) * 10.0f +k;
        float val_B = (owner_row + 1) * 100.0f + (rank_col + 1) * 10.0f + k;
        expected += Nb * val_A * val_B;
    }

    float max_abs_err = 0.0f;
    for (float v : h_C) {
        max_abs_err = std::max(max_abs_err, std::fabs(v - expected));
    }
    float rel_err = max_abs_err / (std::fabs(expected) + 1e-8f);
    // Use 1e-3f threshold to catch logic bug not a false-flagging fp32 imprecision rounding noise
    int local_pass = (rel_err < 1e-3f) ? 1 : 0;

    printf("[Rank %d (row=%d,col=%d)] expected=%.2f max_abs_err=%.6f rel_err=%.2e verified=%d\n",
       world_rank, rank_row, rank_col, expected, max_abs_err, rel_err, local_pass);

    int all_pass;
    MPI_Allreduce(&local_pass, &all_pass, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << (all_pass ? "SUMMA correctness verified across ALL ranks: PASS"
                            : "SUMMA correctness FAILED on at least one rank") << std::endl;
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

