
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cstdlib>


#define BLOCK_DIM 32

// Step 2 of blocked cholesky--update column below diagonal
// A21 = L21 L11^T -> L21 = A21(L11^T)^-1
// Each row of A21 being processed independently in parallel
__global__ void column_update_kernel(float* d_A, int n, int k_block) {

    // shared memory for L11 (from diagnoal factorization kernel result) and A21
    __shared__ float tile_L11[BLOCK_DIM][BLOCK_DIM + 1];
    __shared__ float tile_A21[BLOCK_DIM][BLOCK_DIM + 1];

    // index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // which block below diagonal this thread block is assigned to (panel A21)
    int target_block_row = k_block + 1 + blockIdx.x;

    // global row/col coordinate for L11
    int diag_row = k_block * BLOCK_DIM + ty;
    int diag_col = k_block * BLOCK_DIM + tx;

    // global row/col coordinate for A21
    int target_row = target_block_row * BLOCK_DIM + ty;
    int target_col = k_block * BLOCK_DIM + tx; // same as diag_col

    // Load A21 and L11 to shared memory Tiles
    if (diag_row < n && diag_col < n) {
        tile_L11[ty][tx] = d_A[diag_row * n + diag_col];
    }
    else {
        tile_L11[ty][tx] = 0.0f;
    }
    if (target_row < n && target_col < n) {
        tile_A21[ty][tx] = d_A[target_row * n + target_col];
    }
    else {
        tile_A21[ty][tx] = 0.0f;
    }
    __syncthreads();

    // Triangle Solve Math (TRSM)
    // For each column k in the diagonal L11 block
    for (int k = 0; k < BLOCK_DIM; k++) {

        // Update the column below diagonal A21[j][k] /= L11[k][k]
        // every thread in k column does the scale
        if (tx == k) {
            tile_A21[ty][k] /= tile_L11[k][k];
        }
        __syncthreads();

        // Rank-1 update and broadcast to the remaining columns of A21 by using L11^T's mathing row entries
        // A21[i][j] -= A21[i][k] * L11[j][k] (L11^T row k is L11 col k)
        if (tx > k) {
            tile_A21[ty][tx] -= tile_A21[ty][k] * tile_L11[tx][k];
        }
        __syncthreads();
    }

    // Write the updated panel block to global memory
    if (target_row < n && target_col < n) {
        d_A[target_row * n + target_col] = tile_A21[ty][tx];
    }
}

// step 1 that already verified yesterday
__global__ void diagnoal_factorization_kernel(float* d_A, int n, int k_block) {

    __shared__ float tile_A[BLOCK_DIM][BLOCK_DIM + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int global_row_idx = k_block * BLOCK_DIM + ty;
    int global_col_idx = k_block * BLOCK_DIM + tx;

    if (global_row_idx < n && global_col_idx < n) {
        tile_A[ty][tx] = d_A[global_row_idx * n + global_col_idx];
    }
    else {
        tile_A[ty][tx] = 0.0f;
    }
    __syncthreads();

    // for each column k in the block
    for (int k = 0; k < BLOCK_DIM; k++) {
        // sqrt diagonal
        if (tx == k && ty == k) {
            tile_A[k][k] = sqrtf(tile_A[k][k]);
        }
        __syncthreads();

        // update the column below diagnoal
        if (tx == k && ty > tx) {
            tile_A[ty][k] /= tile_A[k][k];
        }
        __syncthreads();

        // trailing matrix update for the lower triangle (j > k, i >= j)
        if (tx > k && ty >= tx) {
            tile_A[ty][tx] -= tile_A[ty][k] * tile_A[tx][k];
        }
        __syncthreads();
    }

    // write values to the lower triangle and clean the upper triangle
    if (global_row_idx < n && global_col_idx < n && ty >= tx) {
        d_A[global_row_idx * n + global_col_idx] = tile_A[ty][tx];
    }
    else if (global_row_idx < n && global_col_idx < n && ty < tx) {
        d_A[global_row_idx * n + global_col_idx] = 0.0f;
    }
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// CPU reference -- full matrix complete cholesky
void cholesky_cpu_full(std::vector<float>& A, int n) {
    for (int k = 0; k < n; k++) {
        float val = A[k * n + k];
        assert(val > 0.0f && "Assert failure: matrix is not positive definite!");

        // sqrt
        A[k * n + k] = std::sqrt(val);

        // column update
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
        }

        // trailing matrix update
        for (int j = k + 1; j < n; j++) {
            for (int i = j; i < n; i++) {
                A[i * n + j] -= A[i * n + k] * A[j * n + k];
            }
        }

    }

    // clean out the upper triangle
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[i * n + j] = 0.0f;
        }
    }
}

// random symmetric positive definite matrix Generator
// To generate a random number in the range [lower_bound, upper_bound]:
// rand() % (upper_bound - lower_bound + 1) + lower_bound
// A = L L^T
void generate_random_spd_matrix(float* A, int n) {
    std::vector<float> L(n * n, 0.0f);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            // Diagnoal strictly positive
            if (i == j) {
                L[i * n + j] = (float)(rand() % 10 + 1); // random float in [1, 10]
            }
            // off diagnoal could be negative
            else {
                L[i * n + j] = (float)(rand() % 5 - 2); // random float in [-2, 6]
            }
        }
    }

    // compute A = LL^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
}

int main() {

    srand(1337); // seed for reproducibility

    int N = 64;
    int bytes = N * N * sizeof(float);

    std::vector<float> h_A_initial(N * N);
    std::vector<float> h_A_cpu(N * N);
    std::vector<float> h_A_gpu(N * N);

    generate_random_spd_matrix(h_A_initial.data(), N);
    h_A_cpu = h_A_initial;
    h_A_gpu = h_A_initial;

    // cpu reference
    cholesky_cpu_full(h_A_cpu, N);

    float* d_A = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMemcpy(d_A, h_A_gpu.data(), bytes, cudaMemcpyHostToDevice));

    int total_blocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);

    for (int k_block = 0; k_block < total_blocks; k_block++) {
        // kernel 1: cholesky, factorize diagonal block        
        diagnoal_factorization_kernel<<<1, threadsPerBlock>>>(d_A, N, k_block);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // kernel 2: update column panel below the diagonal
        int blocks_below = total_blocks - k_block - 1;
        if (blocks_below > 0) {
            column_update_kernel<<<blocks_below, threadsPerBlock>>>(d_A, N, k_block);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // kernel 3 (TODO): trailing matrix subtract
    }

    CHECK_CUDA(cudaMemcpy(h_A_gpu.data(), d_A, bytes, cudaMemcpyDeviceToHost));

    // Verification
    bool isSuccess = true;
    const float epsilon = 1e-4f;

    // verify the A11 and A21 part (kernel 3 TODO)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < BLOCK_DIM; j++) { // first column panel width
            if (i >= j) { // lower triangle part
                if (std::abs(h_A_cpu[i * N + j] - h_A_gpu[i * N + j]) > epsilon) {
                    isSuccess = false;
                    std::cout << "Mismatch element at row " << i << " col " << j
                              << " Expected CPU value " << h_A_cpu[i * N + j]
                              << " but got GPU value " << h_A_gpu[i * N + j] << std::endl;
                    break;
                }
            }
        }
    }

    std::cout << "Blocked Cholesky diagonal factorization and column panel update kernels isSuccess: "
     << isSuccess << std::endl;

    cudaFree(d_A);

    return 0;
}
