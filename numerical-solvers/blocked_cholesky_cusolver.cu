
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include <cusolverDn.h> // need to compile with -lcusolver -lcublas

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("cuda Error: %s in %s at %d!\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t stat = call; \
    if (stat != CUSOLVER_STATUS_SUCCESS) { \
        printf("cuSolver Error: %d in %s at %d!\n", stat, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// A = L L^T
// A: SPD (symmetric positive-definite) N x N
// [[A11, A21^T], [A21, A22]] = [[L11, 0], [L21, L22]] [[L11^T, L21^T], [0, L22^T]]
// A11: b x b, A11 = L11 L11^T --> L11 = cholesky(A11)
// A21: (N - b) * b, A21 = L21 L11^T --> L21 = A21 (L11^T)^-1
// A22: (N - b) * (N - b), A22 = L21 L21^T + L22 L22^T --> A22' -= L21 L21^T

#define BLOCK_DIM 32

// kernel 1 portf-- square root diagonal
__global__ void portf_diag_kernel(float* d_A, int N, int k_block) {

    // shared memory tile
    __shared__ float tile_A[BLOCK_DIM][BLOCK_DIM + 1];

    // index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int global_row_idx = k_block * BLOCK_DIM + ty;
    int global_col_idx = k_block * BLOCK_DIM + tx;

    // load from global memory to shared memory
    if (global_row_idx < N && global_col_idx < N) {
        tile_A[ty][tx] = d_A[global_row_idx * N + global_col_idx];
    }
    else {
        tile_A[ty][tx] = 0.0f;
    }
    __syncthreads();

    // for each column k in the block
    for (int k = 0; k < BLOCK_DIM; k++) {
        // sqrt the diagonal
        if (tx == k && ty == k) {
            tile_A[ty][tx] = sqrtf(tile_A[k][k]);
        }
        __syncthreads();

        // update column
        if (tx == k && ty > tx) {
            tile_A[ty][tx] /= tile_A[k][k];
        }
        __syncthreads();

        // update trailing lower triangular matrix
        // symmetric rank-1 update Aij = Aij - Lik Ljk, k < j <= i < b
        if (tx > k && ty >= tx) {
            tile_A[ty][tx] -= tile_A[ty][k] * tile_A[tx][k];
        }
        __syncthreads();
    }

    // write back lower triangular and diagonal to the global memory
    if (global_row_idx < N && global_col_idx < N && ty >= tx) {
        d_A[global_row_idx * N + global_col_idx] = tile_A[ty][tx];
    }
    else if (global_row_idx < N && global_col_idx < N && ty < tx) {
        d_A[global_row_idx * N + global_col_idx] = 0.0f;
    }
}

// kernel 2: column update
// L21 = A21 (L11^T)^-1
// A21: (N - b) * b
__global__ void column_update_kernel(float* d_A, int N, int k_block) {
    // shared memory tiles for A21 and L11
    __shared__ float tile_A21[BLOCK_DIM][BLOCK_DIM + 1];
    __shared__ float tile_L11[BLOCK_DIM][BLOCK_DIM + 1];

    // index

    // block index
    int target_block_A21_row = k_block + 1 + blockIdx.x;
    int target_block_A21_col = k_block;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int target_A21_row = target_block_A21_row * BLOCK_DIM + ty;
    int target_A21_col = target_block_A21_col * BLOCK_DIM + tx;

    int diag_block_L11_row = k_block * BLOCK_DIM + ty;
    int diag_block_L11_col = k_block * BLOCK_DIM + tx;

    // load from global memory to tiles
    if (target_A21_row < N && target_A21_col < N) {
        tile_A21[ty][tx] = d_A[target_A21_row * N + target_A21_col];
    }
    else {
        tile_A21[ty][tx] = 0.0f;
    }
    if (diag_block_L11_row < N && diag_block_L11_col < N) {
        tile_L11[ty][tx] = d_A[diag_block_L11_row * N + diag_block_L11_col];
    }
    else {
        tile_L11[ty][tx] = 0.0f;
    }
    __syncthreads();

    // for each column in target A21
    for (int k = 0; k < BLOCK_DIM; k++) {

        // divide the pivot (diagonal)
        if (tx == k) {
            tile_A21[ty][tx] /= tile_L11[k][k];
        }
        __syncthreads();

        // column update (subtract the impact of pivot from column element)
        if (tx > k) {
            tile_A21[ty][tx] -= tile_A21[ty][k] * tile_L11[tx][k];
        }
        __syncthreads();
    }

    // write back to global memory
    if (target_A21_row < N && target_A21_col < N) {
        d_A[target_A21_row * N + target_A21_col] = tile_A21[ty][tx];
    }
}

// kernel 3: trailing matrix update
// A22' -= L21 L21^T
// A22: (N - b) x (N - b)
__global__ void trailing_matrix_kernel(float* d_A, int N, int k_block) {

    // shared memory tiles for L21 row and col
    __shared__ float tile_row_L21[BLOCK_DIM][BLOCK_DIM + 1];
    __shared__ float tile_col_L21[BLOCK_DIM][BLOCK_DIM + 1];

    // index

    // block index
    int target_block_A22_row = k_block + 1 + blockIdx.y;
    int target_block_A22_col = k_block + 1 + blockIdx.x;

    if (target_block_A22_row < target_block_A22_col) {
        return;
    }

    //int target_block_L21_row = k_block + 1 + blockIdx.y;
    //int target_block_L21_col = k_block;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int target_A22_row = target_block_A22_row * BLOCK_DIM + ty;
    int target_A22_col = target_block_A22_col * BLOCK_DIM + tx;

    int row_L21_row = target_A22_row;
    int row_L21_col = k_block * BLOCK_DIM + tx;

    int col_L21_row = target_block_A22_col * BLOCK_DIM + ty;
    int col_L21_col = k_block * BLOCK_DIM + tx;

    // load from global memory to shared memory
    if (row_L21_row < N && row_L21_col < N) {
        tile_row_L21[ty][tx] = d_A[row_L21_row * N + row_L21_col];
    }
    else {
        tile_row_L21[ty][tx] = 0.0f;
    }
    if (col_L21_row < N && col_L21_col < N) {
        tile_col_L21[ty][tx] = d_A[col_L21_row * N + col_L21_col];
    }
    else {
        tile_col_L21[ty][tx] = 0.0f;
    }
    __syncthreads();

    // for each column in the block
    float sum = 0.0f;
    for (int k = 0; k < BLOCK_DIM; k++) {

        // dot product
        sum += tile_row_L21[ty][k] * tile_col_L21[tx][k];
    }

    // subtract the dot product and write back the lower triangular and diagonal to the global memory
    if (target_A22_row < N && target_A22_col < N) {
        if ((target_block_A22_row > target_block_A22_col) || (target_block_A22_row == target_block_A22_col && ty >= tx)) {
            d_A[target_A22_row * N + target_A22_col] -= sum;
        }
    }
}

__host__ void blocked_cholesky_launcher(float* d_A, int N) {

    int total_blocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;

    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM, 1);

    for (int k_block = 0; k_block < total_blocks; k_block++) {
        // diagonal kernel
        portf_diag_kernel<<<1, threadsPerBlock>>>(d_A, N, k_block);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        int below_blocks = total_blocks - k_block - 1;

        // column update kernel
        if (below_blocks > 0) {
            column_update_kernel<<<below_blocks, threadsPerBlock>>>(d_A, N, k_block);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // trailing matrix update kernel
        if (below_blocks > 0) {
            dim3 blocksPerGrid(below_blocks, below_blocks, 1);
            trailing_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, N, k_block);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
}

// CPU reference
void cholesky_cpu_full(float* A, int N) {

    // check postivie definite
    for (int k = 0; k < N; k++) {
        float diag_val = A[k * N + k];
        assert(diag_val > 0.0f && "Input Error: A is not positive defined!");

        // sqrt diagonal
        A[k * N + k] = std::sqrt(diag_val);

        // column update
        for (int i = k + 1; i < N; i++) {
            A[i * N + k] /= A[k * N + k];
        }

        // trailing matrix update
        for (int j = k + 1; j < N; j++) {
            for (int i = j; i < N; i++) {
                A[i * N + j] -= A[i * N + k] * A[j * N + k];
            }
        }
    }

    // clear out the upper triangular
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            A[i * N + j] = 0.0f;
        }
    }
}

// Generate random SPD matrix
// A = L L^T, L should be positive-definite
// random: generate a random number betwee [lower_bound, upper_bound]
// val = rand() % (upper_bound - lower_bound + 1) + lower_bound
void generate_random_spd_matrix(float* A, int N) {
    std::vector<float> L(N * N, 0.0f);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                // diagonal has to be postive
                // generate a random float between [1, 10]
                L[i * N + j] = (float)(rand() % (10 - 1 + 1) + 1);
            }
            else if (i > j) {
                // off-diagonal could be negative
                // generate a random float between [-2, 2]
                L[i * N + j] = (float)(rand() % (2 + 2 + 1) - 2);
            }
        }
    }

    // A = L L^T
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += L[i * N + k] * L[j * N + k];
            }
            A[i * N + j] = sum;
        }
    }
}

bool verify_blocked_cholesky(float* result, float* reference, int N) {
    float epsilon = 1e-4f;
    float max_ref_err = 0.0f;

    // L is in-place in A's lower triangular and diagonal
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            float res = result[i * N + j];
            float ref = reference[i * N + j];
            float err = std::abs(res - ref) / (std::abs(ref) + 1e-8f);
            max_ref_err = std::max(max_ref_err, err);
        }
    }

    std::cout<< "Verify blocked cholesky max ref error " << max_ref_err
        << " tolerance: " << epsilon << std::endl;

    return max_ref_err < epsilon;
}

void run_cusolver_cholesky(float* d_A, int N) {

    cusolverDnHandle_t handle = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    int workspace_size = 0;
    int* d_info = nullptr;
    float* d_workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));

    // 1. Query Workspace Requirements
    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(handle,
                                              CUBLAS_FILL_MODE_UPPER,
                                              N,
                                              d_A,
                                              N, // leading dimension (lda)
                                              &workspace_size));

    // 2. Allocate GPU workspace memory
    CHECK_CUDA(cudaMalloc((void**)&d_workspace, workspace_size * sizeof(float)));

    // 3. Execute cuSolver Cholesky (PORTF)
    CHECK_CUSOLVER(cusolverDnSpotrf(handle,
                                    CUBLAS_FILL_MODE_UPPER,
                                    N,
                                    d_A,
                                    N, // leading dimension (lda)
                                    d_workspace,
                                    workspace_size,
                                    d_info));

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFree(d_workspace);
    cudaFree(d_info);
    CHECK_CUSOLVER(cusolverDnDestroy(handle));
}

int main() {

    srand(1337); // seed for reproducibility

    int N = 1024; // for NSight Profiling

    int sizeN = N * N * sizeof(float);

    std::vector<float> h_A_initial(N * N, 0.0f);
    std::vector<float> h_A_cpu_ref(N * N, 0.0f);
    std::vector<float> h_A(N * N, 0.0f);
    std::vector<float> h_A_cusolver(N * N, 0.0f);

    generate_random_spd_matrix(h_A_initial.data(), N);

    h_A_cpu_ref = h_A_initial;
    h_A = h_A_initial;
    h_A_cusolver = h_A_initial;

    cholesky_cpu_full(h_A_cpu_ref.data(), N);

    float* d_A;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeN));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeN, cudaMemcpyHostToDevice));

    blocked_cholesky_launcher(d_A, N);

    CHECK_CUDA(cudaMemcpy(h_A.data(), d_A, sizeN, cudaMemcpyDeviceToHost));

    bool isSuccess = verify_blocked_cholesky(h_A.data(), h_A_cpu_ref.data(), N);
    std::cout << "Customized Blocked Cholesky verified: " << isSuccess << std::endl;

    float* d_A_cusolver;
    CHECK_CUDA(cudaMalloc((void**)&d_A_cusolver, sizeN));
    CHECK_CUDA(cudaMemcpy(d_A_cusolver, h_A_cusolver.data(), sizeN, cudaMemcpyHostToDevice));
    
    run_cusolver_cholesky(d_A_cusolver, N);

    CHECK_CUDA(cudaMemcpy(h_A_cusolver.data(), d_A_cusolver, sizeN, cudaMemcpyDeviceToHost));

    bool iscusolverSuccess = verify_blocked_cholesky(h_A_cusolver.data(), h_A_cpu_ref.data(), N);
    std::cout << "cuSolver Blocked Cholesky verified: " << iscusolverSuccess << std::endl;

    cudaFree(d_A_cusolver);
    cudaFree(d_A);

    return 0;
}

