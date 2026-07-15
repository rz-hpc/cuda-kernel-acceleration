
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE); \
    }\
}

#define TILE_WIDTH 16
#define THREAD_TILE_SIZE 4

__global__ void mutiplyKernelTiledRegisterPadding(float* d_A, float* d_B, float* d_C, int m, int k, int n) {

    // Register tile the 4 X 4 patch
    float accum[THREAD_TILE_SIZE][THREAD_TILE_SIZE];

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_SIZE; i++) {
        for (int j = 0; j < THREAD_TILE_SIZE; j++) {
            accum[i][j] = 0.0f;
        }
    }

    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Start index of the 4 X 4 path in the output matrix
    int row_start = blockIdx.y * TILE_WIDTH + ty * THREAD_TILE_SIZE;
    int col_start = blockIdx.x * TILE_WIDTH + tx * THREAD_TILE_SIZE;

    // A block owns a shared memory tile, sliding window of the tile
    for (int w = 0; w < (k + TILE_WIDTH - 1) / TILE_WIDTH; w++) {

        // Load A and B to the shared
        // Within a block, a thread owns a 4 X 4 patch
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i += THREAD_TILE_SIZE) {
            for (int j = 0; j < TILE_WIDTH; j += THREAD_TILE_SIZE) {
                int shared_row = ty + i;
                int shared_col = tx + j;

                // A is vertical, to get a patch of A, needs thread row offsets by ty * 4 then moves down i rows
                int global_row_A = blockIdx.y * TILE_WIDTH + shared_row;
                int global_col_A = w * TILE_WIDTH + shared_col;
                if (global_row_A < m && global_col_A < k) {
                    tile_A[shared_row][shared_col] = d_A[global_row_A * k + global_col_A];
                }
                else {
                    tile_A[shared_row][shared_col] = 0.0f;
                }

                // B is horizontal, to get a patch of B, needs thread col offset by tx * 4 then moves right j columns
                int global_row_B = w * TILE_WIDTH + shared_row;
                int global_col_B = blockIdx.x * TILE_WIDTH + shared_col;
                if (global_row_B < k && global_col_B < n) {
                    tile_B[shared_row][shared_col] = d_B[global_row_B * n + global_col_B];
                }
                else {
                    tile_B[shared_row][shared_col] = 0.0f;
                }

            }
        }

        __syncthreads();

        // Outer product of the shared memory tile
        // The thread to grab k_inst col of A and k_inst row of B
        for (int k_inst = 0; k_inst < TILE_WIDTH; k_inst++) {

            // for each patch within the tile
            float fragA[THREAD_TILE_SIZE];
            float fragB[THREAD_TILE_SIZE];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                fragA[i] = tile_A[ty * THREAD_TILE_SIZE + i][k_inst];
                fragB[i] = tile_B[k_inst][tx * THREAD_TILE_SIZE + i];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                    accum[i][j] += fragA[i] * fragB[j];
                }
            }
        }

        __syncthreads();

    }

    // Write the patch result to the output C
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_SIZE; i++) {
        for (int j = 0; j < THREAD_TILE_SIZE; j++) {
            if (row_start + i < m && col_start + j < n) {
                d_C[(row_start + i) * n + (col_start + j)] = accum[i][j];
            }
        }
    }

}


#define THREAD_TILE_W 2  // New width
#define THREAD_TILE_H 4  // New height

__global__ void mutiplyKernelTiledRegisterPaddingWith8By4Thread(float* d_A, float* d_B, float* d_C, int m, int k, int n) {

    // Register tile the 4 X 2 patch (the workload per worker)
    float accum[THREAD_TILE_H][THREAD_TILE_W];

    for (int i = 0; i < THREAD_TILE_H; i++) {
        for (int j = 0; j < THREAD_TILE_W; j++) {
            accum[i][j] = 0.0f;
        }
    }

    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Start index of the 4 X 2 patch in the output matrix
    int row_start = blockIdx.y * TILE_WIDTH + ty * THREAD_TILE_H;
    int col_start = blockIdx.x * TILE_WIDTH + tx * THREAD_TILE_W;

    // A block owns a shared memory tile, sliding window of the tile
    for (int w = 0; w < (k + TILE_WIDTH - 1) / TILE_WIDTH; w++) {

        // Load A and B to the shared
        // Within a block, a thread owns a 4 X 4 patch
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i += 4) {
            for (int j = 0; j < TILE_WIDTH; j += 8) {
                int shared_row = ty + i;
                int shared_col = tx + j;

                // A is vertical, to get a patch of A, needs thread row offsets by ty * 4 then moves down i rows
                int global_row_A = blockIdx.y * TILE_WIDTH + shared_row;
                int global_col_A = w * TILE_WIDTH + shared_col;
                if (global_row_A < m && global_col_A < k) {
                    tile_A[shared_row][shared_col] = d_A[global_row_A * k + global_col_A];
                }
                else {
                    tile_A[shared_row][shared_col] = 0.0f;
                }

                // B is horizontal, to get a patch of B, needs thread col offset by tx * 4 then moves right j columns
                int global_row_B = w * TILE_WIDTH + shared_row;
                int global_col_B = blockIdx.x * TILE_WIDTH + shared_col;
                if (global_row_B < k && global_col_B < n) {
                    tile_B[shared_row][shared_col] = d_B[global_row_B * n + global_col_B];
                }
                else {
                    tile_B[shared_row][shared_col] = 0.0f;
                }

            }
        }

        __syncthreads();

        // Outer product of the shared memory tile
        // The thread to grab k_inst col of A and k_inst row of B
        for (int k_inst = 0; k_inst < TILE_WIDTH; k_inst++) {

            // for each patch within the tile
            float fragA[THREAD_TILE_H];
            float fragB[THREAD_TILE_W];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_H; i++) {
                fragA[i] = tile_A[ty * THREAD_TILE_H + i][k_inst];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_W; j++) {
                fragB[j] = tile_B[k_inst][tx * THREAD_TILE_W + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_H; i++) {
                for (int j = 0; j < THREAD_TILE_W; j++) {
                    accum[i][j] += fragA[i] * fragB[j];
                }
            }
        }

        __syncthreads();

    }

    // Write the patch result to the output C
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_H; i++) {
        for (int j = 0; j < THREAD_TILE_W; j++) {
            if (row_start + i < m && col_start + j < n) {
                d_C[(row_start + i) * n + (col_start + j)] = accum[i][j];
            }
        }
    }

}

__host__ void Multiply(float* h_A, float* h_B, float* h_C, int m, int k, int n) {

    int sizeA = m * k * sizeof(float);
    int sizeB = k * n * sizeof(float);
    int sizeC = m * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    //dim3 ThreadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    //dim3 BlocksPerGrid( (n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    //multiplyKernelNative<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
    //mutiplyKernelTiled<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
    //mutiplyKernelTiledVectorizedA<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    // Still seeing bank conflicts
    //dim3 ThreadsPerBlock(TILE_WIDTH / THREAD_TILE_SIZE, TILE_WIDTH / THREAD_TILE_SIZE, 1);
    //dim3 BlocksPerGrid( (n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    //mutiplyKernelTiledRegisterPadding<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    // To this (32 threads = 1 full Warp):
    dim3 ThreadsPerBlock(8, 4, 1);
    dim3 BlocksPerGrid( (n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    mutiplyKernelTiledRegisterPaddingWith8By4Thread<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}


int main() {

    int M = 1000;
    int K = 500;
    int N = 1000;

    int sizeA = M * K * sizeof(float);
    int sizeB = K * N * sizeof(float);
    int sizeC = M * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(sizeA);
    h_B = (float*)malloc(sizeB);
    h_C = (float*)malloc(sizeC);

    for (int ia = 0; ia < M * K; ia++) {
        h_A[ia] = 1.0f;
    }
    for (int ib = 0; ib < K * N; ib++) {
        h_B[ib] = 2.0f;
    }

    Multiply(h_A, h_B, h_C, M, K, N);

    bool isSuccess = true;
    for (int ic = 0; ic < M * N; ic++){
        if (h_C[ic] != 1.0f * 2.0f * K) {
            isSuccess = false;
            break;
        }
    }
    std::cout << "isSuccess: " << isSuccess << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

