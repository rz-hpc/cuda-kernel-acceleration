
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }\
}

#define TILE_WIDTH 16
#define THREAD_TILE_H 4
#define THREAD_TILE_W 2

__global__ void multiplyKernelDoubleBuffering(float *d_A, float *d_B, float *d_C, int m, int k, int n) {

    // Register tile/patch (workload per worker/thread) for computing
    float accum[THREAD_TILE_H][THREAD_TILE_W];

    #pragma unroll
    for (int i = 0 ; i < THREAD_TILE_H; i++) {
        for (int j = 0; j < THREAD_TILE_W; j++) {
            accum[i][j] = 0.0f;
        }
    }

    // Shared memory tiles with padding for bank conflicts
    __shared__ float tileA[2][TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float tileB[2][TILE_WIDTH][TILE_WIDTH + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Result patch/tile (workload per thread) starting coordinates in the output matrix
    // A is vertical, inner dimension k will disappear in the result
    int row_start = blockIdx.y * TILE_WIDTH + ty * THREAD_TILE_H;
    // B is horizontal, inner dimension k will disapper in the result
    int col_start = blockIdx.x * TILE_WIDTH + tx * THREAD_TILE_W;

    // Prologue-- load the w = 0 data to the tiles
    for (int i = 0; i < TILE_WIDTH; i += 4) { // thread block y dim
          for (int j = 0; j < TILE_WIDTH; j += 8) { // thread block x dim
                int shared_row = ty + i;
                int shared_col = tx + j;

                int global_row_A = blockIdx.y * TILE_WIDTH + shared_row;
                int global_col_A = shared_col;
                if (global_row_A < m && global_col_A < k) {
                    tileA[0][shared_row][shared_col] = d_A[global_row_A * k + global_col_A];
                }
                else {
                    tileA[0][shared_row][shared_col] = 0.0f;
                }

                int global_row_B = shared_row;
                int global_col_B = blockIdx.x * TILE_WIDTH + shared_col;
                if (global_row_B < k && global_col_B < n) {
                    tileB[0][shared_row][shared_col] = d_B[global_row_B * n + global_col_B];
                }
                else {
                    tileB[0][shared_row][shared_col] = 0.0f;
                }
            }
    }

    __syncthreads();


    // Sliding window of shared memory tiles
    for (int w = 0; w < (k + TILE_WIDTH - 1) / TILE_WIDTH; w++) {
        // Phased loop
        int curr_phase = w % 2;
        int next_phase = 1 - curr_phase;

        __syncthreads();

        // Async load -- load next tile into the next_phase
        if (w + 1 < (k + TILE_WIDTH - 1) / TILE_WIDTH) {

            int next_w = w + 1;

            // load the global A B to tile[next_phase][][] with next_w data
            for (int i = 0; i < TILE_WIDTH; i += 4) { // thread block y dim
              for (int j = 0; j < TILE_WIDTH; j += 8) { // thread block x dim
                int shared_row = ty + i;
                int shared_col = tx + j;

                int global_row_A = blockIdx.y * TILE_WIDTH + shared_row;
                int global_col_A = next_w * TILE_WIDTH + shared_col;
                if (global_row_A < m && global_col_A < k) {
                    tileA[next_phase][shared_row][shared_col] = d_A[global_row_A * k + global_col_A];
                }
                else {
                    tileA[next_phase][shared_row][shared_col] = 0.0f;
                }

                int global_row_B = next_w * TILE_WIDTH + shared_row;
                int global_col_B = blockIdx.x * TILE_WIDTH + shared_col;
                if (global_row_B < k && global_col_B < n) {
                    tileB[next_phase][shared_row][shared_col] = d_B[global_row_B * n + global_col_B];
                }
                else {
                    tileB[next_phase][shared_row][shared_col] = 0.0f;
                }
            }
          }
        }

        // Compute with the curr_phase data in tiles
        // Fill the Register patch with outer product results from shared memory tile
        // The thread to grab k_inst col of A and k_inst row of B
        for (int k_inst = 0; k_inst < TILE_WIDTH; k_inst++) {

            // for each patch of the tile
            float fragA[THREAD_TILE_H];
            float fragB[THREAD_TILE_W];

            #pragma unroll
            for (int i = 0 ; i < THREAD_TILE_H; i++) {
                fragA[i] = tileA[curr_phase][ty * THREAD_TILE_H + i][k_inst];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_W; j++) {
                fragB[j] = tileB[curr_phase][k_inst][tx * THREAD_TILE_W + j];
            }

            #pragma unroll
            for (int i = 0 ; i < THREAD_TILE_H; i++) {
                for (int j = 0; j < THREAD_TILE_W; j++) {
                    accum[i][j] += fragA[i] * fragB[j];
                }
            }
        }

        __syncthreads();

    }

    // Write the 4 x 2 patch to the output result
    #pragma unroll
    for (int i = 0 ; i < THREAD_TILE_H; i++) {
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

    // To this (32 threads = 1 full Warp):
    dim3 ThreadsPerBlock(8, 4, 1);
    dim3 BlocksPerGrid( (n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    multiplyKernelDoubleBuffering<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

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
