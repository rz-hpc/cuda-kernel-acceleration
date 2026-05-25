
#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void MultiplyKernelNative(float *d_A, float *d_B, float *d_C, int m, int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // maps to column in C
    int row = blockIdx.y * blockDim.y + threadIdx.y; // maps to row in C

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += d_A[row * k + i] * d_B[i * n + col];
        }
        d_C[row * n + col] = sum;
    }
}

__global__ void MultiplyKernelTiled(float *d_A, float *d_B, float *d_C, int m, int k, int n) {

    // Need two tiles, one for A one for B to be in the shared memory
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH + 1];

    // Identify the row and col of the product element in C to work on
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // decided by A's row
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // decided by B's col

    // Loop over the A and B tiles to get the dot product
    float p = 0.0f;
    // The loop runs over the 'k' dimension (inner dimension)
    for (int w = 0; w < (k + TILE_WIDTH - 1) / TILE_WIDTH; w++) {

        // Load Tile A: row is constant for this thread, column shifts by w
        int A_col = w * TILE_WIDTH + threadIdx.x;
        if (row < m && A_col < k) {
            tile_A[threadIdx.y][threadIdx.x] = d_A[row * k + A_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load Tile B: column is constant for this thread, row shifts by w
        int B_row = w * TILE_WIDTH + threadIdx.y;
        if (B_row < k && col < n) {
            tile_B[threadIdx.y][threadIdx.x] = d_B[B_row * n + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int pi = 0; pi < TILE_WIDTH; pi++) {
            p += tile_A[threadIdx.y][pi] * tile_B[pi][threadIdx.x];
        }

        __syncthreads();
    }

    // Assign the p result to C
    if (row < m && col < n) {
        d_C[row * n + col] = p;
    }
}

__host__ void Multiply(float *A, float *B, float *C, int m, int k, int n) {

    int sizeA = m * k * sizeof(float);
    int sizeB = k * n * sizeof(float);
    int sizeC = m * n * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 BlocksPerGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    //MultiplyKernelNative<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
    MultiplyKernelTiled<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__host__ void run_benchmarks(float *A, float *B, float *C, int m, int k, int n) {

    int sizeA = m * k * sizeof(float);
    int sizeB = k * n * sizeof(float);
    int sizeC = m * n * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 BlocksPerGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;

    //---Benchmark Native---
    cudaEventRecord(start);
    MultiplyKernelNative<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "MultiplyKernal Native elapsed time: " << milliseconds << " ms" << std::endl;

    //---Benchmark Tiled---
    cudaEventRecord(start);
    MultiplyKernelTiled<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "MultiplyKernal Kernel elapsed time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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

    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f;
    }

    for (int j = 0; j < K * N; j++) {
        h_B[j] = 2.0f;
    }

    //Multiply(h_A, h_B, h_C, M, K, N);
    run_benchmarks(h_A, h_B, h_C, M, K, N);

    // Verify that C[i] should be 2.0f * K = 1000.0f
    bool isSuccess = true;
    for (int i = 0; i < M * K; i++) {
        if (h_C[i] != 1000.0f) {
            //std::cout << "h_C " << h_C[0] << " i " << i << std::endl;
            isSuccess = false;
            break;
        }
    }
    std::cout << "Multiplication succeed? " << isSuccess << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
