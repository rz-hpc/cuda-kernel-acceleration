
#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void MultiplyKernelNative(float *d_A, float *d_B, float *d_C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // maps to column in B
    int row = blockIdx.y * blockDim.y + threadIdx.y; // maps to row in A

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += d_A[row * n + i] * d_B[i * n + col];
        }
        d_C[row * n + col] = sum;
    }
}

__global__ void MultiplyKernelTiled(float *d_A, float *d_B, float *d_C, int n) {

    // Need two tiles, one for A one for B to be in the shared memory
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH + 1];

    // Identify the row and col of the product element to work on
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // decided by A's row
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // decided by B's col

    // Loop over the A and B tiles to get the dot product
    float p = 0.0f;
    // "m" moves like a sliding window across the matrix
    for (int m = 0; m < n / TILE_WIDTH; m++) {
        // Load the A and B tiles in the current window into the shared memory
        tile_A[threadIdx.y][threadIdx.x] = d_A[row * n + m * TILE_WIDTH + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = d_B[(threadIdx.y + m * TILE_WIDTH) * n + col];

        __syncthreads();

        // calculate product
        for (int k = 0; k < TILE_WIDTH; k++) {
          p += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Assign the p result to C
    d_C[row * n + col] = p;
}

__host__ void Multiply(float *A, float *B, float *C, int n) {

    int size = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 BlocksPerGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    //MultiplyKernelNative<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, n);
    MultiplyKernelTiled<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__host__ void run_benchmarks(float *A, float *B, float *C, int n) {

    int size = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 BlocksPerGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;

    //---Benchmark Native---
    cudaEventRecord(start);
    MultiplyKernelNative<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "MultiplyKernal Native elapsed time: " << milliseconds << " ms" << std::endl;

    //---Benchmark Tiled---
    cudaEventRecord(start);
    MultiplyKernelTiled<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "MultiplyKernal Kernel elapsed time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {

    int N = 1 << 10; // 1024;
    int size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    //Multiply(h_A, h_B, h_C, N);
    run_benchmarks(h_A, h_B, h_C, N);

    // Verify that C[0] should be 2.0f * N = 2048
    std::cout << h_C[0] << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
