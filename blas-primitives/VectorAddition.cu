
#include <iostream>
#include <cuda_runtime.h>

// Standard macro for error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void vecAddKernel(const float *A, const float *B, float *C, int n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

__host__ void vecAdd(const float *A, const float *B, float *C, int n, int offset) {
    float *d_A, *d_B, *d_C;
    int size = n * sizeof(float);

    // Device Allocation
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Memcpy Host to Device
    CHECK_CUDA(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // ceil(n/256.0)

    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n, offset);

    cudaDeviceSynchronize();

    // Memcpy Device to Host
    CHECK_CUDA(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
  }

int main() {

    int N = 1 << 20; // 2^20 = 1,048,576 elements
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;

    // Allocate Host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize Host vectors
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    vecAdd(h_A, h_B, h_C, N, 0);

    for (int i = 0; i < 10; i++)
    {
      std::cout << "h_C " << h_C[i] << std::endl;
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
