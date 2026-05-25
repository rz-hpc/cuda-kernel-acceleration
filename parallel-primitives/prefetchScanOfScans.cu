
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define SHMEM_INDEX(i) (((i) + ((i) >> 5)))

__global__ void scan_and_export_sums(float* d_in, float* d_out, float* d_block_sums, int n) {

    // shared memory size to be (blockDim.x + blockDim.x / 32 ) * sizeof(float)
    extern __shared__ float temp[];

    int tx = threadIdx.x;
    int b_size = blockDim.x;
    int global_idx = b_size * blockIdx.x + tx;

    // Load input to shared memory
    if (global_idx < n) {
        temp[SHMEM_INDEX(tx)] = d_in[global_idx];
    }
    else {
        temp[SHMEM_INDEX(tx)] = 0.0f;
    }
    __syncthreads();

    // Up-sweep XY[i] += XY[i - stride]
    // element index: 2n - 1, 4n - 1, ...
    // stride 1, 2, ..., up to b_size / 2
    // continous threads to element index mapping:
    // index = (tx + 1) * 2 * stride - 1
    for (int stride = 1; stride < b_size; stride <<= 1) {
        int num_active_threads = b_size / (2 * stride);
        if (tx < num_active_threads) {
            int index = (tx + 1) * 2 * stride - 1;
            if (index < b_size) {
                temp[SHMEM_INDEX(index)] += temp[SHMEM_INDEX(index - stride)];
            }
        }
        __syncthreads();
    }

    // Export to block sums (last thread)
    if (tx == b_size - 1) {
        d_block_sums[blockIdx.x] = temp[SHMEM_INDEX(tx)];
    }

    // Down-sweep XY[i + stride] += XY[i]
    // stride b_size / 4, ... , 1
    // element index 2 * b_size / 4 * n - 1, ...
    // Threads to element index mapping:
    // index = (tx + 1) * 2 * stride - 1
    for (int stride = b_size / 4; stride > 0; stride >>= 1) {
        int num_active_threads = b_size / (2 * stride);
        if (tx < num_active_threads) {
            int index = (tx + 1) * 2 * stride - 1;
            if (index + stride < b_size) {
                  temp[SHMEM_INDEX(index + stride)] += temp[SHMEM_INDEX(index)];
            }
        }
        __syncthreads();
    }

    // Write to output
    if (global_idx < n) {
        d_out[global_idx] = temp[SHMEM_INDEX(tx)];
    }
}

int main() {

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device: %s\n", prop.name);
  printf("Concurrent Managed Access: %d\n", prop.concurrentManagedAccess);

    int N = 1024;
    int sizeN = N * sizeof(float);

    // Unified memory
    float* data;
    CHECK_CUDA(cudaMallocManaged((void**)&data, sizeN));

    for (int i = 0; i < N; i++) data[i] = 1.0f;

    // Prefetch
    //int deviceId;
    //cudaGetDevice(&deviceId);

    //CHECK_CUDA(cudaStreamPrefetchAsync(data, sizeN, deviceId));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int shmemSize = sizeof(float) * (threadsPerBlock + (threadsPerBlock >> 5));

    float* d_block_sums;
    CHECK_CUDA(cudaMalloc((void**)&d_block_sums, blocksPerGrid * sizeof(float)));

    scan_and_export_sums<<<blocksPerGrid, threadsPerBlock, shmemSize>>>(data, data, d_block_sums, N);

    // Important
    CHECK_CUDA(cudaDeviceSynchronize());

    // Prefetch back to CPU
    //CHECK_CUDA(cudaStreamPrefetchAsync(data, sizeN, cudaCpuDeivceId));
    //CHECK_CUDA(cudaDeviceSynchronize());

    printf("last element: %f\n", data[N-1]);

    cudaFree(data);
    cudaFree(d_block_sums);

    return 0;
}
