
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define SHMEM_INDEX(i) (((i) + ((i) >> 5)))

__global__ void scan_and_export_sums(float* d_in, float* d_out, float* d_block_sums, int n) {
    // size: (blockDim.x + blockDim.x/32) * sizeof(float)
    extern __shared__ float temp[];

    int tx = threadIdx.x;
    int b_size = blockDim.x;
    int global_idx = b_size * blockIdx.x + tx;

    // Load
    if (global_idx < n) {
        temp[SHMEM_INDEX(tx)] = d_in[global_idx];
    }
    else {
        temp[SHMEM_INDEX(tx)] = 0.0f;
    }
    __syncthreads();

    // Up-sweep
    // element index 2n - 1, 4n - 1, ...
    // continous threads to element mapping index = (tx + 1) * 2 * stride - 1
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

    // last thread to update block sums
    if (tx == b_size - 1) {
        d_block_sums[blockIdx.x] = temp[SHMEM_INDEX(b_size - 1)];
    }

    // Down-sweep
    // stride starts with b_size / 4
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

    //Write to output
    if (global_idx < n) {
        d_out[global_idx] = temp[SHMEM_INDEX(tx)];
    }

}

int main() {

    int N = 1000000;

    // Unified memory
    float* data;
    CHECK_CUDA(cudaMallocManaged((void**)&data, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        data[i] = 1.0f;
    }

    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int shmemSize = sizeof(float) * (threadsPerBlock + (threadsPerBlock >> 5));

    float* d_block_sum;
    CHECK_CUDA(cudaMalloc((void**)&d_block_sum, sizeof(float) * blocksPerGrid));

    scan_and_export_sums<<<blocksPerGrid, threadsPerBlock, shmemSize>>>(data, data, d_block_sum, N);

    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Last element: %f\n", data[N - 1]);

    cudaFree(data);
}
