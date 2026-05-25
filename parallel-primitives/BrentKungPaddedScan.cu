
#include <iostream>
#include <cuda_runtime.h>

// Padding: add a "ghost column" at the end of every 32 elements
// i = i + i / 32
// For element (0, ... 31), i = i
// For element (32, ... ,63), i = i + 1
// For element (64, ..., 95), i = i + 2
// i / 32 is moving bits to the right by 5 digits (2 ^ 5 = 32)
#define SHMEM_INDEX(i) (((i) + ((i) >> 5)))

__global__ void brent_kung_scan_kernel(float* d_in, float* d_out, int n) {

    // Shared memory
    // The size of temp should be blockDim.x * sizeof(float)
    extern __shared__ float temp[];

    int tx = threadIdx.x;
    int b_size = blockDim.x;
    int global_idx = blockIdx.x * b_size + tx;

    // Load input
    if (global_idx < n) {
        temp[tx] = d_in[global_idx];
    }
    else {
        temp[tx] = 0.0f;
    }

    __syncthreads();

    // Up-sweep
    // XY[i] += XY[i - stride]
    // Element index i: 2n - 1, 4n - 1, etc --> (i + 1) % (stride * 2) == 0
    // Stride starts with 1, 2, 4, ..., blockDim.x / 2
    // To reduce thread control divergence, mapping continous threads to element indices
    // i = (threadIdx.x + 1) * 2 * stride - 1
    // This mapping also makes sure i >= stride
    for (int stride = 1; stride < b_size; stride <<= 1) {

        // optimize for power-- prevent idle threads from calculating the index
        int num_active_threads = b_size / (2 * stride);
        if (tx < num_active_threads) {
            int index = (tx + 1) * 2 * stride - 1;
            if (index < b_size){
                temp[index] += temp[index - stride];
            }
        }

        __syncthreads();
    }

    // Down-sweep
    // XY[i + stride] += XY[i]
    // Stride starts with b_size / 4, b_size / 8, ... , 1
    // Element index 2 * stride - 1
    // The continous threads to element indices mapping:
    // i = (threadIdx.x + 1) * 2 * stride - 1
    for (int stride = b_size / 4; stride > 0; stride >>= 1) {

        // save power from preventing idle threads calculating index
        int num_active_threads = b_size / (2 * stride);
        if (tx < num_active_threads) {
            int index = (tx + 1) * 2 * stride - 1;
            if (index + stride < b_size) {
                temp[index + stride] += temp[index];
            }
        }

        __syncthreads();
    }

    // Write to output
    if (global_idx < n) {
        d_out[global_idx] = temp[tx];
    }
}

__global__ void brent_kung_padded_kernel(float* d_in, float* d_out, int n) {

    // Shared memory
    // Shared memory size needs to be (b_size + b_size/32) * sizeof(float)
    extern __shared__ float temp[];

    int tx = threadIdx.x;
    int b_size = blockDim.x;
    int global_idx = blockIdx.x * b_size + tx;

    // Load Input
    if (global_idx < n) {
        temp[SHMEM_INDEX(tx)] = d_in[global_idx];
    }
    else {
        temp[SHMEM_INDEX(tx)] = 0.0f;
    }

    __syncthreads();

    // Up-sweep
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

    // Down-sweep
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

    // Write Output
    if (global_idx < n) {
        d_out[global_idx] = temp[SHMEM_INDEX(tx)];
    }
}
