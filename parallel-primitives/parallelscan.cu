
#include <iostream>
#include <cuda_runtime.h>

// Kogge-Stone Algorithm
// Time O(nlogn) --> work inefficient comparing to leetcode O(n)
// Space O(n) --> 2n for the temp, so big O is still O(n)
__global__ void native_scan_kernel(float* d_in, float* d_out, int n) {

    // Shared memory for the scan
    // first n of the shared memory is for in
    // second n of the shared memory is for out
    // therefore we need double buffer index pin and pout and swap them
    // make sure allocate it to 2 * blockDim.x * sizeof(float)
    extern __shared__ float temp[];

    int tx = threadIdx.x; // local index, for shared memory
    int b_size = blockDim.x;
    int global_idx = blockIdx.x * b_size + tx; // global index, for d_in, d_out

    // double Buffers
    int pin = 1, pout = 0;

    // load input to the temp shared memory
    // For inclusive prefix sum scan, index i contains (0, ..., i)
    // For exclusive prefix sum scan, index i contains (0, ..., i - 1)
    // below code line is for exclusive, that the last element is not loaded
    // Grid-Scale Exclusive Scan
    temp[pout * b_size + tx] = (global_idx > 0 && global_idx <= n) ? d_in[global_idx - 1] : 0.0f;

    __syncthreads(); // barrier to make sure all threads finish loading

    // every thread i adds the value at i - 2 ^(step - 1)
    for (int offset = 1; offset < n; offset <<= 1) {
        // swap pin and pout buffer
        pin = pout;
        pout = 1 - pout;

        if (tx >= offset) {
            temp[pout * b_size + tx] = temp[pin * b_size + tx] + temp[pin * b_size + tx - offset];
        }
        else {
            temp[pout * b_size + tx] = temp[pin * b_size + tx];
        }

        __syncthreads();
    }

    if (global_idx < n) {
        d_out[global_idx] = temp[pout * b_size + tx];
    }

}


// Brent-Kung Algorithm
// Up-sweep: index 2n - 1, 4n - 1, ...
// Down-sweep: stride starts with N/4
// Time O(n), space O(n)
__global__ void brent_kung_scan_kernel(float* d_in, float* d_out, int n) {

    // shared memory
    // The size of this temp should be blockDim.x * sizeof(float)
    extern __shared__ float temp[];

    int tx = threadIdx.x;
    int b_size = blockDim.x;

    int global_idx = blockIdx.x * b_size + tx;

    if (global_idx < n) {
        temp[tx] =d_in[global_idx];
    }
    else {
        temp[tx] = 0.0f;
    }

    __syncthreads()__;

    // Up-sweep
    // stride 1, 2, 4, ... , N/ 2 --> stride goes up to blockDim.x
    // working thread/element ids: i = 2 * stride * x - 1 --> (i + 1) % ( 2 * stride) == 0
    // 2 * stride the "2 *" here do make sure stride starts with 1
    // Thread index to data index mapping
    // To map a countinous section of threads to the XY k*2^n - 1 to reduce control drivergence
    // index = (threadIdx.x + 1) * 2 * stride - 1
    // XY[i] += XY[i - stride]
    for (int stride = 1; stride < b_size; stride <<= 1) {
        int index = (tx + 1) * 2 * stride - 1;
        if (index < b_size) {
            temp[index] += temp[index - stride];
        }

        __syncthreads();
    }

    // Down-sweep
    // stride starts with blockDim.x / 4, and down to 1
    // working thread/element ids: i + 1 = 2 * stride * x --> (i + 1) % ( 2 * stride) == 0
    // Thread index to data index mapping
    // index = (threadIdx + 1) * 2 * stride - 1
    // start point XY[i], target point XY[i + stride] += XY[i]
    for (int stride = b_size / 4; stride > 0; stride >>=1) { // /= 2
        int index = (tx + 1) * 2 * stride - 1;
        if (index + stride < b_size) {
            temp[index + stride] += temp[index];
        }

        __syncthreads();
    }

    if (global_idx < n) {
        d_out[global_idx] = temp[tx];
    }

}
