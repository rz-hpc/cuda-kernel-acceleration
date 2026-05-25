
#include <iostream>
#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>

void verify_softmax(float* h_in, float* h_out, int n) {
    float max_val = -1e20f;
    for (int i = 0; i < n; i++) if (h_in[i] > max_val) max_val = h_in[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(h_in[i] - max_val);

    float total_prob = 0.0f;
    bool match = true;
    for (int i = 0; i < n; i++) {
        float expected = expf(h_in[i] - max_val) / sum;
        total_prob += h_out[i];
        if (std::abs(h_out[i] - expected) > 1e-5) {
            match = false;
        }
    }

    std::cout << "--- Validation Results ---" << std::endl;
    std::cout << "Max Error Match: " << (match ? "PASS" : "FAIL") << std::endl;
    std::cout << "Total Probability Sum: " << total_prob << " (Should be ~1.0)" << std::endl;
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// verifiy result

// Helper function for warp reduction
__device__ float warpReduceMax(float val) {
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 32/ 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmaxWarpShufflesKernel(float *d_in, float *d_out, int width) {
    // Only need 32 floats for warp shared memory
    __shared__ float warpMaxes[32];
    __shared__ float warpSums[32];

    // For softmax, one block per row
    int tid = threadIdx.x;
    int lane = tid % 32; // id within a warp
    int warpId = tid / 32; // warp id
    int row = blockIdx.x;

    // Since we have 2000 elements in the row and each block has only 1024 threads,
    // we find the local_max in grid loop
    // thread jumps 1024 each time to cover the whole 2000 elements
    float local_max = -1e20f;

    for (int i = tid; i < width; i += blockDim.x) {
        int idx = row * width + i; // global index
        local_max = fmaxf(local_max, d_in[idx]);
    }

    // Warp level reduction-- Local max (no shared memory yet)
    local_max = warpReduceMax(local_max);

    // lane 0 of each warp (local max) writes to the shared memory
    if (lane == 0) {
        warpMaxes[warpId] = local_max;
    }

    __syncthreads();

    // Final reduction-- since we narrowed down to 32 warpMaxes, use the warp 0 threads to do the reduction
    float row_max = (tid < 32) ? warpMaxes[lane] : -1e20f;
    if (warpId == 0) {
        row_max = warpReduceMax(row_max);
        // broadcast within warp 0 so that all 32 threads have it
        row_max = __shfl_sync(0xffffffff, row_max, 0);
    }

    // Broadcase the row_max to all the threads (shared memory)
    __shared__ float final_row_max;
    if (tid == 0) {
        final_row_max = row_max;
    }
    __syncthreads();
    row_max = final_row_max;

    // Sum the exp of x - max

    // local sum
    float local_sum = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) {
        int idx = row * width + i; // global index
        local_sum += expf(d_in[idx] - row_max);
    }

    // warp level reduce to get the sum
    local_sum = warpReduceSum(local_sum);

    // local sum of each warp (lane 0) writes to the shared memory
    if (lane == 0) {
        warpSums[warpId] = local_sum;
    }

    __syncthreads();

    // Final reduction: using the warp 0 threads to get the max
    float row_sum = (tid < 32) ? warpSums[lane] : 0.0f;
    if (warpId == 0) {
        row_sum = warpReduceSum(row_sum);
        // Broadcast to warp 0 so that all 32 threads have it
        row_sum = __shfl_sync(0xffffffff, row_sum, 0);
    }

    // Broadcast the row_sum to all threads (shared memory)
    __shared__ float final_row_sum;
    if (tid == 0) {
        final_row_sum = row_sum;
    }
    __syncthreads();
    row_sum = final_row_sum;

    // Divide and write back to the output
    for (int i = tid; i < width; i += blockDim.x) {
        int idx = row * width + i;
        if (i < width) {
            d_out[idx] = expf(d_in[idx] - row_max) / row_sum;
        }
    }
}

__host__ void softmax(float *h_in, float *h_out, int n) {
    float *d_in, *d_out;
    int sizeN = n * sizeof(float);

    CHECK_CUDA(cudaMalloc((void**)(&d_in), sizeN))
    CHECK_CUDA(cudaMalloc((void**)(&d_out), sizeN));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeN, cudaMemcpyHostToDevice));

    int ThreadsPerBlock = 1024; // use a power of 2 for tree reduction
    int BlocksPerGrid = (n + ThreadsPerBlock - 1)/ ThreadsPerBlock;
    size_t shared_mem_size = ThreadsPerBlock * sizeof(float);

    softmaxWarpShufflesKernel<<<BlocksPerGrid, ThreadsPerBlock, shared_mem_size>>>(d_in, d_out, n);

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeN, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

}

int main() {

    float *h_in, *h_out;
    int N = 2000;
    int sizeN = N * sizeof(float);

    h_in = (float*)malloc(sizeN);
    h_out = (float*)malloc(sizeN);

    for (int i = 0; i < N; i++) {
        h_in[i] = 900 + rand() % 21;
    }

    softmax(h_in, h_out, N);

    verify_softmax(h_in, h_out, N);

    free(h_in);
    free(h_out);

    return 0;
}
