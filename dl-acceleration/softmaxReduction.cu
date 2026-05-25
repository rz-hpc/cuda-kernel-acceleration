
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

__global__ void softmaxReductionKernel(float* d_in, float* d_out, int width) {
    // Dynamic allocate memory for shared memory for unknown size
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int idx = row * width + tid;

    // Each thread load input to the shared memory
    float val = tid < width ? d_in[idx] : -1e20f;
    sdata[tid] = val;

    __syncthreads();

    // Reduction: Find the max value (tree)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {//which means tid + s < blockDim.x
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }

        __syncthreads();
    }

    // Thread 0 holds the max for the row
    float max_val = sdata[0];

    __syncthreads();

    // Calculate the e^(xi - max(x)) and store in the shared memory
    float exp_val = expf(val - max_val);
    sdata[tid] = exp_val;

    __syncthreads();

    // Reduction: Find the sum (tree)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // Thread 0 now has the sum value for the row
    float sum_val = sdata[0];
    __syncthreads();

    // Divide and write back to output
    if (tid < width) {
        d_out[idx] = exp_val / sum_val;
    }

}

__host__ void reduction(float *h_in, float *h_out, int n) {
    float *d_in, *d_out;
    int sizeN = n * sizeof(float);

    CHECK_CUDA(cudaMalloc((void**)(&d_in), sizeN))
    CHECK_CUDA(cudaMalloc((void**)(&d_out), sizeN));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeN, cudaMemcpyHostToDevice));

    int ThreadsPerBlock = 1024; // use a power of 2 for tree reduction
    int BlocksPerGrid = (n + ThreadsPerBlock - 1)/ ThreadsPerBlock;
    size_t shared_mem_size = ThreadsPerBlock * sizeof(float);

    softmaxReductionKernel<<<BlocksPerGrid, ThreadsPerBlock, shared_mem_size>>>(d_in, d_out, n);

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeN, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

}

int main() {

    float *h_in, *h_out;
    int N = 1024;
    int sizeN = N * sizeof(float);

    h_in = (float*)malloc(sizeN);
    h_out = (float*)malloc(sizeN);

    for (int i = 0; i < N; i++) {
        h_in[i] = 900 + rand() % 21;
    }

    reduction(h_in, h_out, N);

    verify_softmax(h_in, h_out, N);

    free(h_in);
    free(h_out);

    return 0;
}
