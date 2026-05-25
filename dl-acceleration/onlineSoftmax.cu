
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

void online_softmax_cpu(std::vector<float>& nums) {
    float val_max = -1e20f;
    float val_sum = 0.0f;

    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] > val_max) {
            val_sum = val_sum * expf(val_max - nums[i]) + expf(nums[i] - val_max);
            val_max = nums[i];
        }
        else {
            val_sum += expf(nums[i] - val_max);
        }
    }

    std::cout<< "Final sum: " << val_sum << " Final max: " << val_max << std::endl;
}

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

// Define a struct to hold both max and sum
// Max and Denominator
struct MD {
  // NVCC supports morden c++ 11/14/17
  //float m_Max = -1e20f;
  //float m_Sum = 0.0f;

  // To avoid overhead from hidden constructor, we don't initial with default value here
  // Later initial the accumulators as MD local_md = {-1e20f, 0.0f};

  float m_Max;
  float m_Sum;
};

__device__ MD mergeMD(MD a, MD b) {
    // The merged MD takes the larger m_Max from a and B
    // And the "winner" sum is kept, while the "loser" sum is scaled and added to the winner

    MD res;

    res.m_Max = fmaxf(a.m_Max, b.m_Max);

    // Rescale both to the winner max, the winner sum won't change while the loser sum will be scaled
    // since e^0 = 1
    res.m_Sum = a.m_Sum * expf(a.m_Max - res.m_Max)
              + b.m_Sum * expf(b.m_Max - res.m_Max);

    return res;
}

__device__ MD warpReduceMD(MD val) {
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        MD neighbor;
        neighbor.m_Max = __shfl_down_sync(0xffffffff, val.m_Max, offset);
        neighbor.m_Sum = __shfl_down_sync(0xffffffff, val.m_Sum, offset);
        val = mergeMD(neighbor, val);
    }
    return val;
}

__global__ void flashSoftmaxKernel(float *d_in, float *d_out, int width) {
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int lane = tid % 32;
    int row = blockIdx.x;

    // Shared memory for the warp
    __shared__ MD warpResults[32];
    __shared__ float final_max;
    __shared__ float final_sum;

    // One pass: grid-stride loop
    // Get the local MD
    MD local_md = {-1e20f, 0.0f};
    for (int i = tid; i < width; i+= blockDim.x) {
        float x = d_in[row * width + i];
        MD cur = {x, 1.0f};// e^(x-x) = 1
        local_md = mergeMD(local_md, cur);
    }

    // Reduction
    // The Hierarchy: Reducing from Warp $\rightarrow$ Shared $\rightarrow$ Block $\rightarrow$ Broadcast.

    // Reduce within warp
    local_md = warpReduceMD(local_md);
    if (lane == 0) {
        warpResults[warpId] = local_md;
    }
    __syncthreads();

    // Reduce within block using the first warp 0
    MD final_md = (tid < 32) ? warpResults[lane] : (MD){-1e20f, 0.0f};
    if (warpId == 0) {
        final_md = warpReduceMD(final_md);
        final_md.m_Max = __shfl_sync(0xffffffff, final_md.m_Max, 0);
        final_md.m_Sum = __shfl_sync(0xffffffff, final_md.m_Sum, 0);
    }
    __syncthreads();

    // Move from warp 0 to the rest of the block
    if (tid == 0) {
        final_max = final_md.m_Max;
        final_sum = final_md.m_Sum;
    }

    __syncthreads();

    // Write to the output
    // Need to do the grid-stride loop again
    for (int i = tid; i < width; i += blockDim.x) {
        int idx = row * width + i;
        d_out[idx] = expf( d_in[idx] - final_max) / final_sum;
    }

}

__host__ void softmax(float *h_in, float *h_out, int n) {
    float *d_in, *d_out;
    int sizeN = n * sizeof(float);

    CHECK_CUDA(cudaMalloc((void**)(&d_in), sizeN))
    CHECK_CUDA(cudaMalloc((void**)(&d_out), sizeN));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeN, cudaMemcpyHostToDevice));

    //int ThreadsPerBlock = 1024; // power of 2
    //int BlocksPerGrid = ( n + ThreadsPerBlock - 1) / ThreadsPerBlock;
    //flashSoftmaxKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(d_in, d_out, n);

    // One block handles One row
    int num_rows = 1;
    int N_per_row = 2000;
    // Total size = num_rows * N_per_row
    flashSoftmaxKernel<<<num_rows, 1024>>>(d_in, d_out, N_per_row);

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
