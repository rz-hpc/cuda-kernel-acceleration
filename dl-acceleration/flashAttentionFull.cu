
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define Br 16
#define Bc 16

void verify_attention(float *Q, float *K, float *V, float *O, int N, int d) {
    float softmax_scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < N; i++) {
        std::vector<float> scores(N);
        float r_max = -INFINITY;
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) sum += Q[i * d + k] * K[j * d + k];
            scores[j] = sum * softmax_scale;
            r_max = std::max(r_max, scores[j]);
        }
        float r_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            scores[j] = expf(scores[j] - r_max);
            r_sum += scores[j];
        }
        for (int col = 0; col < d; col++) {
            float out_val = 0.0f;
            for (int j = 0; j < N; j++) out_val += (scores[j] / r_sum) * V[j * d + col];
            O[i * d + col] = out_val;
        }
    }
}

__global__ void flashAttentionKernel(float *d_Q, float *d_K, float *d_V, float *d_O, int N, int d, float softmax_scale) {
    int head_offset = (blockIdx.x * gridDim.y + blockIdx.y) * N * d;

    // Use local pointers
    float* q_ptr = d_Q + head_offset;
    float* k_ptr = d_K + head_offset;
    float* v_ptr = d_V + head_offset;
    float* o_ptr = d_O + head_offset;

    extern __shared__ float s_mem[];
    float* s_q = s_mem;
    float* s_k = s_mem + (Br * d);
    float* s_v = s_mem + (Br * d) + (Bc * d);
    // Additional small shared space for row stats (16 rows)
    float* s_max = s_v + (Bc * d);
    float* s_sum = s_max + Br;

    int tid = threadIdx.x;
    int row_idx = tid / (d / 8); // 128 threads / (64/8) = 16 rows
    int col_start = (tid % (d / 8)) * 8;

    if (tid < Br) {
        s_max[tid] = -INFINITY;
        s_sum[tid] = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i < N / Br; i++) {
        // Load Q
        for (int k = 0; k < 8; k++)
            s_q[row_idx * d + col_start + k] = q_ptr[(i * Br + row_idx) * d + col_start + k];

        float local_out[8] = {0.0f};
        float l_max = -INFINITY;
        float l_sum = 0.0f;

        __syncthreads();

        for (int j = 0; j < N / Bc; j++) {
            // Load K and V
            for (int k = 0; k < 8; k++) {
                s_k[row_idx * d + col_start + k] = k_ptr[(j * Bc + row_idx) * d + col_start + k];
                s_v[row_idx * d + col_start + k] = v_ptr[(j * Bc + row_idx) * d + col_start + k];
            }
            __syncthreads();

            // Math
            for (int m = 0; m < Bc; m++) {
                float score = 0.0f;
                for (int k = 0; k < d; k++) score += s_q[row_idx * d + k] * s_k[m * d + k];
                score *= softmax_scale;

                float old_max = l_max;
                l_max = fmaxf(l_max, score);
                float exp_scale = expf(old_max - l_max);
                float exp_score = expf(score - l_max);

                l_sum = l_sum * exp_scale + exp_score;
                for (int k = 0; k < 8; k++)
                    local_out[k] = local_out[k] * exp_scale + s_v[m * d + col_start + k] * exp_score;
            }
            __syncthreads();
        }

        // Final Write
        for (int k = 0; k < 8; k++)
            o_ptr[(i * Br + row_idx) * d + col_start + k] = local_out[k] / l_sum;
    }
}

__host__ float launchFlashAttention(float* d_Q, float* d_K, float* d_V, float* d_O, int B, int H, int N, int d) {
  float softmax_scale = 1.0f / sqrtf((float)d);
  dim3 grid(B, H);
  dim3 block(128);
  int smem = (Br * d + Bc * d + Bc * d + Br + Br) * sizeof(float);

  // Timer for Benchmark
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0.0f;

  cudaEventRecord(start);

  flashAttentionKernel<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N, d, softmax_scale);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  return milliseconds;

}

int main() {
    int N = 1024, d = 64, B = 4, H = 12; // Smaller for quick verification
    size_t sz = (size_t)B * H * N * d * sizeof(float);
    float *h_Q = (float*)malloc(sz), *h_K = (float*)malloc(sz), *h_V = (float*)malloc(sz), *h_O = (float*)malloc(sz);
    for(size_t i=0; i<B*H*N*d; i++){ h_Q[i] = (float)rand()/RAND_MAX; h_K[i] = (float)rand()/RAND_MAX; h_V[i] = (float)rand()/RAND_MAX; }
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, sz); cudaMalloc(&d_K, sz); cudaMalloc(&d_V, sz); cudaMalloc(&d_O, sz);
    cudaMemcpy(d_Q, h_Q, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sz, cudaMemcpyHostToDevice);

    // Run and time the kernel
    float ms = launchFlashAttention(d_Q, d_K, d_V, d_O, B, H, N, d);
    cudaMemcpy(h_O, h_O, sz, cudaMemcpyDeviceToHost); // Verify pointer name h_O
    cudaMemcpy(h_O, d_O, sz, cudaMemcpyDeviceToHost);

    // Calculate TFLOPS (Total Floating Point Operations / Time)
    // Flash Attention does roughly 4 * B * H * N^2 * d operations
    double flops = 4.0 * B * H * N * N * d;
    double tflops = (flops / (ms / 1000.0)) / 1e12;

    std::cout << "GPU Time: " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

    float* cpu_O = (float*)malloc(sz);
    verify_attention(h_Q, h_K, h_V, cpu_O, N, d);

    float err = 0.0f;
    for (int i = 0; i < N * d; i++) err = std::max(err, std::abs(h_O[i] - cpu_O[i]));
    std::cout << "Max error: " << err << (err < 1e-3 ? " SUCCESS!" : " FAILURE!") << std::endl;
    return 0;
}
