
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

// Flash Attention:
// O = softmax(QK^t)V
// Q, K, V, O dimension (N, d), QK^t dimension (N, N)

// Helper function
// D: head dimension, the width of the matrix, usually 64, 128, 256
// B: block size/tile height
// Br: block rows-- the height of the Q tile usually 16/32 rows
// Bc: block columns-- the height of the K and V tiles 16 or 32 rows
template <int B, int D>
__device__ void load_tile_to_shared(float *src, float *dst, int row_offset, int tid, int num_threads) {
    // How many float are in a row?
    const int floats_per_thread = (B * D) / num_threads;

    // Convert pointers to float4 pointers for vectorized access
    float4* src4 = reinterpret_cast<float4*>(src + row_offset + D);
    float4* dst4 = reinterpret_cast<float4*>(dst);

    #pragma unroll
    for (int i = 0; i < floats_per_thread / 4; i++) {
        // each thread jumps by num_threads to keep memory access coalesced
        int load_idx = tid + i * num_threads;
        dst4[load_idx] = src4[load_idx];
    }

}

#define Br 16   // Q-Tile height
#define Bc 16   // KV-Tile height

__global__ void flashAttentionKernel(float *d_Q, float *d_K, float *d_V, float *d_O, int N, int d, float softmax_scale) {
    // 1. Thread Identifiers
    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;

    // 2. Global Memory Offsets
    int offset = (batch_id * gridDim.y + head_id) * N * d;
    float* q_head = d_Q + offset;
    float* k_head = d_K + offset;
    float* v_head = d_V + offset;
    float* o_head = d_O + offset;

    // 3. Shared Memory (The "Tape")
    extern __shared__ float s_mem[];
    float* s_Q = s_mem;                               // Size: Br * d
    float* s_K = s_mem + (Br * d);                    // Size: Bc * d
    float* s_V = s_mem + (Br * d) + (Bc * d);         // Size: Bc * d

    // 4. Thread-Local Work Mapping (128 threads / 16 rows = 8 threads per row)
    const int threads_per_row = blockDim.x / Br;
    int local_q_row = tid / threads_per_row;   // Which row of the tile (0-15)
    int row_lane    = tid % threads_per_row;   // Which of the 8 threads am I? (0-7)

    // 5. Accumulators (One set per thread, but only row_lane 0 will be the "Master")
    float row_max = -1e20f;
    float row_sum = 0.0f;
    // We need d accumulators if we want to store the whole output row!
    // For simplicity, let's assume d=64 and this thread handles its chunk
    float local_out[64] = {0.0f};

    // --- OUTER LOOP (Move down the Q rows) ---
    for (int i = 0; i < N / Br; i++) {

        // Load Q-Tile (Once per outer loop)
        load_tile_to_shared<Br, 64>(q_head, s_Q, i * Br, tid, blockDim.x);
        __syncthreads();

        // --- INNER LOOP (Scan across all K, V) ---
        for (int j = 0; j < N / Bc; j++) {

            // Load K and V Tiles
            load_tile_to_shared<Bc, 64>(k_head, s_K, j * Bc, tid, blockDim.x);
            load_tile_to_shared<Bc, 64>(v_head, s_V, j * Bc, tid, blockDim.x);
            __syncthreads();

            // --- THE COMPUTE BLOCK ---
            for (int k_row = 0; k_row < Bc; k_row++) {
                // A. Dot Product (Q * K^T)
                float score = 0.0f;
                for (int d_idx = row_lane; d_idx < d; d_idx += threads_per_row) {
                    score += s_Q[local_q_row * d + d_idx] * s_K[k_row * d + d_idx];
                }

                // B. Warp Shuffle to sum the 8 partial scores
                for (int off = threads_per_row / 2; off > 0; off >>= 1) {
                    score += __shfl_xor_sync(0xffffffff, score, off);
                }

                // C. Master thread of the row (row_lane 0) updates the Softmax stats
                // Note: In real FlashAttention, we'd use registers for speed,
                // but let's keep it simple: row_lane 0 does the math.
                if (row_lane == 0) {
                    float S_ij = score * softmax_scale;
                    float old_max = row_max;
                    row_max = fmaxf(old_max, S_ij);

                    float exp_new = expf(S_ij - row_max);
                    float scale   = expf(old_max - row_max);

                    row_sum = (row_sum * scale) + exp_new;

                    // D. Update Output row (this is the hardest part to do in registers)
                    // Every word in s_V for this k_row needs to be scaled and added
                    for (int d_v = 0; d_v < d; d_v++) {
                        local_out[d_v] = (local_out[d_v] * scale) + (exp_new * s_V[k_row * d + d_v]);
                    }
                }
            }
            __syncthreads(); // Sync before loading next K,V tile
        }

        // --- FINAL WRITE BACK ---
        // row_lane 0 writes the normalized row back to global O
        if (row_lane == 0) {
            for (int d_out = 0; d_out < d; d_out++) {
                o_head[(i * Br + local_q_row) * d + d_out] = local_out[d_out] / row_sum;
            }
        }
    }
}


