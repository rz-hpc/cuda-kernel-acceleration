
// Block -> Head -> Sequence (length N) -> Head Dim (vector for words, length d, width(number of columns) of the matrix)

// Calucate the global offset
int batch_id = blockIdx.x;
int head_id = blockIdx.y;

int head_offset = (batch_id * gridDim.y + head_id) * N * d;

float* q_ptr = d_Q + head_offset;
float* k_ptr = d_K + head_offset;
float* v_ptr = d_V + head_offset;

// The Tile Squad
#define Br 16 // height (how many rows) of the Q tile
#define Bc 16 // height (how many rows) of the KV tiles

int tid = threadIdx.x;
int threads_per_row = blockDim.x / Br;

int row_idx = tid / threads_per_row; // current thread's row index in the tile
int col_idx = tid % threads_per_row;

// Register (d / threads_per_row = 8 columns) for the result
float local_output[d / threads_per_row] = {0.0f};

// The Memory Tape
extern __shared__ float s_mem[];

float* s_q = s_mem; // start address of q in the shared memory
float* s_k = s_mem + (Br * d); // after shared q
float* s_v = s_mem + (Br * d) + (Bc * d); // after shared q and shared k

// load_idx for K tile

int num_blocks_Q = N / Br;
int num_blocks_KV =  N / Bc;

// outer loop Q
for (int i = 0; i < num_blocks_Q; i++) {

    // load q from global memory to shared memory tile
    float* q_tile_start = q_ptr + i * Br * d;

    for (int offset = 0; offset < d; offset += threads_per_row) {
        int cur_col = col_idx + offset;
        int load_idx = row_idx * d + cur_col;
        s_q[load_idx] = q_tile_start[load_idx];
    }

    __syncthreads();

    // inner loop KV
    for (int j = 0; j < num_blocks_KV; j++) {

        // Calculate the start pointer of This tile in the global memory
        float* k_tile_start = k_ptr + j * Bc * d; // jump to the start of jth tile
        float* v_tile_start = v_ptr + j * Bc * d;

        // Each thread find its specific float in this tile

        // Load from global memory to the shared memory
        // Tile-stride since threads might need to do multiple time of work to fill the whole row of width d (number of columns)
        // For example, d = 64 total columns, threads_per_row = 8, so each thread handles 64/8 =8 columns
        // With the offset, thread 0 handles column {0, 8, 16, 24, 32, 40, 48, 56}
        // thread 1 handles column {1, 9, 17, 25, 33, 41, 49, 57}
        // All the 128 threads doing the load
        // Interleaved: Coalesced single trip -- 10x faster
        for (int offset = 0; offset < d; offset += threads_per_row) {
            int cur_col = col_idx + offset;
            int load_idx = row_idx * d + cur_col;
            s_k[load_idx] = k_tile_start[load_idx];
            s_v[load_idx] = v_tile_start[load_idx];
        }

        // Wait for all threads in the block (1 block 1 head) to finish loading before doing match
        __syncthreads();


        // Compute dot product QK^T (for each row in k)
        for (int k_row = 0; k_row < Bc; k_row++) {
            float partial_score = 0.0f;

            // Tile-stride again -- each thread in the squad (8) handles 8 columns of d (64)
            for (int offset = 0; offset < d; offset += threads_per_row) {
                int cur_col = col_idx + offset;

                partial_score += s_q[row_idx * d + cur_col] * s_k[k_row * d + cur_col];
            }

            // The softmaxWarpShuffles, Warp reduction
            for (int mask = threads_per_row / 2; mask > 0; mask >>= 1) {
                partial_score = __shft_xor_sync(0xffffffff, partial_score, mask);
            }

            // Leader of squads comes here and calculates the softmax constants (scale and exp_new)
            // softmax_scare = 1 / sqrt(d), to keep the softmax stable
            //float softmax_scale = 1.0 / sqrtf((float)d);
            float scale = 1.0f;
            float exp_new = 0.0f;
            if (col_idx == 0) {
                float S_ij = partial_score * softmax_scale;

                float old_max = row_max;
                row_max = fmaxf(row_max, S_ij);
                scale = expf(old_max - row_max);
                exp_new = expf(S_ij - row_max);
                row_sum = row_sum * scale + exp_new;
            }

            // Leader broadcast the softmax constants 'scale' and 'exp_new' to the threads in the squad
            scale = __shfl_sync(0xffffffff, scale, 0, threads_per_row);
            exp_new = __shfl_sync(0xffffffff, exp_new, 0, threads_per_row);

            // Every thread in the squad update its columns (d / threads_per_row = 8) output
            for (int ir = 0; ir < d / threads_per_row; ir++) {
                // ir to access local register, cur_col to find corresponding spot in s_v
                int cur_col = col_idx + ir * threads_per_row; // interleaved column

                local_output[ir] = local_output[ir] * scale + s_v[k_row * d + cur_col] * exp_new;
            }
        }

        __syncthreads();

    }

    // Calculate the final Q and write to the output (at the end of the outer loop)
    for (int io = 0; io < d / threads_per_row; io++) {
        int final_col = col_idx + io * threads_per_row;

        float final_val = local_output[io] / row_sum;

        // Write back to global memory
        // i * Br: the start of the current Q tile (16 rows)
        // row_idx: the specific row in that [0, 15] Q tile rows
        int global_row = i * Br + row_idx;
        int global_idx = global_row * d + final_col;

        d_out[global_idx] = final_val;
    }

    // Reset for the next row in the Q tile
    row_max = -INFINITY;
    row_sum = 0.0f;
    for (int ir = 0; ir < d / threads_per_row; ir++) {
        local_out[ir] = 0.0f;
    }
}
