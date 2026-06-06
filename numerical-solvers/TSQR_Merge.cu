
// After the panel_qr_baseline kernel processed level 0 (M x N of the raw A)
// this TSQR merge kernel handls level >= 1, assume input data is two stacked
// R factors from previous layer, forming a 2N x N matrix.
// It zeros out the bottom R factor using Householder transformations.
// It's like apply reduction with Housholder QR factorization operator.

#define BLOCK_DIM 32

// d_tau: size = tree_level * active_blocks_level0 * N * sizeof(float)
// where active_blocks_level0 * N is the max coefficients at level 0
// stride: distance in floats between the stacked R in global memory
// lda: leading dimension of global matrix A
// level: the current level of the tree reduction (used to offset tau writes)
__global__ void tsqr_merge_kernel(float* d_A, float* d_tau, float* d_V_merge, int M, int N, int stride, int lda, int level, int initial_blocks) {

    // shared memory for 2N x N stacked system (max N = 32 -> 64 rows)
    __shared__ float tile_A[64][BLOCK_DIM + 1];
    __shared__ float tile_v[64];
    __shared__ float s_tau;
    __shared__ float s_alpha;
    __shared__ float s_v_first;

    // index
    int tx = threadIdx.x;
    int M_merge = 2 * N; // stacking two N x N Rs to create 2N rows

    // global memory base pointers to this block's top and bottom pair
    // mapping the kernel blockIdx.x to original leaf Block ID
    // for example, blockIdx.x 0, top leaf Block 0, bottom leaf Block 1
    // blockIdx.x 1, top leaf Block 2, bottom leaf Block 3
    float* base_top = d_A + (2 * blockIdx.x * stride);
    float* base_bottom = base_top + stride;

    // load the two separated N x N in to shared tile 2N x N
    int total_elements = N * N;
    for (int index = tx; index < total_elements; index += blockDim.x) {
        int r = index / N;
        int c = index % N;

        // upper triangle of top
        tile_A[r][c] = (r <= c) ? base_top[r * lda + c] : 0.0f;

        // upper triangle of bottom
        tile_A[r + N][c] = (r <= c) ? base_bottom[r * lda + c] : 0.0f;
    }
    __syncthreads();

    // householder reflection for each column in N
    for (int k = 0; k < N; k++) {
        // compute norm, s_alpha, s_v_first, tau (thread 0)
        if (tx == 0) {
            float residual_squares = 0.0f;

            // sum_squares needs row k of top and all rows of bottom
            //sum_squares += tile_A[k][k] * tile_A[k][k]; // top diagonal
            for (int i = k + 1; i < N; i++) {
                residual_squares += tile_A[i][k] * tile_A[i][k];
            }
            for (int i = N; i < M_merge; i++) {
                residual_squares += tile_A[i][k] * tile_A[i][k]; // bottom elements
            }
            float ak = tile_A[k][k];
            float sum_squares = residual_squares + ak * ak;

            float norm = sqrtf(sum_squares);

            s_alpha = (ak > 0.0f) ? -norm : norm;
            s_v_first = ak - s_alpha;

            // scaled tau
            s_tau = (s_v_first == 0.0f) ? 0.0f : 2.0f * s_v_first * s_v_first / (s_v_first * s_v_first + residual_squares);
        }
        __syncthreads();

        // populate reflection vector (grid_stride)
        for (int i = tx; i < M_merge; i += blockDim.x) {
            if (i == k) {
                tile_v[i] = 1.0f; // normalized
            }
            // elements below diagonal in top or anywhere in bottom
            else if (i > k) {
                tile_v[i] = tile_A[i][k] / s_v_first;
            }
            else {
                tile_v[i] = 0.0f; // clear out upper triangle
            }
        }
        __syncthreads();

        // update trailing columns
        // A = A - tau * v * (v^T * A)
        for (int j = k + 1 + tx; j < N; j += blockDim.x) {
            // dot product needs to include top block row k
            float dot = tile_v[k] * tile_A[k][j];
            for (int i = N; i < M_merge; i++) {
                dot += tile_v[i] * tile_A[i][j];
            }

            // Rank-1 update top row and bottom rows
            tile_A[k][j] -= s_tau * tile_v[k] * dot;
            for (int i = N; i < M_merge; i++) {
                tile_A[i][j] -= s_tau * tile_v[i] * dot;
            }
        }
        __syncthreads();

        // save householder commponents below the diagonal
        for (int i = tx; i < M_merge; i += blockDim.x) {
            if (i > k) {
                tile_A[i][k] = tile_v[i];
            }
        }

        // save diagonal element and save tau to the global memory
        if (tx == 0) {
            tile_A[k][k] = s_alpha;

            // using level to prevent overwriting previous coefficients
            int global_tau_idx = (level * (initial_blocks * N)) + (blockIdx.x * N) + k;
            d_tau[global_tau_idx] = s_tau;
        }
        __syncthreads();
    }

    // write back to the global memory

    int merge_block_offset = (level * initial_blocks + blockIdx.x) * (N * N);

    for (int index = tx; index < total_elements; index += blockDim.x) {
        int r = index / N;
        int c = index % N;

        if (r <= c) { // only writes the upper triangle
            base_top[r * lda + c] = tile_A[r][c];
        }

        d_V_merge[merge_block_offset + r * N + c] =tile_A[r + N][c];
    }
}
