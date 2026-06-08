
#define BLOCK_DIM 32


// d_A: M x N
// d_tau: Pointer to an array of size $N$ where scalar reflection coefficients are saved
// lda: Leading Dimension of A. The stride (in elements) between consecutive rows
// this panel qr kernel works for M <= 512 N <= 32
__global__ void panel_qr_baseline(float* d_A, float* d_tau, int M, int N, int lda) {
    // For starting point, assume a tall-skinny chunk where M fits in a fixed max size (e.g., 512)
    __shared__ float tile_A[512][BLOCK_DIM + 1];
    __shared__ float tile_v[512];
    __shared__ float s_tau;
    __shared__ float s_alpha; // the first element of target vector
    __shared__ float s_v_first;

    int tx = threadIdx.x;
    int total_element = M * N;

    // load panel from global memory to shared memory
    // grid-stride over flat panel size
    for (int index = tx; index < total_element; index += blockDim.x) {
        // mapping flat 1D index back to 2D coordinates
        int i = index / N;
        int j = index % N;
        tile_A[i][j] = d_A[i * lda + j];
    }
    __syncthreads();

    // column by column
    for (int k = 0; k < N; k++) {
        // thread 0 to compute norm and tau of column k
        if (tx == 0) {
            float sum_squares = 0.0f;
            for (int i = k; i < M; i++) {
                sum_squares += tile_A[i][k] * tile_A[i][k];
            }
            float norm = sqrtf(sum_squares);

            // target verctor: first element s_alpha
            // opposite sign of the first diagonal element of current (sub)A to avoid catastrophic cancellation
            // same abs value of norm
            float ak = tile_A[k][k];
            s_alpha = (ak > 0.0f) ? -norm : norm;

            s_v_first = ak - s_alpha;
            // Reconstruct v^T * v = (v_first^2) + sum of squares of remaining elements
            float residual_sq = sum_squares - ak * ak;

            // Calulate tau: tau = 2.0/||v_scaled||^2
            // v_scaled = v / v_first, ||v_scaled||^2 = v^2 / v_first^2
            // since v^2 = v^T v = v_first ^ 2 + residual_sq
            // tau = 2.0 * v_first ^ 2 / (v_first ^ 2 + residual_sq)
            float residual_sq = sum_squares - ak * ak;
            s_tau = 2.0f * s_v_first * s_v_first / (s_v_first * s_v_first + residual_sq);
        }
        __syncthreads();

        // Populate the reflection vector V
        for (int i = tx; i < M; i += blockDim.x) {
            if (i == k) {
                tile_v[i] = 1.0f; // normalized v1
            }
            else if (i > k) {
                tile_v[i] = tile_A[i][k] / s_v_first;
                tile_A[i][k] = tile_v[i];
            }
            else {
                tile_v[i] = 0.0f; // upper triangle (above diagonal)
            }
        }

        // thread 0 saves R's diagonal value
        if (tx == 0) {
            tile_A[k][k] = s_alpha;
        }
        __syncthreads();

        // update trailing columns (columns to the right of k)
        // H_1 A = A - tau v_1 (v_1^T A)
        // each thread handles a column j
        // j: coalesced work distribution or a grid-stride loop
        for (int j = k + 1 + tx; j < N; j += blockDim.x) {
            // apply dot product v^T * A_j
            float dot = 0.0f;
            for (int i = k; i < M; i++) {
                dot += tile_v[i] * tile_A[i][j];
            }

            // apply rank-1 update: A_j = A_j - tau * v * dot
            for (int i = k; i < M; i++) {
                tile_A[i][j] -= s_tau * tile_v[i] * dot;
            }
        }
        __syncthreads();

        // save tau scalar of this column to the global memory
        if (tx == 0) {
            d_tau[k] = s_tau;
        }

    }

    // write back the completed upper triangular R and householder vectors to global memory
    // grid-stride over flat panel size
    for (int index = tx; index < total_element; index += blockDim.x) {
        // mapping flat 1D index back to 2D coordinates
        int i = index / N;
        int j = index % N;
        d_A[i * lda + j] = tile_A[i][j];
    }

}

// Verified panel_qr_baseline kernel
__global__ void panel_qr_baseline(float* d_A, float* d_tau, int M, int N, int lda) {

	// shared memory
	__shared__ float tile_A[128][BLOCK_DIM + 1];
	__shared__ float tile_v[128];
	__shared__ float s_tau;
	__shared__ float s_alpha; // target vector first element
	__shared__ float s_v_first; // reflection vector first element

	// index
	int tx = threadIdx.x;
	int total_elements = M * N; // for coalesced memory access

  // offset global memory pointer by block index
  float* block_A = d_A + (blockIdx.x * M * lda);

	// load from global memory to shared memory
	for (int index = tx; index < total_elements; index+= blockDim.x) {
		int r = index / N;
		int c = index % N;
		tile_A[r][c] = block_A[ r * lda + c];
	}
	__syncthreads();

	// for each column k in N
	for (int k = 0; k < N; k++) {
		// calculate (thread 0) sum squares, norm, first element of target vector, reflect vector first element and tau
		if (tx == 0) {
			float residual_squares = 0.0f;
			for (int i = k + 1; i < M; i++) {
				residual_squares += tile_A[i][k] * tile_A[i][k];
			}
      float ak = tile_A[k][k];
      float sum_squares = residual_squares + ak * ak;
			float norm = sqrtf(sum_squares);
			s_alpha = (ak > 0.0f) ? -norm : norm;
			s_v_first = ak - s_alpha;

			// tau_scaled = 2.0f / v_scaled^2
			// v_scaled ^ 2 = v^2 / s_v_first^2
			// v^2 = s_v_first ^ 2 + residual_squares
			// tau_scaled = 2.0f * s_v_first^2 / (s_v_first ^ 2 + residual_squares)
			s_tau = 2.0f * s_v_first * s_v_first / (s_v_first * s_v_first + residual_squares);
		}
		__syncthreads();

		// populate the reflection vector v in shared memory tile and save to shared memory A
		for (int i = tx; i < M; i += blockDim.x) {
			if (i == k) {
				tile_v[i] = 1.0f; // normalized
			}
			else if (i > k) {
				tile_v[i] = tile_A[i][k] / s_v_first;
				tile_A[i][k] = tile_v[i];
			}
			else {
				tile_v[i] = 0.0f; // clear out upper triangle
			}
		}
		__syncthreads();

		// save diagonal for R with the target vector first element
		if (tx == 0) {
			tile_A[k][k] = s_alpha;
		}
		__syncthreads();

		// update trailing columns (the right of k)
		// H1 = I - tau * v1 * v1^T
		// H1 * A = A - tau * v1 * (v1^T * A)
		for (int j = k + 1 + tx; j < N; j += blockDim.x) {
			float dot = 0.0f;
			for (int i = k; i < M; i++) {
				dot += tile_v[i] * tile_A[i][j];
			}

			// rank 1 update A_j = A_j - tau * v * dot
			for (int i = k; i < M; i++) {
				tile_A[i][j] -= s_tau * tile_v[i] * dot;
			}
		}
		__syncthreads();

		// save tau to global memory
		if (tx == 0) {
			d_tau[blockIdx.x * N + k] = s_tau;
		}
	}

	// write back to the global memory
	for (int index = tx; index < total_elements; index+= blockDim.x) {
		int r = index / N;
		int c = index % N;
    block_A[r * lda + c] = tile_A[r][c];
	}
}

// Since the Nsight report pointed out the UncoalescedSharedAccess,
// the optimization is to when calculate dot product replace shared memory tile_v with within warp __shfl_sync,
// to 1. resolve bank conflicts 2. improve execution throughput
// Given the fact that it's a Tall Skinny matrix (N = 4 for example),
// it fits in a warp totally fine.

#define TILE_SIZE 128

// here M is the block rows (not the full rows of origina A)
__global__ void qr_base_optimized_kernel(float* d_A, float* d_tau, int M, int N, int lda) {
    // shared memory
    __shared__ float tile_A[TILE_SIZE][BLOCK_DIM + 1];
    __shared__ float tile_v[TILE_SIZE];
    __shared__ float s_tau;
    __shared__ float s_alpha;
    __shared__ float s_v_first;

    // index
    int tx = threadIdx.x;
    float* block_A = d_A + (blockIdx.x * M * lda);

    for (int row_block = 0; row_block < M; row_block += TILE_SIZE) {

        // load from global memory to shared memory
        for (int i = tx; i < TILE_SIZE; i += blockDim.x) {
            for (int j = 0; j < N; j++) {
                tile_A[i][j] = block_A[(row_block + i) * lda + j];
            }
        }
        __syncthreads();

        // for each column k in N
        for (int k = 0; k < N; k++) {
            // calculate residual_sq, sum_squares, norm, s_alpha, s_v_first, and tau
            if (tx == 0) {
                float residual_sq = 0.0f;
                for (int i = k + 1; i < M; i++) {
                    residual_sq += tile_A[i][k] * tile_A[i][k];
                }
                float ak = tile_A[k][k];
                float sum_squares = residual_sq + ak * ak;
                float norm = sqrtf(sum_squares);
                s_alpha = (ak > 0.0f) ? -norm : norm;
                s_v_first = ak - s_alpha;

                // tau = 2.0f/(v_scaled^T * v_scaled) = 2 / || v_saled||^2
                // v_scaled = v / s_v_first
                // tau = 2.0f * s_v_first ^ 2 / ||v||^2
                s_tau = (s_v_first == 0.0f) ? 0.0f : 2.0f * s_v_first * s_v_first / (s_v_first * s_v_first + residual_sq);
            }
            __syncthreads();

            // populate reflection/householder vector v
            for (int i = tx; i < M; i += blockDim.x) {
                if (i == k) {
                    tile_v[i] = 1.0f; // normalized
                }
                else if (i > k) {
                    tile_v[i] = tile_A[i][k] / s_v_first;
                    tile_A[i][k] = tile_v[i];
                }
                else {
                    tile_v[i] = 0.0f; // clear out upper triangle
                }
            }
            __syncthreads();

            // save the R (diagonal)
            if (tx == 0) {
                tile_A[k][k] = s_alpha;
            }
            __syncthreads();

            // trailing columns update (to the right of k)
            // A = A - tau * v * (v^T * A)
            // Before optimization loop: for (int j = k + 1 + tx; j < N; j += blockDim.x)
            // One thread works on one column: thread 0 works on column k + 1, thread 1 works on column k + 2
            // it won't be helpful to do the warp reduction since there is only one thread
            // For the optimization of warp shuffling, need to change the loop to below
            // so that all 32 threads in a warp work together on one column
            // inner loop i needs to associate with tx, and grid-stride
            for (int j = k + 1; j < N; j++) {
                // dot product using registers and warp shuffle
                float dot = 0.0f;
                // each thread handles a subset of this column
                for (int i = k + tx; i < M; i += blockDim.x) {
                    dot += tile_v[i] * tile_A[i][j];
                }

                // Warp-level reduction: aggregate cross the 32 threads
                for (int offset = 16; offset >= 1; offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffff, dot, offset);
                }

                // Broadcast the result: thread 0 holds the result
                dot = __shfl_sync(0xffffffff, dot, 0);

                // rank-1 update
                for (int i = k + tx; i < M; i += blockDim.x) {
                    tile_A[i][j] -= s_tau * tile_v[i] * dot;
                }
            }
            __syncthreads();

            // save tau to the global memory
            if (tx == 0) {
                d_tau[blockIdx.x * N + k] = s_tau;
            }
        }

        // write back to the global memory
        for (int i = tx; i < TILE_SIZE; i += blockDim.x) {
            for (int j = 0; j < N; j++) {
                block_A[(row_block + i) * lda + j] = tile_A[i][j];
            }
        }
        __syncthreads();
    }
}
