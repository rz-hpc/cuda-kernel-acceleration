
// Reconstructor Order for Q in TSQR
// Q = (levle 0 Transformations) * (Level 1 Transformations) * (Level 3 Transformations) * I

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

// Assume BLOCK_DIM is defined as 32
#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif

// Panel QR Baseline kernel
// A = QR, Q is orthogonal matrix, R is upper triangular
// A: M x N
// Q: M x M
// R: M x N
// the result is updated in-place in A (packed A), that upper triangular is R (include diagonal), lower is reflection vector v for each column

// d_A: input dense matrix of M x N
// d_tau: reflection coefficients vector of size N
// lda: leading dimension of A (for global index in A)
__global__ void panel_qr_baseline_kernel(float* d_A, float* d_tau, int M, int N, int lda) {
	
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

// TSQR merge kernel
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

// Helper function to calculate Frobenius norm
float frobenius_norm(const std::vector<float>& mat, int rows, int cols) {
    float sum = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += mat[i * cols + j] * mat[i * cols + j];
        }
    }
    return std::sqrt(sum);
}

int main() {
    // matrix dimension ( 4 blocks of 128 rows = 512 rows, 4 columns)
    const int M = 512;
    const int N = 4;
    const int BLOCK_ROWS = 128;
    const int lda = N;

    int initial_blocks = M / BLOCK_ROWS; // 4 blocks
    // Level 0 (leaf), level 1 (merge 4 -> 2), level 2 (merge 2-> 1)
    int total_levels = 3;

    size_t matrix_size = M * N * sizeof(float);
    size_t tau_size = total_levels * initial_blocks * N * sizeof(float);
    size_t V_merge_size = total_levels * initial_blocks * N * N * sizeof(float);

    // Initialize host Matrix A
    std::vector<float> h_A(M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Generates a well-conditioned matrix
            h_A[i * N + j] = std::sin(i + 1) * (j + 1) + ((i == j) ? 10.0f : 0.0f);
        }
    }
    std::vector<float> h_A_orig = h_A;

    // allocate device memory
    float *d_A, *d_tau, *d_V_merge;
    cudaMalloc(&d_A, matrix_size);
    cudaMalloc(&d_tau, tau_size);
    cudaMalloc(&d_V_merge, V_merge_size);

    cudaMemset(d_tau, 0, tau_size);
    cudaMemset(d_V_merge, 0, V_merge_size);

    // copy to device
    cudaMemcpy(d_A, h_A.data(), matrix_size, cudaMemcpyHostToDevice);

    // TSQR pipeline vis host drive loop
    std::cout << "Launching level 0 TSQR" << std::endl;

    panel_qr_baseline_kernel<<<initial_blocks, BLOCK_DIM>>>(d_A, d_tau, BLOCK_ROWS, N, lda);
    cudaDeviceSynchronize();

    int active_blocks = initial_blocks;
    int level = 1;

    while (active_blocks > 1) {
        int next_level_blocks = active_blocks / 2;
        int stride = (M / active_blocks) * lda;

        std::cout << "Launch TSQR level " << level << ": Merge " 
        << active_blocks << " blocks -> " << next_level_blocks << " blocks...\n";

        tsqr_merge_kernel<<<next_level_blocks, BLOCK_DIM>>>(d_A, d_tau, d_V_merge, M, N, stride, lda, level, initial_blocks);
        cudaDeviceSynchronize();

        active_blocks = next_level_blocks;
        level++;
    }

    // copy result back to host for verification
    std::vector<float> h_A_factored(M * N);
    std::vector<float> h_tau(total_levels * initial_blocks * N);
    std::vector<float> h_V_merge(total_levels * initial_blocks * N * N);
    cudaMemcpy(h_A_factored.data(), d_A, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tau.data(), d_tau, tau_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_merge.data(), d_V_merge, V_merge_size, cudaMemcpyDeviceToHost);

    // Backward Pass: reconstruct Q on the host
    std::vector<float> Q(M * M, 0.0f);
    for (int i = 0; i < M; i++) {
        Q[i * M + i] = 1.0f; // initial Q as identity matrix
    }

    // Track active blocks per level for reconstruction
    // Level 2: 1 block, Level 1: 2 blocks, Level 0: 4 blocks
    std::vector<int> blocks_at_level = {4, 2, 1};

    // reverse loop, level 2 down to level 0
    for (int lvl = total_levels - 1; lvl >= 0; lvl--) {
        int num_blocks = blocks_at_level[lvl];

        if (lvl > 0) {
            // stride rows: rows in child R
            int current_block_rows = M / num_blocks; 
            int sub_stride = current_block_rows / 2;

            for (int b = 0; b < num_blocks; b++) {
                int base_top_row = 2 * b * sub_stride;
                int base_bottom_row = base_top_row + sub_stride;
                int merge_block_offset = (lvl * initial_blocks + b) * (N * N);

                for (int k = N - 1; k >= 0; k--) {
                    // extract tau
                    int tau_idx = (lvl * (initial_blocks * N)) + b * N + k;
                    float t = h_tau[tau_idx];
                    if (t == 0.0f) continue;

                    // extract implicity reflect vector v from factorized matrix rows
                    std::vector<float> v(2 * N, 0.0f);
                    v[k] = 1.0f; // diagonal element normalized to 1.0f

                    // Top half of the merge verctor is implicity 0.0f
                    // extract the bottom half directly from the workspace
                    for (int i = 0; i < N; i++) {
                        v[N + i] = h_V_merge[merge_block_offset + i * N + k];
                    }

                    // Left multiply Q: Q = (I - tau * v * v^T) * Q
                    std::vector<float> vTQ(M, 0.0f);
                    // compute v^T * Q across active rows
                    for (int j = 0; j < M; j++) {
                        for (int i = k; i < N; i++) {
                            vTQ[j] += v[i] * Q[(base_top_row + i) * M + j];
                        }
                        for (int i = 0; i < N; i++) {
                            vTQ[j] += v[N + i] * Q[(base_bottom_row + i) * M + j];
                        }
                    }
                    // Rank-1 update to Q rows
                    for (int j = 0; j < M; j++) {
                        for (int i = k; i < N; i++) {
                            Q[(base_top_row + i) * M + j] -= t * v[i] * vTQ[j];
                        }
                        for (int i = 0; i < N; i++) {
                            Q[(base_bottom_row + i) * M + j] -= t * v[N + i] * vTQ[j];
                        }
                    }
                }
            }
        }
        else {
            // levle 0: leaf level reconstruction
            for (int b = 0; b < num_blocks; b++) {
                int base_rows = b * BLOCK_ROWS;

                for (int k = N - 1; k >= 0; k--) {
                    int tau_idx = (lvl* (initial_blocks * N) + (b * N) + k);
                    float t = h_tau[tau_idx];
                    if (t == 0.0f) continue;

                    std::vector<float> v(BLOCK_ROWS, 0.0f);
                    v[k] = 1.0f;
                    for (int i = k + 1; i < BLOCK_ROWS; i++) {
                        v[i] = h_A_factored[(base_rows + i) * lda + k];
                    }

                    std::vector<float> vTQ(M, 0.0f);
                    for (int j = 0; j < M; j++) {
                        for (int i = k; i < BLOCK_ROWS; i++) {
                            vTQ[j] += v[i] * Q[(base_rows + i) * M + j];
                        }
                    }
                    for (int i = k; i < BLOCK_ROWS; i++) {
                        for (int j = 0; j < M; j++) {
                            Q[(base_rows + i) * M + j] -= t * v[i] * vTQ[j];
                        }
                    }

                }
            }


        }
    }

    // extract upper triangle matrix (R)
    std::vector<float> R(M * N, 0.0f);
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            R[i * N + j] = h_A_factored[i * lda + j];
        }
    }

    // Compute verification matrix QR
    std::vector<float> QR(M * N, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                QR[i * N + j] += Q[i * M + k] * R[k * N + j];
            }
        }
    }

    // residual || A - QR||
    std::vector<float> diff_A(M * N);
    for (size_t i = 0; i < h_A_orig.size(); i++) {
        diff_A[i] = h_A_orig[i] - QR[i];
    }
    float residual = frobenius_norm(diff_A, M, N) / frobenius_norm(h_A_orig, M, N);

    // Orthogonality check for Q: || Q^T * Q - I ||
    std::vector<float> QtQ(M * M, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < M; k++) {
                QtQ[i * M + j] += Q[k * M + i] * Q[k * M + j]; 
            }
        }
    }
    // subtract identity matric
    for (int i = 0; i < M; i++) {
        QtQ[i * M + i] -= 1.0f;
    }
    float orthogonality = frobenius_norm(QtQ, M, M);

    // validation results
    std::cout << "\n===TSQR VERIFICATION RESULTS===\n";
    std::cout << "Residual || A - QR ||_F / || A ||_F : " << residual << std::endl;
    std::cout << "Orthogonality || Q^T * Q - I ||_F : " << orthogonality << std::endl;

    if (residual < 1e-4 && orthogonality < 1e-4) {
        std::cout << "Status: TSQR Verification Success!" << std::endl;
    }
    else {
        std::cout << "Status: Verification failed!" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_tau);

    return 0;
    
}

