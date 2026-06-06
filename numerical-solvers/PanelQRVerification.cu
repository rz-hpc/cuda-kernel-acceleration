
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} 

// Panel QR kenel to handle M <= 512 N <= 32
// d_A: M X N
// d_tau: scalar reflection coefficients, array of size N
// lda: leading dimension of A (stride)

#define BLOCK_DIM 32

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

// Helper to compute Frobenius norm (sqrt of sum of all entries square) of an M x N matrix
float frobenius_norm(const std::vector<float>& matrix, int rows, int cols){
    float sum = 0.0f;
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * matrix[i * cols + j];
        }
    }
    return std::sqrt(sum);
}

void verify_panel_qr(const std::vector<float>& A_orig, const std::vector<float>& A_packed, const std::vector<float>& tau, int M, int N, int lda) {
    // Extract R (upper triangle (including diagonal), size M X N)
    std::vector<float> R(M * N, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i <= j) {
                R[i * N + j] = A_packed[i * lda + j];
            }
        }
    }

    // Reconstruct Q (size M X M)
    // Start with Q = Identity Matrix
    std::vector<float> Q(M * M, 0.0f);
    for (int i = 0; i < M; i++) {
        Q[i * M + i] = 1.0f;
    }

    // Apply reflections backwards: Q = H_0 * H_1 * ... * H_{N-1}
    // Do this in-place on an identity matrix for each k from N-1 to 0
    for (int k = N - 1; k >= 0; k--) {
        float t = tau[k];
        if (t == 0.0f) continue;

        // Reconstruct vector v of length M, from packed A lower triangle
        std::vector<float> v(M, 0.0f);
        v[k] = 1.0f; // implicit diagonal
        for (int i = k + 1; i < M; i++) {
            v[i] = A_packed[i * lda + k];
        }

        // Apply H_k to the columns of trailing Q
        // left multiplication: Q = H_0 * H_1 ... * H_{N-1}= H_k * Q
        // Q_new = H_k * Q = (I - tau * v * v^T) * Q
        // expand to Q_col = Q_col - tau * v * (v^T * Q)
        // Q_v[j] = v^T * Q[:,j] (dot of v with j-th column of Q)
        std::vector<float> vTQ(M, 0.0f);
        for (int j = 0; j < M; j++)
            for (int i = k; i < M; i++)
                vTQ[j] += v[i] * Q[i * M + j];   // vTQ[j] = v^T · Q[:,j]

        for (int i = k; i < M; i++)
            for (int j = 0; j < M; j++)
                Q[i * M + j] -= t * v[i] * vTQ[j]; // Q = Q - tau·v·(v^T·Q)

    }

    // Compute QR product
    // Q: M x M
    // R: M x N
    // QR: M x N
    std::vector<float> QR(M * N, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                sum += Q[i * M + k] * R[k * N + j];
            }
            QR[i * N + j] = sum;
        }
    }

    // Check orthogonality of Q
    // Q^T * Q = Identity (M x M)
    std::vector<float> QtQ(M * M, 0.0f);
    std::vector<float> Identity(M * M, 0.0f);
    for (int i = 0; i < M; i++) {
        Identity[i * M + i] = 1.0f;
        for (int j = 0; j < M; j++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                sum += Q[k * M + i] * Q[k * M + j];
            }
            QtQ[i * M + j] = sum;
        }
    }

    // Calculate Residual metrics
    std::vector<float> diff_A(M * N);
    for (size_t i = 0; i < diff_A.size(); i++) {
        diff_A[i] = A_orig[i] - QR[i];
    }

    std::vector<float> diff_I(M * M);
    for (size_t i = 0; i < diff_I.size(); i++) {
        diff_I[i] = Identity[i] - QtQ[i];
    }

    float res_A = frobenius_norm(diff_A, M, N) / frobenius_norm(A_orig, M, N);
    float res_I = frobenius_norm(diff_I, M, M);

    std::cout << "\n===PANEL QR VERIFICATION RESULTS===\n";
    std::cout << "Residual ||A - QR||_F / ||A||_F : " << res_A << "\n";
    std::cout << "Orthogonality ||I - Q^T*Q||_F : " << res_I << "\n";

    // Numerical tolerance check for single-precision floats
    if (res_A < 1e-4 && res_I < 1e-4) {
        std::cout << "Status: Verification Success!\n";
    } 
    else {
        std::cout << "Status: Verification Failed!\n";
    }
}

int main() {

    int M = 4;
    int N = 3;
    int lda = N; // leading dimension of A

    // Define a basic non-sigular input matrix
    std::vector<float> h_A_orig = {
        12, -51, 4,
        6, 167, -68,
        -4, 24, -41,
        1, 2, 3
    };
    std::vector<float> h_A = h_A_orig;
    std::vector<float> h_tau(N, 0.0f);

    float *d_A, *d_tau;
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * lda * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_tau, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * lda * sizeof(float), cudaMemcpyHostToDevice));

    // lauch with 1 block, 32 threads
    panel_qr_baseline<<<1, 32>>>(d_A, d_tau, M, N, lda);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(h_A.data(), d_A, M * lda * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_tau.data(), d_tau, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify it
    verify_panel_qr(h_A_orig, h_A, h_tau, M, N, lda);

    cudaFree(d_A);
    cudaFree(d_tau);

    return 0;
}
