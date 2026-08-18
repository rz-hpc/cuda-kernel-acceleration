
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cstdlib>

#define BLOCK_DIM 32

// Trailing Submatrix kernel-- Symmetric GEMM/SYRK
// A22^` = A22 - L21 L21^T
// Subtract the dot product of row i in L21 and row j in L21 transpose from
// lower triangle part of A22
__global__ void trailing_submatrix_naive_kernel(float* d_A, int n, int k_block) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int target_block_row = k_block + 1 + blockIdx.y;
    int target_block_col = k_block + 1 + blockIdx.x;

    // only deal with the lower triangle
    if (target_block_row < target_block_col) return;

    // if deal with diagonal block, only handle the lower triangle
    if (target_block_row == target_block_col && ty < tx) return;

    int global_row_idx = target_block_row * BLOCK_DIM + ty;
    int global_col_idx = target_block_col * BLOCK_DIM + tx;

    // boundary check for global index
    if (global_row_idx >= n || global_col_idx >= n) return;

    // Dot product of L21 L21^T
    // The panel width of L21 block is BLOCK_DIM
    // Element from k_block * BLOCK_DIM to (k_block + 1) * BLOCK_DIM
    float sum = 0.0f;
    int panel_start_col = k_block * BLOCK_DIM;
    for (int dot_idx = 0; dot_idx < BLOCK_DIM; dot_idx++) {
        // row element from L21
        float val_row = d_A[global_row_idx * n + (panel_start_col + dot_idx)];

        // col element from L21^T
        float val_col = d_A[global_col_idx * n + (panel_start_col + dot_idx)];

        sum += val_row * val_col;
    }

    // Subtract from A22 in place for global memory
    d_A[global_row_idx * n + global_col_idx] -= sum;
}

// step 2: column panel update
// L21 = A21(L11^T)^-1
__global__ void column_update_kernel(float* d_A, int n, int k_block) {

    // shared memory for A21 and L11
    __shared__ float tile_A21[BLOCK_DIM][BLOCK_DIM + 1];
    __shared__ float tile_A11[BLOCK_DIM][BLOCK_DIM + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // kernel launched with 1D grid <<<blocks_below, threadsPerBlock>>>
    // so here is blockIdx.x
    int target_block_row = k_block + 1 + blockIdx.x;

    // global idx for diagnoal L11
    int diag_row = k_block * BLOCK_DIM + ty;
    int diag_col = k_block * BLOCK_DIM + tx;

    // A21 global idx
    int target_row = target_block_row * BLOCK_DIM + ty;
    int target_col = k_block * BLOCK_DIM + tx;

    // load values to shared memory
    if (diag_row < n && diag_col < n) {
        tile_A11[ty][tx] = d_A[diag_row * n + diag_col];
    }
    else {
        tile_A11[ty][tx] = 0.0f;
    }

    if (target_row < n && target_col < n) {
        tile_A21[ty][tx] = d_A[target_row * n + target_col];
    }
    else {
        tile_A21[ty][tx] = 0.0f;
    }
    __syncthreads();

    // column below diagnoal update
    // L21 = A21(L11^T)^-1
    // for each column k in the block
    // 1. A21[i][k] /= A11[k][k], i > k
    // 2. A21[i][j] -= A21[i][k] * A11[j][k], j > k (no i >= j here because the whole A21 needs to be updated)
    for (int k = 0; k < BLOCK_DIM; k++) {
        if (tx == 0) {
            tile_A21[ty][k] /= tile_A11[k][k];
        }
        __syncthreads();

        if (tx > k) {
            tile_A21[ty][tx] -= tile_A21[ty][k] * tile_A11[tx][k];
        }
        __syncthreads();
    }

    // write to global memory
    if (target_row < n && target_col < n) {
        d_A[target_row * n + target_col] = tile_A21[ty][tx];
    }
}

// step 1: diagnoal factorization
// A11 = L11 L11^T -> L11 = A11(L11^T)^-1
__global__ void diagnoal_factorization_kernel(float* d_A, int n, int k_block) {

  // shared memory tile
  __shared__ float tile_A[BLOCK_DIM][BLOCK_DIM + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int global_row_idx = k_block * BLOCK_DIM + ty;
  int global_col_idx = k_block * BLOCK_DIM + tx;

  // load the A11 to tile memory
  if (global_row_idx < n && global_col_idx < n) {
      tile_A[ty][tx] = d_A[global_row_idx * n + global_col_idx];
  }
  else {
      tile_A[ty][tx] = 0.0f;
  }
  __syncthreads();

  // for each column in the block
  for (int k = 0; k < BLOCK_DIM; k++) {
      // sqrt
      if (tx == k && ty == k) {
          tile_A[k][k] = sqrtf(tile_A[k][k]);
      }
      __syncthreads();

      // column update
      if (tx == k && ty > k) {
          tile_A[ty][k] /= tile_A[k][k];
      }
      __syncthreads();

      // trailing matrix subtract
      if (tx > k && ty >= tx) {
          tile_A[ty][tx] -= tile_A[ty][k] * tile_A[tx][k];
      }
      __syncthreads();
  }

  // write to the global memory
  if (global_row_idx < n && global_col_idx < n && ty >= tx) { // lower triangle and diagonal
      d_A[global_row_idx * n + global_col_idx] = tile_A[ty][tx];
  }
  else if (global_row_idx < n && global_col_idx < n && ty < tx) {
      d_A[global_row_idx * n + global_col_idx] = 0.0f;
  }
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// CPU reference -- full matrix complete cholesky
void cholesky_cpu_full(float* A, int n) {
    for (int k = 0; k < n; k++) {
        float val = A[k * n + k];
        assert(val > 0.0f && "Assert failure: matrix is not positive definite!");

        // sqrt
        A[k * n + k] = std::sqrt(val);

        // column update
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
        }

        // trailing matrix update
        for (int j = k + 1; j < n; j++) {
            for (int i = j; i < n; i++) {
                A[i * n + j] -= A[i * n + k] * A[j * n + k];
            }
        }

    }

    // clean out the upper triangle
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[i * n + j] = 0.0f;
        }
    }
}

// random symmetric positive definite matrix Generator
// To generate a random number in the range [lower_bound, upper_bound]:
// rand() % (upper_bound - lower_bound + 1) + lower_bound
// A = L L^T
void generate_random_spd_matrix(float* A, int n) {
    std::vector<float> L(n * n, 0.0f);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            // Diagnoal strictly positive
            if (i == j) {
                L[i * n + j] = (float)(rand() % 10 + 1); // random float in [1, 10]
            }
            // off diagnoal could be negative
            else {
                L[i * n + j] = (float)(rand() % 5 - 2); // random float in [-2, 2]
            }
        }
    }

    // compute A = LL^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
}

int main() {
    srand(1337); // seed for reproducibility

    int N = 64;
    int bytes = N * N * sizeof(float);

    std::vector<float> h_A_initial(N * N, 0.0f);
    std::vector<float> h_A_cpu(N * N, 0.0f);
    std::vector<float> h_A_gpu(N * N, 0.0f);

    generate_random_spd_matrix(h_A_initial.data(), N);

    h_A_cpu = h_A_initial;
    h_A_gpu = h_A_initial;

    cholesky_cpu_full(h_A_cpu.data(), N);

    float* d_A = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMemcpy(d_A, h_A_gpu.data(), bytes, cudaMemcpyHostToDevice));

    int total_blocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);

    for (int k_block = 0; k_block < total_blocks; k_block++) {
        // diagonal factorization
        diagnoal_factorization_kernel<<<1, threadsPerBlock>>>(d_A, N, k_block);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        int blocks_below = total_blocks - k_block - 1;
        if (blocks_below > 0) {
            column_update_kernel<<<blocks_below, threadsPerBlock>>>(d_A, N, k_block);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        if (blocks_below > 0) {
            dim3 blocksPerGrid(blocks_below, blocks_below);
            trailing_submatrix_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, N, k_block);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

    }

    CHECK_CUDA(cudaMemcpy(h_A_gpu.data(), d_A, bytes, cudaMemcpyDeviceToHost));

    // Verification
    bool isSuccess = true;
    const float epsilon = 1e-4f;

    // verify the A11 and A21 part (kernel 3 TODO)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) { // check the lower triangle and diagnal (ignore or clean out the upper triangle)
              if (std::abs(h_A_cpu[i * N + j] - h_A_gpu[i * N + j]) > epsilon) {
                  isSuccess = false;
                  std::cout << "Mismatch element at row " << i << " col " << j
                            << " Expected CPU value " << h_A_cpu[i * N + j]
                            << " but got GPU value " << h_A_gpu[i * N + j] << std::endl;
                  break;
              }
        }
    }

    std::cout << "Blocked Cholesky kernels isSuccess: "
     << isSuccess << std::endl;

    cudaFree(d_A);

    return 0;
}
