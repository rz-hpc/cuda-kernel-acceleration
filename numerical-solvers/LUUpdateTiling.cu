
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <math.h>
#include <iomanip>
#include <chrono> // For high-resolution timing

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) {\
        printf("Error: %s in %s at %d! \n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE); \
    } \
}

// find the largest absolute value in column i from row i to row n
__global__ void find_pivot_kernal(float* A, int* pivot_row, int i, int n) {
    if (threadIdx.x == 0) {
        float max_val = fabs(A[i * n + i]);
        int max_idx = i;
        for (int j = i + 1; j < n; j++) {
            float val = fabs(A[j * n + i]);
            if ( val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        pivot_row[0] = max_idx;
    }
}

__global__ void row_swap_kernel(float* A, float* b, int i, int* pivot_row_ptr, int n) {
    // each thread handles a column swap
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int target_row = *pivot_row_ptr;

    if (col_idx < n && i != target_row) {
        // swap A[i][col_idx] with A[target_row][col_idx]
        float tempA = A[i* n + col_idx];
        A[i * n + col_idx] = A[target_row * n + col_idx];
        A[target_row * n + col_idx] = tempA;

        // one thread to swap elements in vector b[i] with b[target_row]
        if (col_idx == 0) {
            float tempb = b[i];
            b[i] = b[target_row];
            b[target_row] = tempb;
        }
    }
}

// LU decomposition: A = LU
// pivot = A[i][i];
// Normalize: A[j][i] /= pivot, j > i
// Update: A[j][k] -= L[j][i] * U[i][k], j > i, k > i
__global__ void lu_normalize_kernel(float* A, int i, int n) {
    int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx > i && row_idx < n) {
        float pivot = A[i * n + i];
        A[row_idx * n + i] /= pivot;
    }
}

__global__ void lu_update_kernel(float* A, int i, int n) {
    int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx > i && row_idx < n && col_idx > i && col_idx < n) {
        A[row_idx * n + col_idx] -= A[row_idx * n + i] * A[i * n + col_idx];
    }
}

// Ax = b -> LUx = b -> Ly = b -> Ux = y

// Lower triangle Ly = b, foward solve
// In-place Ax = b, L is the lower triangle of A with all the diagonal elements == 1
// solve: x[i] = b[i] / L[i][i] = b[i] / 1
// update: b[j] -= L[j][i] * x[i] --> b[j] -= A[j][i] * x[i], j > i
__global__ void forward_solve_kernel(float* A, float* x, float* b, int i, int n) {
    if (threadIdx.x == 0) {
        x[i] = b[i];
    }
}

__global__ void forward_update_kernel(float* A, float* x, float* b, int i, int n) {
    int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx > i && row_idx < n) {
        b[row_idx] -= A[row_idx * n + i] * x[i];
    }
}

// Upper triangle Ux = y, backward solve
// In-place Ax = b, U is the upper triangle of A including diagonals, y is the updated b vector
// solve: x[i] = b[i] / U[i][i] = b[i] / A[i][i]
// update: b[j] -= U[j][i] * x[i] --> b[j] -= A[j][i] * x[i], j < i
__global__ void backward_solve_kernel(float* A, float* x, float* b, int i, int n) {
    if (threadIdx.x == 0) {
        x[i] = b[i] / A[i * n + i];
    }
}

__global__ void backward_update_kernel(float* A, float* x, float* b, int i, int n) {
    int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx < i && row_idx >= 0) {
        b[row_idx] -= A[row_idx * n + i] * x[i];
    }
}

// LU Update: A[j][k] -= L[j][i] * U[i][k] -> A[j][k] -= A[j][i] * A[i][k]
__global__ void lu_update_tiled_kernel(float* A, int i, int n) {
    // Shared memory (block dim 16 * 16, 256 threads)
    // Tiles for shared_pivot_row[16] and shared_pivot_col[16]
    __shared__ float shared_pivot_row[16];
    __shared__ float shared_pivot_col[16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_idx = blockDim.y * blockIdx.y + ty;
    int col_idx = blockDim.x * blockIdx.x + tx;

    // first row (16) threads to load to the shared memory (need to run for all valid matrix index, not just the remaining triangle)
    // the column index in the shared memory row should be the corresponding column thread tx
    if (ty == 0 && col_idx < n) {
        shared_pivot_row[tx] = A[i * n + col_idx];
    }

    // first column (16) threads to load to the shared memory, index corresponding to the row thread ty
    if (tx == 0 && row_idx < n) {
        shared_pivot_col[ty] = A[row_idx * n + i];
    }

    // wait for the row and column tile threads (32 in total) to finish loading the pivot row and column
    __syncthreads();

    if (row_idx > i && row_idx < n && col_idx > i && col_idx < n) {
        A[row_idx * n + col_idx] -= shared_pivot_row[tx] * shared_pivot_col[ty];
    }
}

__host__ void linearSolverHost(float* A, float* x, float* b, int n) {

  size_t sizeN = n * sizeof(float);
  size_t sizeNN = n * sizeN;
  float *d_A, *d_x, *d_b;

  CHECK_CUDA(cudaMalloc((void**)&d_A, sizeNN));
  CHECK_CUDA(cudaMalloc((void**)&d_b, sizeN));
  CHECK_CUDA(cudaMalloc((void**)&d_x, sizeN));

  CHECK_CUDA(cudaMemcpy(d_A, A, sizeNN, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, sizeN, cudaMemcpyHostToDevice));

  int ThreadsPerBlockLU1 = 256;
  int BlocksPerGridLU1 = (n + ThreadsPerBlockLU1 - 1) / ThreadsPerBlockLU1;
  dim3 ThreadsPerBlockLU2(16, 16, 1);
  dim3 BlocksPerGridLU2((n + 16 - 1) / 16, (n + 16 - 1) / 16, 1);

  int* pivot_row;
  CHECK_CUDA(cudaMalloc((void**)&pivot_row, sizeof(int)));

  for (int i = 0; i < n; i++) {
      // Find the pivot
      find_pivot_kernal<<<1, 1>>>(d_A, pivot_row, i, n);

      // swap the rows
      row_swap_kernel<<<BlocksPerGridLU1, ThreadsPerBlockLU1>>>(d_A, d_b, i, pivot_row, n);

      // LU decomposition
      lu_normalize_kernel<<<BlocksPerGridLU1, ThreadsPerBlockLU1>>>(d_A, i, n);

      //lu_update_kernel<<<BlocksPerGridLU2, ThreadsPerBlockLU2>>>(d_A, i, n);
      lu_update_tiled_kernel<<<BlocksPerGridLU2, ThreadsPerBlockLU2>>>(d_A, i, n);

      cudaDeviceSynchronize();
  }

  int ThreadsPerBlockTriangle = 256;
  int BlocksPerGridTriangle = (n + ThreadsPerBlockTriangle - 1) / ThreadsPerBlockTriangle;

  // forward solve

  for (int i = 0; i < n; i++) {
      forward_solve_kernel<<<1, 1>>>(d_A, d_x, d_b, i, n);
      cudaDeviceSynchronize();

      if (i < n - 1) {
          forward_update_kernel<<<BlocksPerGridTriangle, ThreadsPerBlockTriangle>>>(d_A, d_x, d_b, i, n);
          cudaDeviceSynchronize();
      }
  }

  // backward solve

  for (int i = n - 1; i >= 0; i--) {
      backward_solve_kernel<<<1, 1>>>(d_A, d_x, d_x, i, n);
      cudaDeviceSynchronize();

      if (i > 0) {
          backward_update_kernel<<<BlocksPerGridTriangle, ThreadsPerBlockTriangle>>>(d_A, d_x, d_x, i, n);
          cudaDeviceSynchronize();
      }
  }

  CHECK_CUDA(cudaMemcpy(x, d_x, sizeN, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_x));

}

int main1() {
    const int n = 3;
    // This matrix HAS a 0 at A[0][0]. Without pivoting, it fails.
    float h_A[n * n] = {
        0.0f, 1.0f, 1.0f,
        1.0f, 2.0f, 1.0f,
        2.0f, 7.0f, 9.0f
    };
    float h_b[n] = {2.0f, 4.0f, 18.0f};
    float h_x[n] = {0.0f, 0.0f, 0.0f};

    std::cout << "Starting Solver for Pivot-Required Matrix..." << std::endl;
    linearSolverHost(h_A, h_x, h_b, n);

    std::cout << "Solution x: [ ";
    for (int i = 0; i < n; i++) std::cout << std::fixed << std::setprecision(4) << h_x[i] << " ";
    std::cout << "]" << std::endl;

    // Expected Output: [ 1.0000 1.0000 1.0000 ]
    return 0;
}

int main() {
    const int n = 1024;
    size_t sizeNN = n * n * sizeof(float);
    size_t sizeN = n * sizeof(float);

    float* h_A = (float*)malloc(sizeNN);
    float* h_b = (float*)malloc(sizeN);
    float* h_x = (float*)malloc(sizeN);
    float* h_A_copy = (float*)malloc(sizeNN);
    float* h_b_copy = (float*)malloc(sizeN);

    // Fill with random numbers
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_A_copy[i] = h_A[i];
    }
    for (int i = 0; i < n; i++) {
        h_b[i] = (float)rand() / (float)RAND_MAX;
        h_b_copy[i] = h_b[i];
    }

    std::cout << "Starting 1024x1024 Solver..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    linearSolverHost(h_A, h_x, h_b, n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Solved 1,048,576 variables in: " << diff.count() << " seconds" << std::endl;

    // Quick Verify: Check a random row (e.g., row 500)
    // sum(A[500][j] * x[j]) should equal b[500]
    std::cout << "Verifying solution..." << std::endl;
    double max_error = 0.0;
    double total_error = 0.0;

    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            // Use the ORIGINAL h_A here! (Note: h_A was modified by the solver if passed by pointer)
            row_sum += (double)h_A_copy[i * n + j] * h_x[j];
        }
        double error = fabs(row_sum - h_b_copy[i]);
        total_error += error;
        if (error > max_error) max_error = error;
    }

    std::cout << "Average Residual Error: " << total_error / n << std::endl;
    std::cout << "Maximum Residual Error: " << max_error << std::endl;

    free(h_A); free(h_b); free(h_x); free(h_A_copy); free(h_b_copy);
    return 0;
}
