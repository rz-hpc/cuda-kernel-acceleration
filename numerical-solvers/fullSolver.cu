
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernels

#include <math.h>

__global__ void debug_kernel(float* A, float* b, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GPU Check: A[0]=%f, b[0]=%f\n", A[0], b[0]);
    }
}

// To solve system Ax = b
// A = LU -> LUx = b -> Ux = y -> Ly = b

// LU Decomposition A = LU
// Normalize: pivot = A[i][i], A[j][i] /= pivot, j > i
// Update: A[j][k] -= L[j][i] * U[i][k], j > i, k > i, L is calculated/updated in place as a

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

// Forward Sub (lower triangle) Ly = b
// Division: x[i] = b[i] / L[i][i]
// Update: b[j] -=  L[j][i] * x[i]

__global__ void forward_solve_kernel(float* L, float* x, float* b, int i, int n) {
   if (threadIdx.x == 0) {
       x[i] = b[i];// / L[i * n + i]; // since L[i][i] are all 1s, and my kernel doesn't actually put those 1s there for L (inplace A, so the diag has Uii value instead)
   }
}

__global__ void forward_update_kernel(float* L, float* x, float* b, int i, int n) {
   int row_idx = blockDim.x * blockIdx.x + threadIdx.x;


   if (row_idx > i && row_idx < n) {
       b[row_idx] -= L[row_idx * n + i] * x[i];
   }
}

// Backward Sub (upper triangle) Ux = y
// Division: x[i] = y[i] / U[i][i]
// Update: y[j] -= U[j][i] * x[i], j < i

__global__ void backward_solve_kernel(float* U, float* x, float* y, int i, int n) {
   if (threadIdx.x == 0) {
       x[i] = y[i] / U[i * n + i];
   }
}

__global__ void backward_update_kernel(float* U, float* x, float* y, int i, int n) {
   int row_idx = blockDim.x * blockIdx.x + threadIdx.x;


   if (row_idx < i && row_idx >= 0) {
       y[row_idx] -= U[row_idx * n + i] * x[i];
   }
}

__host__ void linearSolverHost(float* h_A, float* h_x, float* h_b, int n) {
    int sizeN = n * sizeof(float);
    int sizeNN = n * sizeN;

    float *d_A, *d_x, *d_b;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeNN));
    CHECK_CUDA(cudaMalloc((void**)&d_x, sizeN));
    CHECK_CUDA(cudaMalloc((void**)&d_b, sizeN));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeNN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, sizeN, cudaMemcpyHostToDevice));

    // launch kernel calls

    // LU Decomposition
   int ThreadsPerBlockLU1 = 256;
   int BlocksPerGridLU1 = (n + ThreadsPerBlockLU1 - 1) / ThreadsPerBlockLU1;
   dim3 ThreadsPerBlockLU2(32, 16, 1);
   dim3 BlocksPerGridLU2( (n + ThreadsPerBlockLU2.x - 1)/ThreadsPerBlockLU2.x, (n + ThreadsPerBlockLU2.y - 1) / ThreadsPerBlockLU2.y, 1);
   for (int i = 0; i < n ; i++) {
       lu_normalize_kernel<<<BlocksPerGridLU1, ThreadsPerBlockLU1>>>(d_A, i, n);
       lu_update_kernel<<<BlocksPerGridLU2, ThreadsPerBlockLU2>>>(d_A, i, n);
   }

   cudaDeviceSynchronize();

   int ThreadsPerBlockTriangle = 256;
   dim3 BlocksPerGridTriangle((n + ThreadsPerBlockTriangle - 1) / ThreadsPerBlockTriangle);

   // Forward, lowerTriangle Ly = b to solve y
   for (int i = 0; i < n; i++) {
       forward_solve_kernel<<<1, 1>>>(d_A, d_x, d_b, i, n);
       cudaDeviceSynchronize();
       if (i < n - 1) {
           forward_update_kernel<<<BlocksPerGridTriangle, ThreadsPerBlockTriangle>>>(d_A, d_x, d_b, i, n);
           cudaDeviceSynchronize();
       }
   }

   //cudaDeviceSynchronize();

   // Backward, upperTriangle Ux = y to solve x
   for (int i = n - 1; i >= 0; i--) {
       backward_solve_kernel<<<1, 1>>>(d_A, d_x, d_x, i, n);

       if (i > 0) {
           backward_update_kernel<<<BlocksPerGridTriangle, ThreadsPerBlockTriangle>>>(d_A, d_x, d_x, i, n);
       }
   }

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(h_x, d_x, sizeN, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_b));
}


int main() {
    int N = 3;
    float h_A[9] = {4, 3, 1, 8, 12, 3, 12, 24, 10};
    float h_b[3] = {1, 2, 3};
    float h_x[3] = {0, 0, 0};

    // Keep copies for verification
    float h_A_orig[9] = {4, 3, 1, 8, 12, 3, 12, 24, 10};
    float h_b_orig[3] = {1, 2, 3};

    linearSolverHost(h_A, h_x, h_b, N);

    printf("Solution x: [ ");
    for(int i=0; i<N; i++) printf("%f ", h_x[i]);
    printf("]\n");

    // CPU Verification: Ax should equal b
    printf("Verification (Ax - b):\n");
    for (int i = 0; i < N; i++) {
        float sum = 0;
        for (int j = 0; j < N; j++) {
            sum += h_A_orig[i * N + j] * h_x[j];
        }
        printf("Row %d: %f (Expected %f)\n", i, sum, h_b_orig[i]);
    }

    return 0;
}
