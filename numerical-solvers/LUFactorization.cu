
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void lu_normalize_kernel(float* d_A, int i, int n) {
    // 1D grid: each thread to handle one row below the pivot row i
    int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx > i && row_idx < n) {
        float pivot = d_A[i * n + i]; // diagonal element

        // calulate L_ji and store in place (j here is row_idx)
        // L_ji = A[j][i] = A[j][i] / A[i][i] = A[j][i] / pivot
        d_A[row_idx * n + i] /= pivot;
    }
}

__global__ void lu_update_kernel(float* d_A, int i, int n) {
    // 2D grids: each thread to handle one element of the remaining sub-matrix
    int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;

    // j (row_idx) > i, k (col_idx) > i
    if (row_idx > i && row_idx < n && col_idx > i && col_idx < n) {
        // A[j][k] -= L[j][i] * U[i][k]
        d_A[row_idx * n + col_idx] -= d_A[row_idx * n + i] * d_A[i * n + col_idx];
    }

}

__host__ void LUFactorization(float* h_A, int n) {

    size_t sizeNN = n * n * sizeof(float);

    float* d_A;

    cudaMalloc((void**)&d_A, sizeNN);

    cudaMemcpy(d_A, h_A, sizeNN, cudaMemcpyHostToDevice);

    int ThreadsPerBlock1 = 256;
    int BlocksPerGrid1 = (n + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;
    dim3 ThreadsPerBlock2(32, 16, 1);
    dim3 BlocksPerGrid2( (n + ThreadsPerBlock2.x - 1) / ThreadsPerBlock2.x, (n + ThreadsPerBlock2.y - 1)/ ThreadsPerBlock2.y, 1);

    for (int i = 0; i < n; i++) {
        // Normalize current column on GPU (calculate L)
        lu_normalize_kernel<<<BlocksPerGrid1, ThreadsPerBlock1>>>(d_A, i, n);

        // Apply Gaussian elimilation on the sub-matrix
        lu_update_kernel<<<BlocksPerGrid2, ThreadsPerBlock2>>>(d_A, i, n);

    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_A, d_A, sizeNN, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
}

int main() {

    int N = 3;
    float h_A[9] = {4, 3, 1, 8, 12, 3, 12, 24, 10};

    LUFactorization(h_A, N);

    float res_ref[9] = {4, 3, 1, 2, 6, 1, 3, 2.5, 4.5};

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (std::abs(h_A[i * N + j] - res_ref[i * N + j]) > 1e-3) {
                printf("Wrong calculation row %d col %d, expecting %f but got %f \n", i, j, res_ref[i * N + j], h_A[i * N + j]);
            }
        }
    }

    return 0;
}
