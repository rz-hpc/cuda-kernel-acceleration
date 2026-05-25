
#include <iostream>
#include <cuda_runtime.h>

// 1. LU Decomposition
__global__ void lu_normalize_kernel(float* A, int i, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > i && row < n) A[row * n + i] /= A[i * n + i];
}

__global__ void lu_update_kernel(float* A, int i, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > i && row < n && col > i && col < n) {
        A[row * n + col] -= A[row * n + i] * A[i * n + col];
    }
}

// 2. Forward Substitute (Ly = b) -> L has 1s on diag, so y[i] = b[i]
__global__ void forward_update_kernel(float* A, float* b, int i, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > i && row < n) {
        b[row] -= A[row * n + i] * b[i];
    }
}

// 3. Backward Substitute (Ux = y) -> x[i] = y[i] / U[i][i]
__global__ void backward_solve_kernel(float* A, float* b, int i, int n) {
    if (threadIdx.x == 0) b[i] /= A[i * n + i];
}

__global__ void backward_update_kernel(float* A, float* b, int i, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < i) {
        b[row] -= A[row * n + i] * b[i];
    }
}

__host__ void linearSolverHost(float* h_A, float* h_x, float* h_b, int n) {
    float *d_A, *d_b;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // --- LU ---
    for (int i = 0; i < n; i++) {
        lu_normalize_kernel<<<(n+255)/256, 256>>>(d_A, i, n);
        lu_update_kernel<<<dim3((n+15)/16, (n+15)/16), dim3(16, 16)>>>(d_A, i, n);
    }
    cudaDeviceSynchronize();

    // --- FORWARD (b becomes y) ---
    for (int i = 0; i < n; i++) {
        forward_update_kernel<<<1, 256>>>(d_A, d_b, i, n);
        cudaDeviceSynchronize(); // Crucial: row i update must finish
    }

    // --- BACKWARD (y becomes x) ---
    for (int i = n - 1; i >= 0; i--) {
        backward_solve_kernel<<<1, 1>>>(d_A, d_b, i, n);
        cudaDeviceSynchronize(); // Crucial: solve must finish before update starts

        if (i > 0) {
            backward_update_kernel<<<1, 256>>>(d_A, d_b, i, n);
            cudaDeviceSynchronize(); // Crucial: update must finish before next solve starts
        }
    }

    cudaMemcpy(h_x, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_b);
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
