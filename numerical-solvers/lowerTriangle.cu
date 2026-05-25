
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__host__ bool verify_lowerTriangle_cpu(float* h_L, float* h_x, float* h_b, int n) {
    bool isSuccess = true;

    for (int i = 0; i < n; i++) {
        float lhs = 0.0f;
        for (int j = 0; j < n; j++) {
            lhs += h_L[i * n + j] * h_x[j];
        }
        if (std::abs(lhs - h_b[i]) > 1e-3) {
            isSuccess = false;
            printf("Mismatch: Row %d, LHS %f, RHS %f", i, lhs, h_b[i]);
            return isSuccess;
        }
    }

    return isSuccess;

}

__global__ void lowerTriangleNativeKernel(float* d_in, int n) {
    int tid = threadIdx.x;

    int row_idx = tid / n;
    int col_idx = tid % n;

    if (row_idx >= col_idx) {
        printf("lower traingle element row %d, column %d, value %f \n", row_idx, col_idx, d_in[row_idx * n + col_idx]);
    }
}

// thread k solves x_k = b_k / L_kk
// boradcast to threads j > k that b_j -= L_jk * x_k
__global__ void forwardSubUpdateKernel(float* L, float* b, float x_solved, int solved_idx, int n) {
    int row = blockDim.x * blockIdx.x + threadIdx.x; // j

    if (row > solved_idx && row < n) {
        b[row] -= L[row * n + solved_idx] * x_solved;
    }
}

// Updated version with two kernel functions: one to only calculate the step, one to do the update/subtracting
__global__ void solveStepKernel(float* L, float* b, float* x, int i, int n) {
    if (threadIdx.x == 0) {
        x[i] = b[i] / L[i * n + i];
    }
}

__global__ void updateStepKernel(float* L, float* b, float* x, int i, int n) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row > i && row < n) {
        b[row] -= L[row * n + i] * x[i];
    }
}


__host__ void lowerTriangle(float* L, float* x, float* b, int n) {

    size_t sizeN = n * sizeof(float);
    size_t sizeNN = n * sizeN;

    float *d_L, *d_x, *d_b;
    cudaMalloc((void**)&d_L, sizeNN);
    cudaMalloc((void**)&d_x, sizeN);
    cudaMalloc((void**)&d_b, sizeN);

    cudaMemcpy(d_L, L, sizeN * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeN, cudaMemcpyHostToDevice);

    // To be safe
    cudaMemset(d_x, 0, sizeN);

    for (int i = 0; i < n; i++) {

        // solve x[i] on GPU
        solveStepKernel<<<1, 1>>>(d_L, d_b, d_x, i, n);

        // update remaining rows on GPU
        if (i < n -1) {
            // only do the update when there are more rows below/remaining
            // note that without this i < n - 1, kernel update also won't execute for the last row
            // however, it wastes coule of ms for the "no-op(eration)" kernal launch, so better have the if here
            int ThreadsPerBlock = 256;
            int BlocksPerGrid = (n + ThreadsPerBlock - 1) / ThreadsPerBlock;
            updateStepKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(d_L, d_b, d_x, i, n);
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x, sizeN, cudaMemcpyDeviceToHost);

    cudaFree(d_L);
    cudaFree(d_x);
    cudaFree(d_b);
}

int main() {

    int N = 4;
    size_t sizeN = N * sizeof(float);
    size_t sizeNN = N * sizeN;

    float *L, *b, *x;

    L = (float*)malloc(sizeNN);
    b = (float*)malloc(sizeN);
    x = (float*)malloc(sizeN);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            L[i * N + j] = (float)rand() / RAND_MAX;
        }
        b[i] = (float)rand() / RAND_MAX;
    }

    lowerTriangle(L, x, b, N);

    // verify
    //for (int i = 0; i < N; i++) {
        //printf("%d element in x is %f \n", i, x[i]);
    //}
    bool isSuccess = verify_lowerTriangle_cpu(L, x, b, N);
    printf("isSuccess: %d", isSuccess);

    free(L);
    free(b);
    free(x);

    return 0;
}
