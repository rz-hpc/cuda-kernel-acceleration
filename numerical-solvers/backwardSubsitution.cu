
#include <iostream>
#include <cuda_runtime.h>

__global__ void solveStepUKernel(float* y, float* U, int i, int n) {
    // solve current x[i] in place
    if ( threadIdx.x == 0) {
        y[i] = y[i] / U[i * n + i];
    }
}

__global__ void updateStepUkernel(float* y, float* U, int i, int n) {
    // boardcast the result to the rows above i
    int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx < i && row_idx >= 0 ) {
        y[row_idx] -= U[row_idx * n + i] * y[i];
    }
}

__host__ void backwardSubsitution(float *h_y, *h_U, int n) {

    size_t sizeN = n * sizeof(float);
    size_t sizeNN = n * sizeN;

    float *d_y, *d_U;

    cudaMalloc((void**)&d_y, sizeN);
    cudaMalloc((void**)&d_U, sizeNN);

    cudaMemcpy(d_y, h_y, sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, sizeNN, cudaMemcpyHostToDevice);

    for (int i = n - 1; i >= 0; i--) {
        int ThreadsPerBlock1 = 1;
        int BlocksPerGrid1 = 1;
        solveStepUKernel<<<BlocksPerGrid1, ThreadsPerBlock1>>>(d_y, d_U, i, n);

        // update the above/remaining rows
        if ( i > 0) {
            // only do the update when there are more rows below/remaining
            // note that without this i > 0, kernel update also won't execute for the last row
            // however, it wastes coule of ms for the "no-op(eration)" kernal launch, so better have the if here
            int ThreadsPerBlock2 = 256;
            int BlocksPerGrid2 = (n + ThreadsPerBlock2 - 1) / ThreadsPerBlock2;
            updateStepUkernel<<<BlocksPerGrid2, ThreadsPerBlock2>>>(d_y, d_U, i, n);
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, sizeN, cudaMemcpyDeviceToHost);

}

int main() {

    return 0;
}
