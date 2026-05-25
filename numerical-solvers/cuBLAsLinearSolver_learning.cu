
#include <iostream>
#include <cuda_runtime.h>

#include <cublas_v2.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) {\
        printf("Error: %s in %s at %d! \n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE); \
    } \
}

__host__ void cublasLinearSolverHost(float* h_A, float* h_x, float* h_b, int n, int batch_size) {

    size_t sizeMat = n * n * batch_size * sizeof(float);
    size_t sizeVec = n * batch_size * sizeof(float);

    float *d_A, *d_b;

    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeMat));
    CHECK_CUDA(cudaMalloc((void**)&d_b, sizeVec));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeMat, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, sizeVec, cudaMemcpyHostToDevice));

    // cuBLAs batching
    // cuBLAs array of pointers on Host (instead of a giant d_A matrix for GPU)
    float* h_A_ptrs[batch_size];
    for (int i = 0; i < batch_size; i++) {
        h_A_ptrs[i] = d_A + i * n * n;
    }

    float* h_b_ptrs[batch_size];
    for (int i = 0; i < batch_size; i++) {
        h_b_ptrs[i] = d_b + i * n;
    }

    // cuBLAs pointer of pointer on GPU (copy the array of pointers on host to gpu)
    float** d_A_ptrs;
    CHECK_CUDA(cudaMalloc((void**)&d_A_ptrs, batch_size * sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(d_A_ptrs, h_A_ptrs, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

    float** d_b_ptrs;
    CHECK_CUDA(cudaMalloc((void**)&d_b_ptrs, batch_size * sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(d_b_ptrs, h_b_ptrs, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

    // Solver

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;

    // Setup cuBLAs handle and auxiliary arrays
    cublasHandle_t handle;
    cublasCreate(&handle);

    int *d_pivot, *d_info;
    CHECK_CUDA(cudaMalloc((void**)&d_pivot, n * batch_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_info, batch_size * sizeof(int)));

    cudaEventRecord(start);

    // LU Factorization
    // This overwrites d_A with L and U combimed
    cublasSgetrfBatched(handle, n, d_A_ptrs, n, d_pivot, d_info, batch_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "cuBLAs LU Factorization milliseconds: " << milliseconds << std::endl;

    cudaEventRecord(start);

    // Solve Ax = b
    // Overwrites d_b with the solution x, forward and backward
    int info_solve = 0;
    cublasSgetrsBatched(handle, CUBLAS_OP_N, n, 1, (const float**)d_A_ptrs, n, d_pivot, d_b_ptrs, n, &info_solve, batch_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "cuBLAs forward and backward solver milliseconds: " << milliseconds << std::endl;

    // The cuBLAs overwrites d_b with the final solution x, so here the source should be d_b
    CHECK_CUDA(cudaMemcpy(h_x, d_b, sizeVec, cudaMemcpyDeviceToHost));

    // Clean up

    cublasDestroy(handle);

    CHECK_CUDA(cudaFree(d_A_ptrs));
    CHECK_CUDA(cudaFree(d_b_ptrs));
    CHECK_CUDA(cudaFree(d_pivot));
    CHECK_CUDA(cudaFree(d_info));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_b));
}

int main() {
    const int n = 32;          // Small matrix
    const int batch_size = 10000; // HUGE batch

    float* h_A = (float*)malloc(n * n * batch_size * sizeof(float));
    float* h_b = (float*)malloc(n * batch_size * sizeof(float));
    float* h_x = (float*)malloc(n * batch_size * sizeof(float));

    float* h_A_copy = (float*)malloc(n * n * batch_size * sizeof(float));
    float* h_b_copy = (float*)malloc(n * batch_size * sizeof(float));

    // Fill with random data for 1000 different problems
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n * n; i++) {
            h_A[b * n * n + i] = (float)rand() / RAND_MAX;
            h_A_copy[b * n * n + i] = h_A[b * n * n + i];
        }
        for (int i = 0; i < n; i++) {
            h_b[b * n + i] = (float)rand() / RAND_MAX;
            h_b_copy[b * n + i] = h_b[b * n + i];
        }
    }

    std::cout << "Launching Batched Solver for " << batch_size << " matrices..." << std::endl;

    cublasLinearSolverHost(h_A, h_x, h_b, n, batch_size);

    // Verification Loop for the Batch
    std::cout << "Verifying 1,000 solutions..." << std::endl;
    double total_max_error = 0.0;
    int failed_matrices = 0;

    for (int b = 0; b < batch_size; b++) {
        float max_error_in_this_matrix = 0.0;

        for (int row = 0; row < n; row++) {
            double row_sum = 0.0;
            for (int col = 0; col < n; col++) {
                // Use the original copies of A and x from the batch
                //row_sum += (double)h_A_copy[b * n * n + row * n + col] * h_x[b * n + col];

                // cuBLAs is column major, need to transpose
                row_sum += (double)h_A_copy[b * n * n + col * n + row] * h_x[b * n + col];
            }

            double error = fabs(row_sum - h_b_copy[b * n + row]);
            if (error > max_error_in_this_matrix) max_error_in_this_matrix = error;
        }

        if (max_error_in_this_matrix > 1e-3) {
            failed_matrices++;
        }
        if (max_error_in_this_matrix > total_max_error) {
            total_max_error = max_error_in_this_matrix;
        }
    }

    std::cout << "Batch Verification Complete!" << std::endl;
    std::cout << "Max Error across all 1,000 matrices: " << total_max_error << std::endl;
    if (failed_matrices > 0) {
        std::cout << "⚠️ Warning: " << failed_matrices << " matrices failed verification!" << std::endl;
    } else {
        std::cout << "✅ All 1,000 matrices solved correctly." << std::endl;
    }

    return 0;
}
