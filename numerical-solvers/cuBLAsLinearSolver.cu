
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) {\
        printf("Error: %s in %s at %d! \n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE); \
    } \
}

// Dedicated macro for cuBLAS errors
#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error: %d in %s at %d! \n", status, __FILE__, __LINE__); \
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

    float* h_A_ptrs[batch_size];
    float* h_b_ptrs[batch_size];
    for (int i = 0; i < batch_size; i++) {
        h_A_ptrs[i] = d_A + i * n * n;
        h_b_ptrs[i] = d_b + i * n;
    }

    float **d_A_ptrs, **d_b_ptrs;
    CHECK_CUDA(cudaMalloc((void**)&d_A_ptrs, batch_size * sizeof(float*)));
    CHECK_CUDA(cudaMalloc((void**)&d_b_ptrs, batch_size * sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(d_A_ptrs, h_A_ptrs, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_ptrs, h_b_ptrs, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int *d_pivot, *d_info;
    CHECK_CUDA(cudaMalloc((void**)&d_pivot, n * batch_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_info, batch_size * sizeof(int)));

    // Step 1: LU Factorization
    CHECK_CUBLAS(cublasSgetrfBatched(handle, n, d_A_ptrs, n, d_pivot, d_info, batch_size));

    // Step 2: Solve (In-place on d_b)
    int info_solve = 0;
    CHECK_CUBLAS(cublasSgetrsBatched(handle, CUBLAS_OP_N, n, 1, (const float**)d_A_ptrs, n, d_pivot, d_b_ptrs, n, &info_solve, batch_size));

    CHECK_CUDA(cudaMemcpy(h_x, d_b, sizeVec, cudaMemcpyDeviceToHost));

    cublasDestroy(handle);
    cudaFree(d_A_ptrs); cudaFree(d_b_ptrs);
    cudaFree(d_pivot); cudaFree(d_info);
    cudaFree(d_A); cudaFree(d_b);
}

int main() {
    const int n = 32;
    const int batch_size = 10000;

    float* h_A = (float*)malloc(n * n * batch_size * sizeof(float));
    float* h_b = (float*)malloc(n * batch_size * sizeof(float));
    float* h_x = (float*)malloc(n * batch_size * sizeof(float));

    // To verify, we need the original A and b
    float* h_A_orig = (float*)malloc(n * n * batch_size * sizeof(float));
    float* h_b_orig = (float*)malloc(n * batch_size * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < n; j++) { // j is column
            for (int i = 0; i < n; i++) { // i is row
                // COLUMN-MAJOR: index = col * n + row
                float val = (float)rand() / RAND_MAX;
                h_A[b * n * n + j * n + i] = val;
                h_A_orig[b * n * n + j * n + i] = val;
            }
        }
        for (int i = 0; i < n; i++) {
            float val = (float)rand() / RAND_MAX;
            h_b[b * n + i] = val;
            h_b_orig[b * n + i] = val;
        }
    }

    cublasLinearSolverHost(h_A, h_x, h_b, n, batch_size);

    // Final Verification
    double total_max_err = 0;
    int failed = 0;
    for (int b = 0; b < batch_size; b++) {
        float max_err = 0;
        for (int i = 0; i < n; i++) { // Row
            double sum = 0;
            for (int j = 0; j < n; j++) { // Col
                // Column-Major access: j * n + i
                sum += (double)h_A_orig[b * n * n + j * n + i] * h_x[b * n + j];
            }
            double err = fabs(sum - h_b_orig[b * n + i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 1e-3) failed++;
        if (max_err > total_max_err) total_max_err = max_err;
    }

    std::cout << "Verification Result: " << (failed == 0 ? "✅ PASS" : "⚠️ FAIL") << std::endl;
    std::cout << "Max Error: " << total_max_err << std::endl;
    if (failed > 0) std::cout << "Failed matrices: " << failed << std::endl;

    return 0;
}
