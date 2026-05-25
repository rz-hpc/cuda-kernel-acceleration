
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) {\
        printf("Error: %s in %s at %d! \n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE); \
    } \
}

// Find pivot batched kernel-- find the element A[j][i] with biggest abs value, j >= i so the row j be the pivot row
// in both the non-batched and batched versions, only threadIdx.x == 0 does the work
// Now each batch needs its own pivot row, the pivot_row_array[batch_size] vs before just one row
// In this case, batch_idx = blockIdx.x, one block per matrix in the batch
__global__ void find_pivot_batched_kernel(float* A, int* pivot_row_array, int i, int n, int batch_size) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    float* batch_A = &A[batch_idx * n * n];

    if (threadIdx.x == 0) {
        float max_val = fabs(batch_A[i * n + i]);
        int max_idx = i;
        for (int j = i + 1; j < n; j++) {
            float cur_val = fabs(batch_A[j * n + i]);
            if (cur_val > max_val) {
                max_val = cur_val;
                max_idx = j;
            }
        }
        pivot_row_array[batch_idx] = max_idx;
    }

}

// swap kernel: swap current row with the pivot row (of a given batch)
// Both matrix A and vector b need to be swapped together
__global__ void row_swap_batched_kernel(float* A, float* b, int row, int* pivot_row_array, int n, int batch_size) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) {
        return;
    }

    float* batch_A = &A[batch_idx * n * n];
    float* batch_b = &b[batch_idx * n];

    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int target_row = pivot_row_array[batch_idx];
    if (col_idx < n && row != target_row) {
        float tempA = batch_A[row * n + col_idx];
        batch_A[row * n + col_idx] = batch_A[target_row * n + col_idx];
        batch_A[target_row * n + col_idx] = tempA;

        if (col_idx == 0) { //exactly one thread in the entire row across all blocks
            float tempB = batch_b[row];
            batch_b[row] = batch_b[target_row];
            batch_b[target_row] = tempB;
        }
    }
}

// A = LU
// Normalize: A[j][i] /= A[i][i], j > i, the column
__global__ void lu_normalize_batched_kernel(float *A, int i, int n, int batch_size) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) {
        return;
    }
    float* batch_A = &A[batch_idx * n * n];

    int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(row_idx > i && row_idx < n) {
        batch_A[row_idx * n + i] /= batch_A[i * n + i];
    }
}

// Update: A[j][k] -= A[j][i] * A[i][k], j > i, k > i
// For tiling, shared_pivot_row[tx] to store all the element of row i
// shared_pivot_col[ty] to store all the element of column i
// Do the update in shared memory
__global__ void lu_update_batched_kernel(float* A, int i, int n, int batch_size) {

    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size) {
        return;
    }

    // Offset to the batch corresponding start index in A
    float* batch_A = &A[batch_idx * n * n];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_idx = blockDim.y * blockIdx.y + ty;
    int col_idx = blockDim.x * blockIdx.x + tx;

    __shared__ float shared_pivot_row[16];
    __shared__ float shared_pivot_col[16];

    if (ty == 0 && col_idx < n) {
        shared_pivot_row[tx] = batch_A[i * n + col_idx];
    }

    if (tx == 0 && row_idx < n) {
        shared_pivot_col[ty] = batch_A[row_idx * n + i];
    }

    __syncthreads();

    if (row_idx > i && row_idx < n && col_idx > i && col_idx < n) {
        batch_A[row_idx * n + col_idx] -= shared_pivot_col[ty] * shared_pivot_row[tx];
    }

}

// Ax = b -> LUx = b -> Ly = b, Ux = y
// forward solver Ly = b -> Ax = b
// solve: x[i] = b[i] / A[i][i]
// because for L, all the diagonal elements are 1s, we don't actually need the matrix A here
// vector x and b need to be offset to batch
__global__ void forward_solve_batched_kernel(float* A, float* x, float* b, int i, int n, int batch_size) {

    int batch_idx = blockIdx.x; // grid: dim3(batch_size, 1, 1)
    if (batch_idx >= batch_size) {
        return;
    }

    float* batch_b = &b[batch_idx * n];
    float* batch_x = &x[batch_idx * n];

    if (threadIdx.x == 0) {
        batch_x[i] = batch_b[i];
    }
}

// update: b[j] -= A[j][i] * x[i], j > i
__global__ void forward_update_batched_kernel(float* A, float* x, float* b, int i, int n, int batch_size) {

    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) {
        return;
    }

    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    float* batch_A = &A[batch_idx * n * n];
    float* batch_x = &x[batch_idx * n];
    float* batch_b = &b[batch_idx * n];

    if (row_idx > i && row_idx < n) {
        batch_b[row_idx] -= batch_A[row_idx * n + i] * batch_x[i];
    }

}

// Backward solver Ux = y --> Ax = b
// Solve: x[i] = b[i] / A[i][i]
__global__ void backward_solve_batched_kernel(float* A, float* x, float* b, int i, int n, int batch_size) {
    int batch_idx = blockIdx.x; // grid: dim3(batch_size, 1, 1)
    if (batch_idx >= batch_size) {
        return;
    }

    float* batch_A = &A[batch_idx * n * n];
    float* batch_x = &x[batch_idx * n];
    float* batch_b = &b[batch_idx * n];

    if (threadIdx.x == 0) {
        batch_x[i] = batch_b[i] / batch_A[i * n + i];
    }
}

// Update: b[j] -= A[j][i] * x[i], j < i
__global__ void backward_update_batched_kernel(float* A, float* x, float* b, int i, int n, int batch_size) {

    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) {
        return;
    }

    float* batch_A = &A[batch_idx * n * n];
    float* batch_x = &x[batch_idx * n];
    float* batch_b = &b[batch_idx * n];

    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < i && row_idx >= 0) {
        batch_b[row_idx] -= batch_A[row_idx * n + i] * batch_x[i];
    }

}

__host__ void linearSolverBatchedHost(float* h_A, float* h_x, float* h_b, int n, int batch_size) {
    size_t sizeMat = n * n * sizeof(float);
    size_t sizeVec = n * sizeof(float);

    float *d_A, *d_b, *d_x;
    int *pivot_row_array;

    // 1. Allocate for the WHOLE BOOK of matrices
    CHECK_CUDA(cudaMalloc(&d_A, sizeMat * batch_size));
    CHECK_CUDA(cudaMalloc(&d_b, sizeVec * batch_size));
    CHECK_CUDA(cudaMalloc(&d_x, sizeVec * batch_size));
    CHECK_CUDA(cudaMalloc(&pivot_row_array, batch_size * sizeof(int)));

    // 2. Copy the entire stack to the GPU
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeMat * batch_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, sizeVec * batch_size, cudaMemcpyHostToDevice));

    dim3 blockPivot(1, 1, 1);
    dim3 gridPivot(batch_size, 1, 1);

    dim3 blockSwap(256, 1, 1);
    dim3 gridSwap((n + 255) / 256, batch_size, 1);

    dim3 blockLUNormalize(256, 1, 1);
    dim3 gridLUNormalize((n + 255) / 256, batch_size, 1);

    dim3 blockLUUpdate(16, 16, 1);
    dim3 gridLUUpdate((n + 15) / 16, (n + 15) / 16, batch_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;

    cudaEventRecord(start);

    for (int i = 0; i < n; i++) {
        // find the pivot
        find_pivot_batched_kernel<<<gridPivot, blockPivot>>>(d_A, pivot_row_array, i, n, batch_size);

        // swap
        row_swap_batched_kernel<<<gridSwap, blockSwap>>>(d_A, d_b, i, pivot_row_array, n, batch_size);

        // LU decomposition
        lu_normalize_batched_kernel<<<gridLUNormalize, blockLUNormalize>>>(d_A, i, n, batch_size);

        // LU update
        lu_update_batched_kernel<<<gridLUUpdate, blockLUUpdate>>>(d_A, i, n, batch_size);

        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "LU Decomposition elapsed time milliseconds: " << milliseconds << std::endl;

    dim3 blockSolve(1, 1, 1);
    dim3 gridSolve(batch_size, 1, 1);

    dim3 blockUpdate(256, 1, 1);
    dim3 gridUpdate((n + 255) / 256, batch_size, 1);

    cudaEventRecord(start);

    // forward solver
    for (int i = 0; i < n; i++) {
        // Solve
        forward_solve_batched_kernel<<<gridSolve, blockSolve>>>(d_A, d_x, d_b, i, n, batch_size);
        cudaDeviceSynchronize();

        // Update
        if (i < n - 1) {
            forward_update_batched_kernel<<<gridUpdate, blockUpdate>>>(d_A, d_x, d_b, i, n, batch_size);
            cudaDeviceSynchronize();
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Forward solver elapsed time milliseconds: " << milliseconds << std::endl;

    cudaEventRecord(start);

    // backward solver
    for (int i = n - 1; i >= 0; i--) {
        // Solve
        backward_solve_batched_kernel<<<gridSolve, blockSolve>>>(d_A, d_x, d_x, i, n, batch_size);
        cudaDeviceSynchronize();

        // Update
        if (i > 0) {
            backward_update_batched_kernel<<<gridUpdate, blockUpdate>>>(d_A, d_x, d_x, i, n, batch_size);
            cudaDeviceSynchronize();
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Backward solver elapsed time milliseconds: " << milliseconds << std::endl;

    CHECK_CUDA(cudaMemcpy(h_x, d_x, sizeVec * batch_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
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

    linearSolverBatchedHost(h_A, h_x, h_b, n, batch_size);

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
                row_sum += (double)h_A_copy[b * n * n + row * n + col] * h_x[b * n + col];
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
