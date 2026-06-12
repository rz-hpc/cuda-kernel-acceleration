
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// spMV: y = Ax
// kernel loaded example:
// threadsPerBlock = 128
// blocksPerGrid = (N + threadsPerBlock - 1)/ threadsPerBlock
// <<<blocksPerGrid, threadsPerBlock>>>
// scalar: one thread per row
__global__ void spmv_csr_scalar_kernel(const int num_rows,
                                const int* __restrict__ d_row_ptr,
                                const int* __restrict__ d_col_indices,
                                const float* __restrict__ d_values,
                                const float* __restrict__ d_x,
                                float* __restrict__ d_y) {
    // map thread to a specific global row
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float dot = 0.0f;

        int row_start = d_row_ptr[row];
        int row_end = d_row_ptr[row + 1];

        for (int i = row_start; i < row_end; i++) {
            int col = d_col_indices[i];
            float val = d_values[i];
            dot += val * d_x[col];
        }

        d_y[row] = dot;
    }
}

// vector: one warp (32 threads) per row
__global__ void spmv_csr_vector_kernel(const int num_rows,
                                       const int* __restrict__ d_row_ptr,
                                       const int* __restrict__ d_col_indices,
                                       const float* __restrict__ d_values,
                                       const float* __restrict__ d_x,
                                       float* __restrict__ d_y) {

    int row_idx = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (row_idx < num_rows) {
        float sum = 0.0f;

        int row_start = d_row_ptr[row_idx];
        int row_end = d_row_ptr[row_idx + 1];

        for (int i = row_start + lane_id; i < row_end; i += 32) {
            int col_idx = d_col_indices[i];
            sum += d_values[i] * d_x[col_idx];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            d_y[row_idx] = sum;
        }
    }
}

void convertToCSR(std::vector<float>& h_A,
                  int num_rows,
                  int num_cols,
                  std::vector<int>& row_ptr,
                  std::vector<int>& col_indices,
                  std::vector<float>& values ) {
    int nnz = 0;
    row_ptr.resize(num_rows + 1, 0);
    row_ptr[0] = 0;

    for(int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (h_A[i * num_cols + j] != 0.0f) {
                nnz++;
                col_indices.push_back(j);
                values.push_back(h_A[i * num_cols + j]);
            }
        }
        row_ptr[i + 1] = nnz;
    }
}

// Helper to generate a random sparse matrix
void generateRandomSparse(int rows, 
                          int cols,
                          int approx_nnz_per_row,
                          std::vector<float>& A) {
    A.assign(rows * cols, 0.0f);
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.1f, 10.0f);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < approx_nnz_per_row; j++) {
            int col = col_dist(gen);
            A[i * cols + col] = dist(gen);
        }
    }
}

int main() {

    int rows = 10000;
    int cols = 10000;
    int nnz_per_row = 100;

    std::vector<float> h_A;
    generateRandomSparse(rows, cols, nnz_per_row, h_A);
    std::vector<float> h_x(cols, 1.0f);
    std::vector<float> h_y_scalar(rows, 0);
    std::vector<float> h_y_vector(rows, 0);

    std::vector<int> h_row_ptr;
    std::vector<int> h_col_indices;
    std::vector<float> h_values;

    convertToCSR(h_A, rows, cols, h_row_ptr, h_col_indices, h_values);
    int nnz = h_values.size();
    
    int *d_row_ptr, *d_col_indices;
    float *d_values, *d_x, *d_y_scalar, *d_y_vector;
    CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_indices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y_scalar, rows * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y_vector, rows * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlockScalar = 256;
    int blocksPerGridScalar = (rows + threadsPerBlockScalar - 1) / threadsPerBlockScalar;
    spmv_csr_scalar_kernel<<<blocksPerGridScalar, threadsPerBlockScalar>>>(rows, d_row_ptr, d_col_indices, d_values, d_x, d_y_scalar);
    cudaDeviceSynchronize();

    int threadsPerBlockVector = 128;//4 warps per block
    int rowsPerBlock = threadsPerBlockVector / 32;
    int blocksPerGridVector = (rows + rowsPerBlock - 1) / rowsPerBlock;
    spmv_csr_vector_kernel<<<blocksPerGridVector, threadsPerBlockVector>>>(rows, d_row_ptr, d_col_indices, d_values, d_x, d_y_vector);

    CHECK_CUDA(cudaMemcpy(h_y_scalar.data(), d_y_scalar, rows * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_vector.data(), d_y_vector, rows * sizeof(float), cudaMemcpyDeviceToHost));

    bool isSuccess = true;
    // Set an epsilon that accounts for the scale of the numbers
    float epsilon = 1e-3f; 

    for (int i = 0; i < rows; i++) {
        float diff = std::abs(h_y_scalar[i] - h_y_vector[i]);
        
        // Use a relative error check for larger numbers
        float max_val = std::max(std::abs(h_y_scalar[i]), std::abs(h_y_vector[i]));
        float relative_error = (max_val > 1.0f) ? (diff / max_val) : diff;

        if (std::isnan(h_y_scalar[i]) || std::isnan(h_y_vector[i]) || relative_error > epsilon) {
            isSuccess = false;
            std::cout << "Error row " << i << " scalar value " << h_y_scalar[i] \
                      << " vector value " << h_y_vector[i] \
                      << " (Rel Error: " << relative_error << ")" << std::endl;
            break;
        }
    }

    std::cout << "isSuccess: " << (isSuccess ? "TRUE" : "FALSE") << std::endl;

    std::cout << "isSuccess: " << isSuccess << std::endl;

    cudaFree(d_row_ptr); cudaFree(d_col_indices); cudaFree(d_values); 
    cudaFree(d_x); cudaFree(d_y_scalar); cudaFree(d_y_vector);

    return 0;
}
