
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
}

// Solve the matrix A (in CSR format) multiply by vector x, result saved in y
__global__ void spmv_csr_kernel(
    const int num_rows,
    const int* row_ptr,
    const int* col_indices,
    const float* values,
    const float* x,
    float* y) {

      // One thread Per Row
      int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

      if (row_idx < num_rows) {
          // row start index in v and c
          int row_start = row_ptr[row_idx];
          int row_end = row_ptr[row_idx + 1];

          float product = 0.0f;

          for (int i = row_start; i < row_end; i++) {
              int col_idx = col_indices[i];
              product += x[col_idx] * values[i];
          }

          y[row_idx] = product;
      }
}

// Discovery
void convertToCSR(const std::vector<float>& A, int rows, int cols,
                    std::vector<float>& values,
                    std::vector<int>& col_indices,
                    std::vector<int>& row_ptr) {

    row_ptr.clear();
    row_ptr.push_back(0);
    int current_nnz = 0; // number of none zero elements

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = A[i * cols + j];
            if (val != 0.0f) {
                values.push_back(val);
                col_indices.push_back(j);
                current_nnz++;
            }
        }
        row_ptr.push_back(current_nnz);
    }
}

__host__ void spmv_csr_host(const std::vector<float>& h_A, const int rows, const int cols,
                            const std::vector<float>& h_x,
                            std::vector<float>& h_y) {

    // Discovery
    // Use std::vector for dynamic allocation because before the discovery, the nnz is unknown
    std::vector<float> h_values_vec;
    std::vector<int> h_col_indices_vec;
    std::vector<int> h_row_ptr_vec;

    convertToCSR(h_A, rows, cols, h_values_vec, h_col_indices_vec, h_row_ptr_vec);

    int nnz = h_values_vec.size();

    // For decoupled reality and always following the N + 1 Rule
    // row ptr vector size is row number + 1
    int num_rows = h_row_ptr_vec.size() - 1;

    int *d_row_ptr, *d_col_indices;
    float *d_values;
    float *d_x, *d_y;

    CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_indices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, rows * sizeof(float)));

    // Use vector.data() to get raw float* and int* for cudaMemcpy
    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr_vec.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices, h_col_indices_vec.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values_vec.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel call
    spmv_csr_kernel<<<32, 16>>>(rows, d_row_ptr, d_col_indices, d_values, d_x, d_y);

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_row_ptr));
    CHECK_CUDA(cudaFree(d_col_indices));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

}

int main() {

    std::vector<float> A = {10, 20, 0, 0, 0, 30, 0, 40, 0, 0, 50, 0, 60, 0, 70, 80};
    std::vector<float> x = {1, 1, 1, 1};
    std::vector<float> y_expected = {30, 70, 50, 210};
    std::vector<float> y(4, 0);

    spmv_csr_host(A, 4, 4, x, y);

    bool isSuccess = true;
    for (int i = 0; i < 4; i++) {
        if (y[i] != y_expected[i]) {
            isSuccess = false;
            printf("Calculate Error: %d element is expected to be %f, but got %f!", i, y_expected[i], y[i]);
            break;
        }
    }

    std::cout << "isSuccess: " << isSuccess << std::endl;
}
