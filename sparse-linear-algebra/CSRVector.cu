
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

// stress test
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

__global__ void spmv_csr_vector_kernel(
    const int num_rows,
    const int* row_ptr,
    const int* col_indices,
    const float* values,
    const float* x,
    float* y) {

      // One row per warp (32 threads)
      int row_idx = (blockDim.x * blockIdx.x + threadIdx.x) / 32;

      // Thread id within the warp
      int lane = threadIdx.x % 32;

      if (row_idx < num_rows) {
          // row start and end index in vector x and v
          // i.e. row[i] and row[i+1] for row i
          int row_start = row_ptr[row_idx];
          int row_end = row_ptr[row_idx + 1];

          float product = 0.0f;

          // Coalesced loop for the whole wrap
          // Start with current thread (so there is "+ lane")
          for (int i = row_start + lane; i < row_end; i += 32) {
              int col_idx = col_indices[i];
              product += values[i] * x[col_idx];
          }

          // Warp shuffle reduction
          for (int offset = 32 / 2; offset > 0; offset >>= 1) {
              product += __shfl_down_sync(0xffffffff, product, offset);
          }

          // Thread id 0 writes the final result of the row to the output y
          if (lane == 0) {
              y[row_idx] = product;
          }

      }
}

void convertToCSR(const std::vector<float>& A,
                  const int num_rows,
                  const int num_cols,
                  std::vector<int>& row_ptr,
                  std::vector<int>& col_indices,
                  std::vector<float>& values) {

    row_ptr.clear();
    row_ptr.push_back(0);
    int current_nnz = 0;

    for (int i = 0; i < num_rows; i++){
        for (int j = 0; j < num_cols; j++) {
            float val = A[i * num_cols + j];
            if (val != 0.0f) {
                col_indices.push_back(j);
                values.push_back(val);
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

    convertToCSR(h_A, rows, cols, h_row_ptr_vec, h_col_indices_vec, h_values_vec);

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

__host__ void spmv_csr_vector_host(const std::vector<float>& h_A,
                                   const int rows,
                                   const int cols,
                                   const std::vector<float>& h_x,
                                   std::vector<float>& h_y) {

      // Discovery
      std::vector<int> h_row_ptr_vec;
      std::vector<int> h_col_indices_vec;
      std::vector<float> h_values_vec;

      convertToCSR(h_A, rows, cols, h_row_ptr_vec, h_col_indices_vec, h_values_vec);

      int nnz = h_values_vec.size();
      int num_rows = h_row_ptr_vec.size() - 1;

      int *d_row_ptr, *d_col_indices;
      float* d_values;

      CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&d_col_indices, nnz * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&d_values, nnz * sizeof(float)));

      CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr_vec.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_col_indices, h_col_indices_vec.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_values, h_values_vec.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

      float *d_x, *d_y;
      CHECK_CUDA(cudaMalloc((void**)&d_x, cols * sizeof(float)));
      CHECK_CUDA(cudaMalloc((void**)&d_y, rows * sizeof(float)));

      CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

      // Kernel call
      spmv_csr_vector_kernel<<<1, 128>>>(num_rows, d_row_ptr, d_col_indices, d_values, d_x, d_y);

      CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

      CHECK_CUDA(cudaFree(d_row_ptr));
      CHECK_CUDA(cudaFree(d_col_indices));
      CHECK_CUDA(cudaFree(d_values));

      CHECK_CUDA(cudaFree(d_x));
      CHECK_CUDA(cudaFree(d_y));

}

// Helper to generate a random sparse matrix
void generateRandomSparse(int rows, int cols, int approx_nnz_per_row,
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

int main2() {

    std::vector<float> A = {10, 20, 0, 0, 0, 30, 0, 40, 0, 0, 50, 0, 60, 0, 70, 80};
    std::vector<float> x = {1, 1, 1, 1};
    std::vector<float> y_expected = {30, 70, 50, 210};
    std::vector<float> y(4, 0);

    spmv_csr_vector_host(A, 4, 4, x, y);

    bool isSuccess = true;
    for (int i = 0; i < 4; i++) {
        if (y[i] != y_expected[i]) {
            isSuccess = false;
            printf("Calculate Error: %d element is expected to be %f, but got %f!", i, y_expected[i], y[i]);
            break;
        }
    }

    std::cout << "isSuccess: " << isSuccess << std::endl;

    return 0;
}

int main() {
    int rows = 10000;
    int cols = 10000;
    int nnz_per_row = 100;

    std::cout << "Step 1: Generating " << rows << "x" << cols << " matrix..." << std::endl;
    std::vector<float> h_A;
    generateRandomSparse(rows, cols, nnz_per_row, h_A);

    std::vector<float> h_x(cols, 1.0f);
    std::vector<float> h_y_scalar(rows, 0);
    std::vector<float> h_y_vector(rows, 0);

    // Step 2: Convert to CSR (Once, on Host)
    std::vector<int> h_row_ptr, h_col_indices;
    std::vector<float> h_values;
    convertToCSR(h_A, rows, cols, h_row_ptr, h_col_indices, h_values);
    int nnz = h_values.size();

    // Step 3: GPU Allocation & Upload
    int *d_row_ptr, *d_col_indices;
    float *d_values, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_indices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    // Timing Setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_scalar = 0, ms_vector = 0;

    // --- TEST 1: SCALAR KERNEL ---
    int threads_scalar = 256;
    int blocks_scalar = (rows + threads_scalar - 1) / threads_scalar;

    cudaEventRecord(start);
    spmv_csr_kernel<<<blocks_scalar, threads_scalar>>>(rows, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_scalar, start, stop);
    CHECK_CUDA(cudaMemcpy(h_y_scalar.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // --- TEST 2: VECTOR KERNEL (32 threads per row) ---
    int threads_vector = 128; // 4 warps per block
    int rows_per_block = threads_vector / 32;
    int blocks_vector = (rows + rows_per_block - 1) / rows_per_block;

    cudaEventRecord(start);
    spmv_csr_vector_kernel<<<blocks_vector, threads_vector>>>(rows, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_vector, start, stop);
    CHECK_CUDA(cudaMemcpy(h_y_vector.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Results
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Scalar Kernel Time: " << ms_scalar << " ms" << std::endl;
    std::cout << "Vector Kernel Time: " << ms_vector << " ms" << std::endl;
    std::cout << "Speedup: " << ms_scalar / ms_vector << "x" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // Accuracy Check (Numerical Epsilon)
    bool match = true;
    for (int i = 0; i < rows; i++) {
        if (fabs(h_y_scalar[i] - h_y_vector[i]) > 1e-3) {
            match = false;
            std::cout << "Mismatch at " << i << ": Scalar=" << h_y_scalar[i] << " Vector=" << h_y_vector[i] << std::endl;
            break;
        }
    }
    std::cout << "Results Match: " << (match ? "YES" : "NO") << std::endl;

    // Cleanup
    cudaFree(d_row_ptr); cudaFree(d_col_indices); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
