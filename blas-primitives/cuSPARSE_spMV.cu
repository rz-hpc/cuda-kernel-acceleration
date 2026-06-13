
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

#define CHECK_CUDA(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
  }\
}

#define CHECK_CUSPARSE(call) { \
  cusparseStatus_t status = call; \
  if (status != CUSPARSE_STATUS_SUCCESS) { \
      printf("cuSparse Error: %d in %s at %d!", status, __FILE__, __LINE__); \
      exit(EXIT_FAILURE); \
  } \
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

// Compute SpMV: y = a A x + b y
int main() {

    int rows = 10000;
    int cols = 10000;
    int nnz_per_row = 100;

    std::vector<float> h_A;
    generateRandomSparse(rows, cols, nnz_per_row, h_A);
    std::vector<float> h_x(cols, 1.0f);
    std::vector<float> h_y(rows, 0.0f);

    std::vector<int> h_row_ptr;
    std::vector<int> h_col_indices;
    std::vector<float> h_values;

    convertToCSR(h_A, rows, cols, h_row_ptr, h_col_indices, h_values);
    int nnz = h_values.size();

    int *d_row_ptr, *d_col_indices;
    float *d_values, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_indices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, rows * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));


    // 1. Handle (once on startup)
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    //cusparseSetStream(handle, stream);

    // 2. Descriptors (once per matrix/vector, no data is moved)
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(
      &matA, 
      (int64_t)rows, (int64_t)cols, (int64_t)nnz, // rows, cols, non-zeros
      (void*)d_row_ptr,  // int32* on device
      (void*)d_col_indices,  // int32* on device
      (void*)d_values,    // float* on device
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F);

    // 3. Workspace (once per op shape)
    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize = 0;

    // query required bytes (pass NULL workspace)
    cusparseSpMV_bufferSize(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE, // for A^Tx use _transpose
      &alpha, matA, vecX, &beta, vecY,
      CUDA_R_32F, 
      CUSPARSE_SPMV_ALG_DEFAULT, 
      &bufSize);

    // buffer allocate (no hidden cudaMalloc in the library, reuse with the same shape)
    void* dBuf; cudaMalloc(&dBuf, bufSize);

    // 4. Compute (every iteration)
    cusparseSpMV(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, // scalar a (host pointer)
      matA,   // sparse matrix descriptor
      vecX,   // dense vector x
      &beta,  // scalar b (host pointer)
      vecY,   // dense vector y (in/out)
      CUDA_R_32F, // compute type
      CUSPARSE_SPMV_ALG_DEFAULT, 
      dBuf    // workspace
      );

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));


    // 5. Cleanup (reverse order)
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle); // last - owns the stream binding

    cudaFree(d_row_ptr); cudaFree(d_col_indices); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_y);
    cudaFree(dBuf);

    return 0;
}
