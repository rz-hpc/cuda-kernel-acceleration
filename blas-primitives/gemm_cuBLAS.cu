
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cuda_fp16.h>
#include <limits>
#include <type_traits>
#include <cublas_v2.h> // compile with -lcublas

// To Solve C = alpha * A x B + beta * C
// Compare my own gemm Nsight profiling vs the cublasSgemm/cublasDgemm

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error: %d in %s at %d!\n", stat, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// GEMM

// gemm_traits

// default
template <typename T>
struct gemm_traits {
    using compute_t = T;
    using scalar_t = T;
    static constexpr float tolerance = 1e-4f;
    static constexpr bool use_tensor_cores = false;
    static constexpr const char* name = "fp32_fallback";
};

// half -- tensor cores
template <>
struct gemm_traits<half> {
    using compute_t = float;
    using scalar_t = half;
    static constexpr float tolerance = 1e-2f;
    static constexpr bool use_tensor_cores = true;
    static constexpr const char* name = "fp16_tensorcore";
};

// double
template <>
struct gemm_traits<double> {
    using compute_t = double;
    using scalar_t = double;
    static constexpr float tolerance = 1e-8f;
    static constexpr bool use_tensor_cores = false;
    static constexpr const char* name = "fp64_fallback";
};

template <int TILE_SIZE, int THREAD_TILE_SIZE, typename T>
__global__ void gemm_kernel(const T* d_A, const T* d_B, T* d_C,
                            int M, int N, int K,
                            typename gemm_traits<T>::scalar_t alpha,
                            typename gemm_traits<T>::scalar_t beta) {
      using compute_t = typename gemm_traits<T>::compute_t;

      compute_t alpha_c = static_cast<compute_t>(alpha);
      compute_t beta_c = static_cast<compute_t>(beta);

      // shared memory tiles using T (the I/O should be using T)
      __shared__ T tile_A[TILE_SIZE][TILE_SIZE + 1];
      __shared__ T tile_B[TILE_SIZE][TILE_SIZE + 1];

      // register tile accum using compute_t, initial with compute_t(0)
      compute_t accum[THREAD_TILE_SIZE][THREAD_TILE_SIZE];
      #pragma unroll
      for (int i = 0; i < THREAD_TILE_SIZE; i++) {
          for (int j = 0; j < THREAD_TILE_SIZE; j++) {
              accum[i][j] = compute_t(0);
          }
      }

      // index
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int global_row_start = blockIdx.y * TILE_SIZE + ty * THREAD_TILE_SIZE;
      int global_col_start = blockIdx.x * TILE_SIZE + tx * THREAD_TILE_SIZE;

      // for each tile phase in K dimension
      // A: phase K horizontally for columns 
      // B: phase K vertically for rows
      for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ph++) {

          // load from global memory to shared tiles (strided)
          for (int i = 0; i < TILE_SIZE; i += TILE_SIZE / THREAD_TILE_SIZE) {
              for (int j = 0; j < TILE_SIZE; j += TILE_SIZE / THREAD_TILE_SIZE) {
                  int shared_row = ty + i;
                  int shared_col = tx + j;

                  // load A
                  int global_A_row = blockIdx.y * TILE_SIZE + shared_row;
                  int global_A_col = ph * TILE_SIZE + shared_col;
                  if (global_A_row < M && global_A_col < K) {
                      tile_A[shared_row][shared_col] = d_A[global_A_row * K + global_A_col];
                  }
                  else {
                      tile_A[shared_row][shared_col] = T(0);
                  }

                  // load B
                  int global_B_row = ph * TILE_SIZE + shared_row;
                  int global_B_col = blockIdx.x * TILE_SIZE + shared_col;
                  if (global_B_row < K && global_B_col < N) {
                      tile_B[shared_row][shared_col] = d_B[global_B_row * N + global_B_col];
                  }
                  else {
                      tile_B[shared_row][shared_col] = T(0);
                  }
              }
          }
          __syncthreads();

          // Dot product
          // for instruction in the phase K tile
          // outer product (fixed k)
          for (int k_inst = 0; k_inst < TILE_SIZE; k_inst++) {

              // register tiles for fragA and fragB
              compute_t fragA[THREAD_TILE_SIZE];
              compute_t fragB[THREAD_TILE_SIZE];

              // load from tiles to the register tiles
              #pragma unroll
              for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                  fragA[i] = static_cast<compute_t>(tile_A[ty * THREAD_TILE_SIZE + i][k_inst]);
              }
              #pragma unroll
              for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                  fragB[j] = static_cast<compute_t>(tile_B[k_inst][tx * THREAD_TILE_SIZE + j]);
              }

              // outer product
              #pragma unroll
              for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                    accum[i][j] += fragA[i] * fragB[j];
                }
              }
          }
          __syncthreads();
      }

      // compute the alpha * AxB + beta * C, and write from register accum to the global memory
      for (int i = 0; i < THREAD_TILE_SIZE; i++) {
          for (int j = 0; j < THREAD_TILE_SIZE; j++) {
              int global_row_C = global_row_start + i;
              int global_col_C = global_col_start + j;
              if (global_row_C < M && global_col_C < N) {
                  int global_idx = global_row_C * N + global_col_C;
                  compute_t prev_C = static_cast<compute_t>(d_C[global_idx]);
                  compute_t result = alpha_c * accum[i][j] + beta_c * prev_C;
		  d_C[global_idx] = static_cast<T>(result);
              }
          }
      }
}

// launch gemm kernel
template <typename T>
__host__ void launch_gemm(const T* h_A, const T* h_B, T* h_C,
                          int M, int N, int K,
                          typename gemm_traits<T>::scalar_t alpha,
                          typename gemm_traits<T>::scalar_t beta) {
    // each block has TILE_SIZE / THREAD_TILE_SIZE * TILE_SIZE / THREAD_TILE_SIZE threads
    // for performance, this number of threads should be at least 32 (for a warp)
    // In first try, the TILE_SIZE was 32 and THREAD_TILE_SIZE was 8
    // which made the threadsPerBlock only (32/8) * (32/8) = 16, which was less than 32 (a warp)
    // this gemm kernel performance was 1.7x slower than the cuBLAS
    // With current setting, (32 / 4) * (32 / 4) = 8 * 8 = 64 threads
    // each block would need (32 * 33 * 2) * sizeof(T) shared memory L2 (check the limit T4 has 64kb L2)
    const int TILE_SIZE = 32;
    const int THREAD_TILE_SIZE = 4;

    const int SIZE_A = M * K * sizeof(T);
    const int SIZE_B = K * N * sizeof(T);
    const int SIZE_C = M * N * sizeof(T);

    T *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, SIZE_A));
    CHECK_CUDA(cudaMalloc((void**)&d_B, SIZE_B));
    CHECK_CUDA(cudaMalloc((void**)&d_C, SIZE_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, SIZE_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, SIZE_B, cudaMemcpyHostToDevice));

    if (beta != static_cast<typename gemm_traits<T>::scalar_t>(0)) {
        CHECK_CUDA(cudaMemcpy(d_C, h_C, SIZE_C, cudaMemcpyHostToDevice));
    }

    // launch the gemm kernel
    dim3 threadsPerBlock(TILE_SIZE / THREAD_TILE_SIZE, TILE_SIZE / THREAD_TILE_SIZE, 1);
    dim3 blocksPerGrid( (N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);

    gemm_kernel<TILE_SIZE, THREAD_TILE_SIZE, T><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);    

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, SIZE_C, cudaMemcpyDeviceToHost));

    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
}

// if constexpr dispatch
template <typename T>
__host__ void gemm_dispatcher(const T* h_A, const T* h_B, T* h_C,
                              int M, int N, int K,
                              typename gemm_traits<T>::scalar_t alpha,
                              typename gemm_traits<T>::scalar_t beta) {
      using traits = gemm_traits<T>;

      if constexpr (traits::use_tensor_cores) {
          //launch_gemm_tensorcore<T, typename traits::compute_t>(A, B, C, M, N, K, alpha, beta);
      }
      else {
          launch_gemm<T>(h_A, h_B, h_C, M, N, K, alpha, beta);
      }
}

// launch cuBLAS
template <typename T>
__host__ void launch_cublas_gemm(const T* h_A, const T* h_B, T* h_C,
                                int M, int N, int K,
                                typename gemm_traits<T>::scalar_t alpha,
                                typename gemm_traits<T>::scalar_t beta) {
    const int SIZE_A = M * K * sizeof(T);
    const int SIZE_B = K * N * sizeof(T);
    const int SIZE_C = M * N * sizeof(T);

    T *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, SIZE_A));
    CHECK_CUDA(cudaMalloc((void**)&d_B, SIZE_B));
    CHECK_CUDA(cudaMalloc((void**)&d_C, SIZE_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, SIZE_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, SIZE_B, cudaMemcpyHostToDevice));

    if (beta != static_cast<typename gemm_traits<T>::scalar_t>(0)) {
        CHECK_CUDA(cudaMemcpy(d_C, h_C, SIZE_C, cudaMemcpyHostToDevice));
    }

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    using compute_t = typename gemm_traits<T>::compute_t;
    compute_t alpha_c = static_cast<compute_t>(alpha);
    compute_t beta_c = static_cast<compute_t>(beta);

    // Lamda wrapper for cuBLAS call (dispatcher for different T)
    // translates Row-Major C = A * B to column major C^T = B^T * A^T
    auto call_cublas = [&]() {
        if constexpr (std::is_same_v<T, float>) {
            CHECK_CUBLAS(cublasSgemm(handle, 
                                    CUBLAS_OP_N, // no transpose operation for first matrix (memory trick does it)
                                    CUBLAS_OP_N, // no transpose operation for second matrix (memory trick does it)
                                    N, M, K, // transpose: N, M, K instead of M, N, K
                                    &alpha_c, // pointer to scalar
                                    d_B,
                                    N, // leading dimension of the first matrix to multiply (rows) B^T
                                    d_A,
                                    K, // leading dimension of the second matrix to multiply (rows) A^T
                                    &beta_c,
                                    d_C,
                                    N)); // leading dimension of the output matrix (rows) C^T
        }
        else if constexpr (std::is_same_v<T, double>) {
            CHECK_CUBLAS(cublasDgemm(handle,
                                    CUBLAS_OP_N, 
                                    CUBLAS_OP_N,
                                    N, M, K, // transpose: N, M, K instead of M, N, K
                                    &alpha_c,
                                    d_B,
                                    N,
                                    d_A,
                                    K,
                                    &beta_c,
                                    d_C,
                                    N));

        }
    };

    call_cublas();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, SIZE_C, cudaMemcpyDeviceToHost));

    CHECK_CUBLAS(cublasDestroy(handle));
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
}


// verify 
template <typename T>
bool verify_gemm(const T* result, const T* reference, const int size) {
    using traits = gemm_traits<T>;
    float max_ref_err = 0.0f;

    for (int i = 0; i < size; i++) {
        float res = static_cast<float>(result[i]);
        float ref = static_cast<float>(reference[i]);
        float ref_err = std::abs(res - ref) / (std::abs(ref) + 1e-8f);
        max_ref_err = std::max(max_ref_err, ref_err);
    }

    printf("[%s] max relative error: %e (tolerence %e)",
        traits::name, max_ref_err, traits::tolerance);

    return max_ref_err < traits::tolerance;
}

int main() {
    int M = 1000;
    int N = 1000;
    int K = 500;

    // float path
    {
        int sizeA = M * K * sizeof(float);
        int sizeB = K * N * sizeof(float);
        int sizeC = M * N * sizeof(float);

        float *h_A, *h_B, *h_C, *h_C_cublas, *h_C_ref;
        h_A = (float*)malloc(sizeA);
        h_B = (float*)malloc(sizeB);
        h_C = (float*)malloc(sizeC);
        h_C_cublas = (float*)malloc(sizeC);
        h_C_ref = (float*)malloc(sizeC);

        for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
        for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;
        for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
        for (int i = 0; i < M * N; i++) h_C_ref[i] = 1.0f * 2.0f * K;

        gemm_dispatcher<float>(h_A, h_B, h_C, M, N, K, 1.0, 0.0);

        bool isSuccess = verify_gemm<float>(h_C, h_C_ref, M * N);
        std::cout << "float gemm verified: " << isSuccess << std::endl;

        launch_cublas_gemm<float>(h_A, h_B, h_C_cublas, M, N, K, 1.0, 0.0);

        bool iscublasSuccess = verify_gemm<float>(h_C_cublas, h_C_ref, M * N);
        std::cout << "float cublas gemm verified: " << iscublasSuccess << std::endl;

        free(h_C_ref);
        free(h_C_cublas);
        free(h_C);
        free(h_B);
        free(h_A);
    }

    // double path
    {
        int sizeA = M * K * sizeof(double);
        int sizeB = K * N * sizeof(double);
        int sizeC = M * N * sizeof(double);

        double *h_A, *h_B, *h_C, *h_C_cublas, *h_C_ref;
        h_A = (double*)malloc(sizeA);
        h_B = (double*)malloc(sizeB);
        h_C = (double*)malloc(sizeC);
        h_C_cublas = (double*)malloc(sizeC);
        h_C_ref = (double*)malloc(sizeC);

        for (int i = 0; i < M * K; i++) h_A[i] = 1.0;
        for (int i = 0; i < K * N; i++) h_B[i] = 2.0;
        for (int i = 0; i < M * N; i++) h_C[i] = 0.0;
        for (int i = 0; i < M * N; i++) h_C_cublas[i] = 0.0;
        for (int i = 0; i < M * N; i++) h_C_ref[i] = 1.0 * 2.0 * K;

        gemm_dispatcher<double>(h_A, h_B, h_C, M, N, K, 1.0, 0.0);

        bool isSuccess = verify_gemm<double>(h_C, h_C_ref, M * N);
        std::cout << "double gemm verified: " << isSuccess << std::endl;

        launch_cublas_gemm<double>(h_A, h_B, h_C_cublas, M, N, K, 1.0, 0.0);

        bool iscublasSuccess = verify_gemm<double>(h_C_cublas, h_C_ref, M * N);
        std::cout << "double cublas gemm verified: " << iscublasSuccess << std::endl;

        free(h_C_ref);
        free(h_C_cublas);
        free(h_C);
        free(h_B);
        free(h_A);
    }

}
