
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>
#include <type_traits>
#include <limits>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }\
}



// Layer 1: gemm traits struct

// Traits are compile-time lookup tables keyed on a type,
// so need full tempalte specialization for later "if constexpr"
// to brach within a function body when T is known.
// Specialization picks a variant, if constexpr prunes dead branches

// Primary template -- default
template <typename T>
struct gemm_traits {
    using compute_t = T; // type used for accumulation
    using scalar_t = T;  // type used for alpha/beta
    static constexpr float tolerance = 1e-4f;
    static constexpr bool use_tensor_cores = false;
    static constexpr const char* name = "fp32_fallback";
};

// Specialization for half precision
// Compute/Accumulate in half precision overflows/loses precision fast
// so tensor-core kernels always accumulate in fp32 even when i/o are half
template <>
struct gemm_traits<half> {
  using compute_t = float; // accumulate in fp32 even though inputs are half
  using scalar_t = half;
  static constexpr float tolerance = 1e-2f; // looser -- half has ~3 decimal digits
  static constexpr bool use_tensor_cores = true;
  static constexpr const char* name = "fp16_tensorcore";
};

// Specialization for double
template <>
struct gemm_traits<double> {
    using compute_t = double;
    using scalar_t = double;
    static constexpr float tolerance = 1e-8f;
    static constexpr bool use_tensor_cores = false;
    static constexpr const char* name = "fp64_fallback";
};

// ===============================================
// Compile-time sanity checks -- catch traits mistakes before nvcc
// Run these with:
//   g++ -std=c++17 -fsyntax-only -x c++ -include cuda_fp16.h gemm_traits_template_v2.cu
// (needs cuda_fp16.h on the include path even without a GPU)
// ===============================================
static_assert(gemm_traits<float>::tolerance == 1e-4f, "float tolerance mismatch");
static_assert(gemm_traits<double>::tolerance == 1e-8f, "double tolerance mismatch");
static_assert(gemm_traits<half>::use_tensor_cores == true, "half should route to tensor cores");
static_assert(std::is_same_v<gemm_traits<half>::compute_t, float>, "half must accumulate in fp32");
static_assert(std::is_same_v<gemm_traits<double>::compute_t, double>, "double accumulates in double");


// ===============================================
// gemm kernel with shared memory and register tiles
// C = alpha * (A x B) + beta * C
// C: M x N, A: M x K, B: K x N
// ===============================================
template <int TILE_SIZE, int THREAD_TILE_SIZE, typename T>
__global__ void gemm_kernel(const T* d_A, const T* d_B, T* d_C,
                            int M, int K, int N,
                            typename gemm_traits<T>::scalar_t alpha,
                            typename gemm_traits<T>::scalar_t beta) {
    using compute_t = typename gemm_traits<T>::compute_t;

    compute_t alpha_c = static_cast<compute_t>(alpha);
    compute_t beta_c = static_cast<compute_t>(beta);

    // register tile -- accumulate using compute_t
    compute_t accum[THREAD_TILE_SIZE][THREAD_TILE_SIZE];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_SIZE; i++) {
        for (int j = 0; j < THREAD_TILE_SIZE; j++) {
            accum[i][j] = compute_t(0);
        }
    }

    // shared memory tile -- storage/bandwidth format using T
    __shared__ T tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ T tile_B[TILE_SIZE][TILE_SIZE + 1];

    // index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int global_row_start = blockIdx.y * TILE_SIZE + ty * THREAD_TILE_SIZE;
    int global_col_start = blockIdx.x * TILE_SIZE + tx * THREAD_TILE_SIZE;

    // for each phase: cross tile_A columns (horizontally), tile_B rows (vertically)
    // THREAD_STRIDE has to be blockDim.y for row and blockDim.x for column
    // since the way threadsPerBlock was defined, blockDim.y = blockDim.x = TILE_SIZE/THREAD_TILE_SIZE
    const int THREAD_STRIDE = TILE_SIZE / THREAD_TILE_SIZE;
    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ph++) {
        // load from global memory to shared memory tiles coalesced
        for (int i = 0; i < TILE_SIZE; i += THREAD_STRIDE) {
            for (int j = 0; j < TILE_SIZE; j += THREAD_STRIDE) {
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
        // for each instruction in the TILE along the k dimension
        for (int k_inst = 0; k_inst < TILE_SIZE; k_inst++) {
            // register tiles
            compute_t regA[THREAD_TILE_SIZE];
            compute_t regB[THREAD_TILE_SIZE];

            // load from shared memory to register tiles
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                regA[i] = tile_A[ty * THREAD_TILE_SIZE + i][k_inst];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                regB[j] = tile_B[k_inst][tx * THREAD_TILE_SIZE + j];
            }

            // outer product
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    // write from shared memory to global memory
    for (int i = 0; i < THREAD_TILE_SIZE; i++) {
        for (int j = 0; j < THREAD_TILE_SIZE; j++) {
            int global_row_C = global_row_start + i;
            int global_col_C = global_col_start + j;
            if (global_row_C < M && global_col_C < N) {
                int idx = global_row_C * N + global_col_C;
                compute_t c_prev = static_cast<compute_t>(d_C[idx]);
                compute_t result = alpha_c * accum[i][j] + beta_c * c_prev;
                d_C[idx] = static_cast<T>(result);
            }
        }
    }
}

// ===============================================
// templated host launcher
// ===============================================

template <typename T>
__host__ void launch_gemm_tiled(const T* h_A, const T* h_B, T* h_C,
                                int M, int K, int N,
                                typename gemm_traits<T>::scalar_t alpha,
                                typename gemm_traits<T>::scalar_t beta) {

    constexpr int TILE_SIZE = 32;
    constexpr int THREAD_TILE_SIZE = 8;

    int sizeA = M * K * sizeof(T);
    int sizeB = K * N * sizeof(T);
    int sizeC = M * N * sizeof(T);

    T *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE / THREAD_TILE_SIZE, TILE_SIZE / THREAD_TILE_SIZE, 1);
    dim3 blocksPerGrid( (N + TILE_SIZE - 1)/ TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);

    gemm_kernel<TILE_SIZE, THREAD_TILE_SIZE, T><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N, alpha, beta);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
}

// Layer 2: if constexpr dispatch inside the kernel launcher

template <typename T>
void gemm_dispatch(const T* A, const T* B, T* C,
                    int M, int K, int N,
                    typename gemm_traits<T>::scalar_t alpha,
                    typename gemm_traits<T>::scalar_t beta) {

      using traits = gemm_traits<T>;

      if constexpr (traits::use_tensor_cores) {
          // only compiles/instantiates for T = half
          // WMMA API calls (nvcuda::wmma::fragment, etc)
          //launch_gemm_tensorcore<T, typename traits::compute_t>(A, B, C, M, N. K, alpha, beta);
      }
      else {
          // tiled FP32/FP64 kernel
          launch_gemm_tiled<T>(A, B, C, M, K, N, alpha, beta);
      }
}

// Layer 3: tolerance-based verification

template <typename T>
bool verify_gemm(const T* result, const T* reference, int size) {
    using traits = gemm_traits<T>;
    float max_rel_error = 0.0f;

    for (int i = 0; i < size; i++) {
        float ref = static_cast<float>(reference[i]);
        float res = static_cast<float>(result[i]);
        float rel_error = std::abs(ref - res) / (std::abs(ref) + 1e-8f);
        max_rel_error = std::max(max_rel_error, rel_error);
    }

    printf("[%s] max relative error: %e (tolerance: %e)\n",
          traits::name, max_rel_error, traits::tolerance);

    return max_rel_error < traits::tolerance;
}

int main() {
    int M = 1000, K = 500, N = 1000;

    // float path
    {
        int sizeA = M * K * sizeof(float);
        int sizeB = K * N * sizeof(float);
        int sizeC = M * N * sizeof(float);

        float* h_A = (float*)malloc(sizeA);
        float* h_B = (float*)malloc(sizeB);
        float* h_C = (float*)malloc(sizeC);
        float* h_ref = (float*)malloc(sizeC);

        for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
        for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;
        for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
        for (int i = 0; i < M * N; i++) h_ref[i] = 1.0f * 2.0f * K;

        gemm_dispatch<float>(h_A, h_B, h_C, M, K, N, 1.0f, 0.0f);
        bool isSuccess = verify_gemm<float>(h_C, h_ref, M * N);
        std::cout << "float gemm_dispath correctness: " << isSuccess << std::endl;
        
        free(h_A); free(h_B); free(h_C); free(h_ref);
    }

    // double path
    {
        int sizeA = M * K * sizeof(double);
        int sizeB = K * N * sizeof(double);
        int sizeC = M * N * sizeof(double);

        double *h_A = (double*)malloc(sizeA);
        double *h_B = (double*)malloc(sizeB);
        double *h_C = (double*)malloc(sizeC);
        double *h_ref = (double*)malloc(sizeC);

        for (int i = 0; i < M * K; i++) h_A[i] = 1.0;
        for (int i = 0; i < K * N; i++) h_B[i] = 2.0;
        for (int i = 0; i < M * N; i++) h_C[i] = 0.0;
        for (int i = 0; i < M * N; i++) h_ref[i] = 1.0 * 2.0 * K;

        gemm_dispatch<double>(h_A, h_B, h_C, M, K, N, 1.0, 0.0);
        bool isSuccess = verify_gemm(h_C, h_ref, M * N);
        std::cout << "double gemm_dispatch correctness: " << isSuccess << std::endl;

        free(h_A); free(h_B); free(h_C); free(h_ref);
    }

    // half path -- compile error from the static_assert right now

    return 0;
}



