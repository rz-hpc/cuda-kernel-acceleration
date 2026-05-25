
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define TILE_WIDTH 16

// Matrix M dimension m x k
// Matrix N dimension k * n
// Multiplication Result Matrix P dimension m x n
__global__ void gemm_kernel(const float* d_M, const float* d_N, float* d_P, int m, int k, int n) {

    __shared__ float s_m[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_n[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row and column index in the global input matrices
    int row_idx = blockIdx.y * blockDim.y + ty;
    int col_idx = blockIdx.x * blockDim.x + tx;

    float PValue = 0.0f;

    // Iterating TILE_WIDTH phases through the row (column elements)
    // Reuse the same TILE slots
    // Why iteration: For the PValue of element (0, 0), we need the whole row 0 in M and whole column 0 in N
    for (int ph = 0; ph < (k + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {

        // Load M row row_idx
        int m_col_idx = ph * TILE_WIDTH + tx;
        if (row_idx < m && m_col_idx < k) {
            s_m[ty][tx] = d_M[ row_idx * k + m_col_idx];
        }
        else {
            s_m[ty][tx] = 0.0f;
        }

        // Load N col col_idx
        int n_row_idx = ph * TILE_WIDTH + ty;
        if (n_row_idx < k && col_idx < n) {
            s_n[ty][tx] = d_N[ n_row_idx * n + col_idx];
        }
        else {
            s_n[ty][tx] = 0.0f;
        }

        // Barrier to make sure all the threads finish loading m and n values
        __syncthreads();

        // Calculate P in the tile
        for (int pi = 0; pi < TILE_WIDTH; pi++) {
            PValue += s_m[ty][pi] * s_n[pi][tx];
        }

        // Barrier to make sure PValue is calculated for all the elements in the tile
        __syncthreads();
    }

    // Update the output
    if (row_idx < m && col_idx < n) {
        d_P[row_idx * n + col_idx] = PValue;
    }

}

template<typename T, int TILE_SIZE>
__global__ void gemm_templated_kernel(const T* d_M, const T* d_N, T* d_P, int m, int k, int n) {

    __shared__ T s_m[TILE_SIZE][TILE_SIZE];
    __shared__ T s_n[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // global matrices index
    int row_idx = blockDim.y * blockIdx.y + ty;
    int col_idx = blockDim.x * blockIdx.x + tx;

    T PValue = T(0); // or static_cast<T>(0);

    // Load corresponding M row and N column into shared memory TILE and compute the PValue phase by phase
    for (int ph = 0; ph < (k + TILE_SIZE - 1) / TILE_SIZE; ph++) {

        // M row_idx corresponding column
        int m_col_idx = ph * TILE_SIZE + tx;
        if (row_idx < m && m_col_idx < k) {
            s_m[ty][tx] = d_M[row_idx * k + m_col_idx];
        }
        else {
            s_m[ty][tx] = T(0);
        }

        // N col_idx corresponding row
        int n_row_idx = ph * TILE_SIZE + ty;
        if (n_row_idx < k && col_idx < n) {
            s_n[ty][tx] = d_N[ n_row_idx * n + col_idx];
        }
        else {
            s_n[ty][tx] = T(0);
        }

        // wait for all threads finish loading
        __syncthreads();

        // Caluclate PValue in this tile
        for (int pi = 0; pi < TILE_SIZE; pi++) {
            PValue += s_m[ty][pi] * s_n[pi][tx];
        }

        // wait for all threads finish calucalting
        __syncthreads();
    }

    // Assign final result to d_P
    if (row_idx < m && col_idx < n) {
        d_P[ row_idx * n + col_idx ] = PValue;
    }

}

// RAII: Resource Acquisition Is Initilization
template <typename T>
class CudaBuffer {
private:
    T* d_ptr = nullptr;
    size_t size_bytes;

public:
    CudaBuffer(size_t n) : size_bytes(n * sizeof(T)) {
        // The HPC Approach: Usually, we throw a std::runtime_error.
        // This allows the higher-level application to catch the error,
        // log it, and shut down gracefully.
        // (Instead of exit())
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_ptr), size_bytes));
    }

    ~CudaBuffer() {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }

    T* get() { return d_ptr; }
    const T* get() const { return d_ptr; }

    // Move constructors are allowed
    CudaBuffer(CudaBuffer&& other) noexcept : d_ptr(other.d_ptr), size_bytes(other.size_bytes) {
        other.d_ptr = nullptr;
    }

    // Disable copying to avoid double free
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

};

template<typename T, int TILE_SIZE>
__host__ void launch_gemm_templated(const T* d_M, const T* d_N, T* d_P, int m, int k, int n, cudaStream_t stream = 0) {

    dim3 ThreadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 BlocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    gemm_templated_kernel<T, TILE_SIZE><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(
      d_M, d_N, d_P, m, k, n
      );

    CHECK_CUDA(cudaGetLastError());

}

int main() {

    using Precision = float; // change to double etc should work directly
    const int TS = 16;

    int M = 1024, N = 1024, K = 1024;

    // RAII Buffers
    CudaBuffer<Precision> d_M(M * K);
    CudaBuffer<Precision> d_N(K * N);
    CudaBuffer<Precision> d_P(M * N);

    // Intial global matrices
    Precision *h_M, *h_N, *h_P;
    h_M = (Precision*)malloc(M * K * sizeof(Precision));
    h_N = (Precision*)malloc(K * N * sizeof(Precision));
    h_P = (Precision*)malloc(M * N * sizeof(Precision));

    for (int i = 0; i < M * K; i++){
        h_M[i] = Precision(1);
    }

    for (int j = 0; j < K * N; j++) {
        h_N[j] = Precision(2);
    }

    // Stream load data and launch GEMM kernel
    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    CHECK_CUDA(cudaMemcpyAsync(d_M.get(), h_M, M * K * sizeof(Precision), cudaMemcpyHostToDevice, compute_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_N.get(), h_N, K * N * sizeof(Precision), cudaMemcpyHostToDevice, compute_stream));

    launch_gemm_templated<Precision, TS>(d_M.get(), d_N.get(), d_P.get(), M, K, N, compute_stream);

    CHECK_CUDA(cudaMemcpyAsync(h_P, d_P.get(), M * N * sizeof(Precision), cudaMemcpyDeviceToHost, compute_stream));

    CHECK_CUDA(cudaStreamSynchronize(compute_stream));

    bool isSuccess = true;
    for (int p = 0; p < M * N; p++) {
        if (h_P[p] != Precision(2048)){
            std::cout<< "Error in P row " << p / N << " col " << p % N << std::endl;
            isSuccess = false;
            break;
        }
    }

    std::cout << "Verification isSuccess: " << isSuccess << std::endl;

    free(h_M); free(h_N); free(h_P);
    cudaStreamDestroy(compute_stream);

    return 0;
}


