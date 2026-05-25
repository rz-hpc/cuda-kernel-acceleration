
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <vector>
#include <iostream>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// AXPY: vector addition/scaling to update the guess
// y = y + alpha * x
__global__ void axpy_kernel(const int n, const float alpha, const float* d_x, float* d_y) {
    int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx < n) {
        d_y[row_idx] += d_x[row_idx] * alpha;
    }
}

// Comimed x and r update together to save memory trip
// x += x * p, r -= r * q
__global__ void update_x_r_kernel(int n, float alpha,
                                  float* d_x, const float* d_p,
                                  float* d_r, const float* d_q) {
    int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx < n) {
        d_x[row_idx] += alpha * d_p[row_idx];
        d_r[row_idx] -= alpha * d_q[row_idx];
    }
}

// New direction is old error + current path
// p = r + b * p
__global__ void update_p_kernel(const int n, const float beta,
                                const float* r, float* p) {
      int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

      if (row_idx < n) {
          p[row_idx] = r[row_idx] + beta * p[row_idx];
      }
}

__global__ void spmv_csr_vector_kernel(
    const int num_rows,
    const int* row_ptr,
    const int* col_indices,
    const float* values,
    const float* x,
    float* y) {

      // One row per block (32 threads)
      int row_idx = (blockDim.x * blockIdx.x + threadIdx.x) / 32;

      // thread id
      int lane = threadIdx.x % 32;

      if (row_idx < num_rows) {

            int row_start = row_ptr[row_idx];
            int row_end = row_ptr[row_idx + 1];
            float product = 0.0f;

            // Memory coalesced
            // thread 0 works on 0, 32, 64 ...
            // thread 1 works on 1, 33, 65 ...
            for (int i = row_start + lane; i < row_end; i += 32) {
                int col_idx = col_indices[i];
                product += x[col_idx] * values[i];
            }

            // Warp shuffle
            for (int offset = 16; offset > 0; offset >>= 1) {
                product += __shfl_down_sync(0xffffffff, product, offset);
            }

            // one thread to assign to the final result
            if (lane == 0) {
                y[row_idx] = product;
            }
      }
}

// CSR: d_ptr row pointers, d_cols column indices, d_vals values
// n total rows of the CSR matrix (A)
// d_b: right hand side vector
// d_x: solution vector
float solveCG(int n, int* d_ptr, int* d_cols, float* d_vals, float* d_b, float* d_x, const int max_iter, const float tolerance) {
    // 1. Initial Residual: r = b - Ax (Assume x=0, so r=b)
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(float)));

    float *d_r;
    CHECK_CUDA(cudaMalloc((void**)&d_r, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice));

    // 2. Initial Direction: p = r
    float *d_p;
    CHECK_CUDA(cudaMalloc((void**)&d_p, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(float), cudaMemcpyDeviceToDevice));

    // 3. Initial Result of spMV: q = A * p
    float *d_q;
    CHECK_CUDA(cudaMalloc((void**)&d_q, n * sizeof(float)));

    // 4. Warp pointers to thrust
    thrust::device_ptr<float> r_ptr = thrust::device_pointer_cast(d_r);
    thrust::device_ptr<float> p_ptr = thrust::device_pointer_cast(d_p);
    thrust::device_ptr<float> q_ptr = thrust::device_pointer_cast(d_q);

    // 5. Initial error rho = r * r
    // thrust::inner_product // Compute dot product: sum(A[i] * B[i])
    //int result = thrust::inner_product(
    //    A.begin(), A.end(),  // First range
    //    B.begin(),           // Second range
    //    0                    // Initial value
    //);
    float r_dot_r = thrust::inner_product(r_ptr, r_ptr + n, r_ptr, 0.0f);

    // 5. Loop:
    for(int k=0; k < max_iter; k++) {

        // q = A * p (YOUR VECTOR KERNEL)
        int threads_vector = 128; // 4 warps per block
        int rows_per_block = threads_vector / 32;
        int blocks_vector = (n + rows_per_block - 1) / rows_per_block;
        spmv_csr_vector_kernel<<<blocks_vector, threads_vector>>>(n, d_ptr, d_cols, d_vals, d_p, d_q);

        // alpha = (r_dot_r) / (p_dot_q)
        float p_dot_q = thrust::inner_product(p_ptr, p_ptr + n, q_ptr, 0.0f);
        float alpha = r_dot_r / p_dot_q;

        // Update x and r
        int threads_update = 256;
        int blocks_update = (n + threads_update - 1) / threads_update;
        update_x_r_kernel<<<blocks_update, threads_update>>>(n, alpha, d_x, d_p, d_r, d_q);

        // Check convergence...
        float new_r_dot_r = thrust::inner_product(r_ptr, r_ptr + n, r_ptr, 0.0f);
        if (sqrt(new_r_dot_r) < tolerance) break;

        // Update p (Search Direction)
        float beta = new_r_dot_r / r_dot_r;
        update_p_kernel<<<blocks_update, threads_update>>>(n, beta, d_r, d_p);

        r_dot_r = new_r_dot_r;

        if (k == 0) {
            float h_alpha;
            h_alpha = alpha; // This triggers a sync
            std::cout << "Iteration 0: alpha = " << h_alpha << " r_dot_r = " << r_dot_r << std::endl;
        }
    }

    // Cleanup workspace
    cudaFree(d_r); cudaFree(d_p); cudaFree(d_q);
    return sqrt(r_dot_r); // Return final residual
}

// Generator for a 1D Poisson Matrix (Tridiagonal)
// Represents the equation: -u'' = f
void generatePoisson1D(int n, std::vector<int>& row_ptr, std::vector<int>& col_indices, std::vector<float>& values, std::vector<float>& b) {
    row_ptr.clear();
    col_indices.clear();
    values.clear();
    b.assign(n, 1.0f); // Source term (e.g., uniform heat)

    int nnz = 0;
    row_ptr.push_back(0);

    for (int i = 0; i < n; i++) {
        // Left neighbor
        if (i > 0) {
            col_indices.push_back(i - 1);
            values.push_back(-1.0f);
            nnz++;
        }
        // Center (Self)
        col_indices.push_back(i);
        values.push_back(2.0f);
        nnz++;
        // Right neighbor
        if (i < n - 1) {
            col_indices.push_back(i + 1);
            values.push_back(-1.0f);
            nnz++;
        }
        row_ptr.push_back(nnz);
    }
}

int main() {
    int n = 1000; // Let's solve a 1000-node physics problem!
    int max_iter = 2000;
    float tolerance = 1e-6f;

    // 1. Generate the problem on Host
    std::vector<int> h_ptr, h_cols;
    std::vector<float> h_vals, h_b;
    generatePoisson1D(n, h_ptr, h_cols, h_vals, h_b);

    // 2. Prepare Device Memory
    int *d_ptr, *d_cols;
    float *d_vals, *d_b, *d_x;
    int nnz = h_vals.size();

    cudaMalloc(&d_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_cols, nnz * sizeof(int));
    cudaMalloc(&d_vals, nnz * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));

    cudaMemcpy(d_ptr, h_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, h_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Solve!
    std::cout << "Starting CG Solver for N=" << n << "..." << std::endl;
    float final_res = solveCG(n, d_ptr, d_cols, d_vals, d_b, d_x, max_iter, tolerance);

    // 4. Verification
    std::vector<float> h_x(n);
    cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Solver finished. Final Residual: " << final_res << std::endl;
    std::cout << "Mid-point solution value (u[500]): " << h_x[500] << std::endl;

    // Cleanup
    cudaFree(d_ptr); cudaFree(d_cols); cudaFree(d_vals); cudaFree(d_b); cudaFree(d_x);
    return 0;
}
