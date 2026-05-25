
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

struct GPUTimer {

    cudaEvent_t start, stop;

    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start(cudaStream_t stream = 0) {
        cudaEventRecord(start, stream);
    }

    void Stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
    }

    float Elapsed() {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

};

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

// Fused these two together
// q = A * p (1 read p, 1 write q) --> 2n
// p_dot_q (1 read p, 1 read q) --> 2n
// Reduce the total memory moves from 2n + 2n to 3n
// 1 read p, 1 write q, 1 write partial_product
__global__ void spmv_fused_scalar_kernel(const int n,
                                         const int* row_ptr,
                                         const int* col_indices,
                                         const float* values,
                                         const float* p,
                                         float* q,
                                         float* partial_product) {

      int row_idx = blockDim.x * blockIdx.x + threadIdx.x;

      if (row_idx < n) {

          // Scalar
          float sum = 0.0f;
          int row_start = row_ptr[row_idx];
          int row_end = row_ptr[row_idx + 1];
          for (int i = row_start; i < row_end; i++) {
              int col_idx = col_indices[i];
              sum += values[i] * p[col_idx];
          }

          q[row_idx] = sum;

          // Calculate the p_dot_q directly (sum is the q)
          partial_product[row_idx] = p[row_idx] * sum;
      }

}

// CSR: d_ptr row pointers, d_cols column indices, d_vals values
// n total rows of the CSR matrix (A)
// d_b: right hand side vector
// d_x: solution vector
float solveCG(int n, int nnz, int* d_ptr, int* d_cols, float* d_vals, float* d_b, float* d_x, const int max_iter, const float tolerance) {

    GPUTimer timer;
    timer.Start();

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

    // For fused scalar kernel, the moves of q are optimized
    // Instead, we need Initial d_partial_product here
    float *d_partial_product;
    CHECK_CUDA(cudaMalloc((void**)&d_partial_product, n * sizeof(float)));

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

    int actual_iters = 0; // count the executed interations for performance measure

    for(int k=0; k < max_iter; k++) {

        actual_iters++;

        /*
        // q = A * p (YOUR VECTOR KERNEL)
        int threads_vector = 128; // 4 warps per block
        int rows_per_block = threads_vector / 32;
        int blocks_vector = (n + rows_per_block - 1) / rows_per_block;
        spmv_csr_vector_kernel<<<blocks_vector, threads_vector>>>(n, d_ptr, d_cols, d_vals, d_p, d_q);

        // alpha = (r_dot_r) / (p_dot_q)
        float p_dot_q = thrust::inner_product(p_ptr, p_ptr + n, q_ptr, 0.0f);
        */

        int threads_fused_scalar = 256;
        int blocks_fused_scalar = (n + threads_fused_scalar - 1) / threads_fused_scalar;
        spmv_fused_scalar_kernel<<<blocks_fused_scalar, threads_fused_scalar>>>(n, d_ptr, d_cols, d_vals, d_p, d_q, d_partial_product);

        // This replaces the old thrust::inner_product(p, p+n, q, 0.0f)
        float p_dot_q = thrust::reduce(thrust::device_pointer_cast(d_partial_product),
                               thrust::device_pointer_cast(d_partial_product) + n, 0.0f);

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

    timer.Stop();
    float total_ms = timer.Elapsed();
    float avg_ms = total_ms/ actual_iters;

    // CALCULATION: How much data did we touch per iteration?
    // 1 SpMV (ptr, cols, vals, p, q) + 3 Dot Products + 2 Vector Updates
    // row_ptr (n + 1 rows) int
    // col_indices nnz int
    // values nnz float
    // that "10" = spMV (1 read p, 1 write q)
    //            + dotProduct (1 read p, 1 read q)
    //            + update x (1 read x, 1 read p, 1 write x)
    //            + update r (1 read r, 1 read q, 1 write r)
    //size_t byte_per_iter = (n + 1) * sizeof(int) + (nnz) * (sizeof(float) + sizeof(int)) + (n * 10 * sizeof(float));

    // Fused Version Math
    size_t byte_per_iter = (n + 1) * sizeof(int) + // row_ptr
                       (nnz) * (sizeof(float) + sizeof(int)) + // values + cols
                       (n * 9 * sizeof(float)); // We saved 1n by not re-reading q
    double gb_moved = (double)byte_per_iter * actual_iters / 1e9;
    double bandwidth = gb_moved / (total_ms / 1000.0);

    std::cout << "\n--- Performance Results ---" << std::endl;
    std::cout << "Total Time: " << total_ms << " ms" << std::endl;
    std::cout << "Avg Iter Time: " << avg_ms << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Cleanup workspace
    cudaFree(d_r); cudaFree(d_p); cudaFree(d_q);
    return sqrt(r_dot_r); // Return final residual
}

void generatePoisson2D(const int grid_size,
                       std::vector<int>& row_ptr,
                       std::vector<int>& col_indices,
                       std::vector<float>& values,
                       std::vector<float>& b) {

    int n = grid_size * grid_size; // total number of nodes

    row_ptr.clear();
    row_ptr.push_back(0);
    col_indices.clear();
    values.clear();

    b.assign(n, 1.0f);

    int nnz = 0; //current number of non-zero elements/nodes

    // on the phyical plane, there are grid_size x grid_size nodes
    // Iterating the nodes using i and j loops
    // Each node corresponding to one ROW in matrix A, since we need one equation for each node
    // i * grid_size + j is the node id, aka. row_idx in equation matrix A
    // the NWEN four neighbors ids are the col_indices
    // matrix A values are the phyical meaning like temperature, height, etc
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            int row_idx = i * grid_size + j;

            // Ordered by id ascending
            // North neighbor
            if (i > 0) {
                col_indices.push_back(row_idx - grid_size);
                values.push_back(-1.0f);
                nnz++;
            }

            // West neighbor
            if (j > 0) {
                col_indices.push_back(row_idx - 1);
                values.push_back(-1.0f);
                nnz++;
            }

            col_indices.push_back(row_idx);
            values.push_back(4.0f);
            nnz++;

            // East neighbor
            if (j < grid_size - 1) {
                col_indices.push_back(row_idx + 1);
                values.push_back(-1.0f);
                nnz++;
            }

            // South neighbor
            if (i < grid_size - 1) {
                col_indices.push_back(row_idx + grid_size);
                values.push_back(-1.0f);
                nnz++;
            }

            // Note: this has to be within the j loop
            // Because the i j loops are iterating node, one node mapping to one row in the equation matrix A
            row_ptr.push_back(nnz);

        }
    }
}

int main() {

    int grid_size = 512;//100; // This creates a 100x100 grid
    int n = grid_size * grid_size; // Total nodes = 10,000
    int max_iter = 2;// set to small number for NSight profiling 10000;
    float tolerance = 1e-7f;

    // 1. Generate the problem on Host
    std::vector<int> h_ptr, h_cols;
    std::vector<float> h_vals, h_b;

    // Generate the 2D problem
    generatePoisson2D(grid_size, h_ptr, h_cols, h_vals, h_b);

    int nnz = h_vals.size();

    // 2. Prepare Device Memory
    int *d_ptr, *d_cols;
    float *d_vals, *d_b, *d_x;

    cudaMalloc(&d_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_cols, nnz * sizeof(int));
    cudaMalloc(&d_vals, nnz * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));

    cudaMemcpy(d_ptr, h_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, h_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Solve! Call the solver
    std::cout << "Starting CG Solver for N=" << n << "..." << std::endl;
    float final_res = solveCG(n, h_cols.size(), d_ptr, d_cols, d_vals, d_b, d_x, max_iter, tolerance);

    // 4. Verification
    std::vector<float> h_x(n);
    cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Center 5x5 Heat Map:" << std::endl;
    int start = (grid_size/2 - 2);
    for (int i = start; i < start + 5; i++) {
        for (int j = start; j < start + 5; j++) {
            printf("%7.0f ", h_x[i * grid_size + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_ptr); cudaFree(d_cols); cudaFree(d_vals); cudaFree(d_b); cudaFree(d_x);
    return 0;
}
