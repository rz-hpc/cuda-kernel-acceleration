
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Paddings
#define SHMEM_INDEX(i) (((i) + ((i) >> 5)))

// Scan of Scans
// Kernel 1: Brent-Kung with writing block sum to the global memory array
__global__ void scan_and_export_sums(float* d_in, float* d_out, float* d_block_sums, int n) {

    // shared memory
    // shared memory size: (blockDim.x  + blockDim.x / 32) * sizeof(float)
    extern __shared__ float temp[];

    int tx = threadIdx.x;
    int b_size = blockDim.x;
    int global_idx = b_size * blockIdx.x + tx;

    // load input
    if (global_idx < n) {
        temp[SHMEM_INDEX(tx)] = d_in[global_idx];
    }
    else {
        temp[SHMEM_INDEX(tx)] = 0.0f;
    }

    __syncthreads();

    // up-sweep-- XY[i] += XY[i - stride]
    // stride from 1 to b_size / 2
    // element i: 2n - 1, 4n - 1, ...
    // continous threads to element mapping:
    // index = (tx + 1) * 2 * stride - 1
    for (int stride = 1; stride < b_size; stride <<= 1) {
        // save power by preventing idle threads calculating index
        int num_active_threads = b_size / (2 * stride);
        if (tx < num_active_threads) {
            int index = (tx + 1) * 2 * stride - 1;
            if (index < b_size) {
                temp[SHMEM_INDEX(index)] += temp[SHMEM_INDEX(index - stride)];
            }
        }
        __syncthreads();
    }

    // last thread in the block writes to the block sums array
    if (tx == b_size - 1) {
        d_block_sums[blockIdx.x] = temp[SHMEM_INDEX(b_size - 1)];
    }

    // down-sweep-- XY[i + stride] += XY[i]
    // stride from b_size / 4 to 1
    // continous threads to element mappings:
    // index = (tx + 1) * 2 * stride - 1
    for (int stride = b_size / 4; stride > 0; stride >>= 1) {
        int num_active_threads = b_size / (2 * stride);
        if (tx < num_active_threads) {
            int index = (tx + 1) * 2 * stride - 1;
            if (index + stride < b_size) {
                temp[SHMEM_INDEX(index + stride)] += temp[SHMEM_INDEX(index)];
            }
        }
        __syncthreads();
    }

    // Write to output
    if (global_idx < n) {
        d_out[global_idx] = temp[SHMEM_INDEX(tx)];
    }

}


// Kernel 2: normal brent-kung scan (no export sum to global block sum array)
__global__ void brent_kung_padded_kernel(float* d_in, float* d_out, int n) {

    // shared memory
    // shared memory size: (blockDim.x  + blockDim.x / 32) * sizeof(float)
    extern __shared__ float temp[];

    int tx = threadIdx.x;
    int b_size = blockDim.x;
    int global_idx = b_size * blockIdx.x + tx;

    // load input
    if (global_idx < n) {
        temp[SHMEM_INDEX(tx)] = d_in[global_idx];
    }
    else {
        temp[SHMEM_INDEX(tx)] = 0.0f;
    }

    __syncthreads();

    // up-sweep-- XY[i] += XY[i - stride]
    // stride from 1 to b_size / 2
    // element i: 2n - 1, 4n - 1, ...
    // continous threads to element mapping:
    // index = (tx + 1) * 2 * stride - 1
    for (int stride = 1; stride < b_size; stride <<= 1) {
        // save power by preventing idle threads calculating index
        int num_active_threads = b_size / (2 * stride);
        if (tx < num_active_threads) {
            int index = (tx + 1) * 2 * stride - 1;
            if (index < b_size) {
                temp[SHMEM_INDEX(index)] += temp[SHMEM_INDEX(index - stride)];
            }
        }
        __syncthreads();
    }

    // down-sweep-- XY[i + stride] += XY[i]
    // stride from b_size / 4 to 1
    // continous threads to element mappings:
    // index = (tx + 1) * 2 * stride - 1
    for (int stride = b_size / 4; stride > 0; stride >>= 1) {
        int num_active_threads = b_size / (2 * stride);
        if (tx < num_active_threads) {
            int index = (tx + 1) * 2 * stride - 1;
            if (index + stride < b_size) {
                temp[SHMEM_INDEX(index + stride)] += temp[SHMEM_INDEX(index)];
            }
        }
        __syncthreads();
    }

    // Write to output
    if (global_idx < n) {
        d_out[global_idx] = temp[SHMEM_INDEX(tx)];
    }
}

// Kernel 3: Get the offset from the global memory block sums, and add offset to each thread in each block
__global__ void add_block_offsets(float* d_out, float* d_block_sums_scanned, int n) {
      int tx = threadIdx.x;
      int b_size = blockDim.x;
      int global_idx = b_size * blockIdx.x + tx;

      if (blockIdx.x == 0)
          return;

      float offset = d_block_sums_scanned[blockIdx.x - 1];

      if (global_idx < n) {
          d_out[global_idx] += offset;
      }
}

// Host code -- orchestration the Kernels
// Assume each blocks deals 1024 elements and gets 1 sum
// For array of size N
// Level 0: memory N (original input)
// Level 1: memory N / 1024 (block sums of level 0)
// Level 2: memory N / (1024 * 1024) (block sum of level 1)
__host__ void run_hierarchical_scan(float* h_in, float* h_out, int n) {

    int sizeN = n * sizeof(float);
    const int threadsPerBlock = 1024;

    // Shared memory size is blockDim.x + padding (/ 32)
    const int shmemSize = sizeof(float) * (threadsPerBlock + (threadsPerBlock >> 5));

    float *d_in, *d_out;

    CHECK_CUDA(cudaMalloc((void**)&d_in, sizeN));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeN));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeN, cudaMemcpyHostToDevice));


    // level 1 block sums memory
    int blocksPerGridLevel1 = (n + threadsPerBlock - 1) / threadsPerBlock;
    float *d_block_sums1;
    CHECK_CUDA(cudaMalloc((void**)&d_block_sums1, blocksPerGridLevel1 * sizeof(float)));

    // level 2 block sums memory
    int blocksPerGridLevel2 = (blocksPerGridLevel1 + threadsPerBlock - 1) / threadsPerBlock;
    float *d_block_sums2;
    CHECK_CUDA(cudaMalloc((void**)&d_block_sums2, blocksPerGridLevel2 * sizeof(float)));

    // Kernel 1-- scan and export
    // need to call twice: first time for level 0 to level 1 (each block)
    // second time for level 1 to level 2 (block sum scan)
    scan_and_export_sums<<<blocksPerGridLevel1, threadsPerBlock, shmemSize>>>(d_in, d_out, d_block_sums1, n);

    scan_and_export_sums<<<blocksPerGridLevel2, threadsPerBlock, shmemSize>>>(d_block_sums1, d_block_sums1, d_block_sums2, blocksPerGridLevel1);

    // Kernel 2
    brent_kung_padded_kernel<<<1, threadsPerBlock, shmemSize>>>(d_block_sums2, d_block_sums2, blocksPerGridLevel2);

    // Kernel 3-- add the offset
    // Also need two steps: 1. level 2 scanned sum to level 1 scanned sum
    // 2. level 1 scanned sum to level 0 each block each element
    add_block_offsets<<<blocksPerGridLevel2, threadsPerBlock>>>(d_block_sums1, d_block_sums2, blocksPerGridLevel1);

    add_block_offsets<<<blocksPerGridLevel1, threadsPerBlock>>>(d_out, d_block_sums1, n);

    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeN, cudaMemcpyDeviceToHost));

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_block_sums1); cudaFree(d_block_sums2);
}

int main() {
    // 1. Setup a large input (e.g., 2 million elements)
    // This forces our code to use all 3 levels of the hierarchy
    const int N = 2000000;
    size_t size = N * sizeof(float);

    std::vector<float> h_in(N);
    std::vector<float> h_out(N, 0.0f);
    std::vector<float> cpu_ref(N, 0.0f);

    // Initialize with a simple value (e.g., 1.0f)
    // The prefix sum of [1, 1, 1...] should be [1, 2, 3... N]
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
    }

    // 2. Run CPU Version for verification
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += h_in[i];
        cpu_ref[i] = sum;
    }

    printf("Starting GPU Hierarchical Scan for N = %d...\n", N);

    // 3. Call your orchestrated function
    // (Ensure run_hierarchical_scan is defined above this)
    run_hierarchical_scan(h_in.data(), h_out.data(), N);

    // 4. Verify Results
    bool match = true;
    for (int i = 0; i < N; i++) {
        // Using a small epsilon for float comparison
        if (std::abs(h_out[i] - cpu_ref[i]) > 0.1f) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_out[i], cpu_ref[i]);
            match = false;
            break;
        }
    }

    if (match) {
        printf("SUCCESS! GPU scan matches CPU reference.\n");
        printf("Last element: GPU=%f (Expected %f)\n", h_out[N-1], (float)N);
    } else {
        printf("FAILURE. Check kernel indexing.\n");
    }

    return 0;
}
