
#include <iostream>
#include <cuda_runtime.h>

__global__ void transposeKernelNative(float* in, float* out, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n)
    {
        out[col * n + row] = in[row * n + col];
    }
}

#define TILE_WIDTH 32

__global__ void transposeKernelTiled(float* in, float* out, int n) {
    // Shared 2D "scratchpad", assessible by all threads in this block
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1];

    // Calculate the global coordinate of the input matrix
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    // Load the tile from global memory to shared memory
    // Coalesced read-- thread in a warp access adjacent floats
    if (x < n && y < n) {
        // Row-major (C++ memory), maps to 2D grid, the x represents horizontal grid (column), while y represents the vertical grid (row)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }

    __syncthreads();

    // Recalculate and transpose the x y for output matrix
    // Note that threadIdx stays and not swap
    x = blockIdx.y * TILE_WIDTH + threadIdx.x;
    y = blockIdx.x * TILE_WIDTH + threadIdx.y;

    // Write to the output
    // Coalesced write
    if (x < n && y < n) {
        // Note: can't do: out[x * n + y] = tile[threadIdx.y][threadIdx.x]; because it will be uncoalesced write as the native version
        out[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }

}

__host__ void transpose(float* in, float* out, int n) {
    int size = n * n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(32, 32, 1);
    dim3 BlocksPerGrid((n + 32 - 1) / 32, (n + 32 - 1) / 32, 1);
    //transposeKernelNative<<<BlocksPerGrid, ThreadsPerBlock>>>(d_in, d_out, n);

    transposeKernelTiled<<<BlocksPerGrid, ThreadsPerBlock>>>(d_in, d_out, n);

    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

void run_benchmarks(float* in, float* out, int n) {
    int size = n * n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);

    // --- Benchmark Native ---
    cudaEventRecord(start);
    transposeKernelNative<<<blocks, threads>>>(d_in, d_out, n);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Native Kernel Time: " << milliseconds << " ms" << std::endl;

    // --- Benchmark Tiled ---
    cudaEventRecord(start);
    transposeKernelTiled<<<blocks, threads>>>(d_in, d_out, n);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Tiled Kernel Time:  " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {

    int N = 1 << 10; // 1024
    int size = N * N * sizeof(float); // 2D N*N, flattern

    float *h_in, *h_out;
    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);
    // cudaMallocHost((void**)&h_in, size); // pinned host memory, fast copy to device, avoid temporate staging buffer

    // Initial the input matrix with value i for easy checking the transposed result later
    for (int i = 0; i < N * N; i++)
    {
        h_in[i] = (float)i;
    }

    //transpose(h_in, h_out, N);

    run_benchmarks(h_in, h_out, N);

    bool success = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (h_out[j * N + i] != h_in[i * N + j]) {
                success = false;
                break;
            }
        }
    }
    std::cout << "Transpose " << (success ? "SUCCESSFUL!" : "FAILED!") << std::endl;

    free(h_in);
    free(h_out);

    return 0;
}
