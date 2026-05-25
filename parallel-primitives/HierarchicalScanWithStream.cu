
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define SHMEM_INDEX(i) (((i) + ((i) >> 5)))

__host__ void run_hierarchical_scan_async(float* h_in, float* h_out, int n) {

    const int threadsPerBlock = 1024;
    const int sizeN = n * sizeof(float);
    const int shmemSize = sizeof(float) * (threadsPerBlock + (threadsPerBlock >> 5));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *d_in, *d_out, *d_block_sums1, *d_block_sums2;
    CHECK_CUDA(cudaMalloc((void**)&d_in, sizeN));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeN));

    CHECK_CUDA(cudaMemcpyAsync(d_in, h_in, sizeN, cudaMemcpyHostToDevice, stream));

    int blocksPerGridLevel1 = (n + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridLevel2 = (blocksPerGridLevel1 + threadsPerBlock - 1) / threadsPerBlock;

    CHECK_CUDA(cudaMalloc((void**)&d_block_sums1, blocksPerGridLevel1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_block_sums2, blocksPerGridLevel2 * sizeof(float)));

    scan_and_export_sums<<<blocksPerGridLevel1, threadsPerBlock, shmemSize, stream>>>(d_in, d_out, d_block_sums1, n);
    scan_and_export_sums<<<blocksPerGridLevel2, threadsPerBlock, shmemSize, stream>>>(d_block_sums1, d_block_sums1, d_block_sums2, blocksPerGridLevel1);

    brent_kung_padded_kernel<<<1, threadsPerBlock, shmemSize, stream>>>(d_block_sums2, d_block_sums2, blocksPerGridLevel2);

    add_block_offsets<<<blocksPerGridLevel2, threadsPerBlock, 0, stream>>>(d_block_sums1, d_block_sums2, blocksPerGridLevel1);
    add_block_offsets<<<blocksPerGridLevel1, threadsPerBlock, 0, stream>>>(d_out, d_block_sums1, n);

    CHECK_CUDA(cudaMemcpyAsync(h_out, d_out, sizeN, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaStreamDestroy(stream);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_block_sums1); cudaFree(d_block_sums2);
}
