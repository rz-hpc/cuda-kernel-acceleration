
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAs error: %d in %s at %d!", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__host__ void cublasLinearSolverTwoStreams(float* h_A, float* h_x, float* h_b, int n, int batch_size) {

      // Set up streams and handle
      cudaStream_t s1, s2;
      cudaStreamCreate(&s1);
      cudaStreamCreate(&s2);
      cublasHandle_t handle;
      CHECK_CUBLAS(cublasCreate(&handle));

      int chunk_size = batch_size / 2;

      size_t matSizeFull = n * n * batch_size * sizeof(float);
      size_t vecSizeFull = n * batch_size * sizeof(float);
      size_t matSizeHalf = n * n * chunk_size * sizeof(float);
      size_t vecSizeHalf = n * chunk_size * sizeof(float);

      float *d_A, *d_b;
      CHECK_CUDA(cudaMalloc((void**)&d_A, matSizeFull));
      CHECK_CUDA(cudaMalloc((void**)&d_b, vecSizeFull));

      float *h_A_ptrs[batch_size];
      float *h_b_ptrs[batch_size];
      for (int i = 0; i < batch_size; i++) {
          h_A_ptrs[i] = d_A + i * n * n;
          h_b_ptrs[i] = d_b + i * n;
          // Note: here are the pointers, eqivalent, h_A_ptrs[i] = &d_A[i * n * n]
          // Node2: h_A_ptrs are the "maps" to the GPU address &d_A, not cpu &h_A,
          // because later d_A_ptrs will copy this list/maps and they have to point to the GPU address
      }

      float **d_A_ptrs, **d_b_ptrs;
      CHECK_CUDA(cudaMalloc((void**)&d_A_ptrs, batch_size * sizeof(float*)));
      CHECK_CUDA(cudaMalloc((void**)&d_b_ptrs, batch_size * sizeof(float*)));

      // Async copy to device
      CHECK_CUDA(cudaMemcpy(d_A_ptrs, h_A_ptrs, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_b_ptrs, h_b_ptrs, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

      int *d_pivot, *d_info;
      CHECK_CUDA(cudaMalloc((void**)&d_pivot, n * batch_size * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&d_info, batch_size * sizeof(int)));

      // Stream 1 (first half)
      cublasSetStream(handle, s1);

      // Async copy to device
      CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, matSizeHalf, cudaMemcpyHostToDevice, s1));
      CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, vecSizeHalf, cudaMemcpyHostToDevice, s1));

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      float milliseconds = 0.0f;

      cudaEventRecord(start);

      // LU Factorization
      CHECK_CUBLAS(cublasSgetrfBatched(handle, n, d_A_ptrs, n, d_pivot, d_info, chunk_size));

      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);

      std::cout << "Stream 1 LU Factorization milliseconds: " << milliseconds << std::endl;

      // Solve (in-place on d_b)
      int info_solve1 = 0;

      cudaEventRecord(start);

      CHECK_CUBLAS(cublasSgetrsBatched(handle, CUBLAS_OP_N, n, 1, (const float**)d_A_ptrs, n, d_pivot, d_b_ptrs, n, &info_solve1, chunk_size));

      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);

      std::cout << "Stream 1 Forward and Backward solvers milliseconds: " << milliseconds << std::endl;

      // Async copy back to the host
      CHECK_CUDA(cudaMemcpyAsync(h_x, d_b, vecSizeHalf, cudaMemcpyDeviceToHost, s1));

      // Stream 2 (second half)
      cublasSetStream(handle, s2);

      // Async copy to device
      CHECK_CUDA(cudaMemcpyAsync(d_A + (chunk_size * n * n), h_A + (chunk_size * n * n), matSizeHalf, cudaMemcpyHostToDevice, s2));
      CHECK_CUDA(cudaMemcpyAsync(d_b + (chunk_size * n), h_b + (chunk_size * n), vecSizeHalf, cudaMemcpyHostToDevice, s2));

      cudaEventRecord(start);

      // LU Factorization
      CHECK_CUBLAS(cublasSgetrfBatched(handle, n, d_A_ptrs + chunk_size, n, d_pivot + chunk_size * n, d_info + chunk_size, chunk_size));

      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);

      std::cout << "Stream 2 LU Factorization milliseconds: " << milliseconds << std::endl;

      // Solve (in-place on d_b)
      int info_solve2 = 0;

      cudaEventRecord(start);

      CHECK_CUBLAS(cublasSgetrsBatched(handle, CUBLAS_OP_N, n, 1, (const float**)(d_A_ptrs + chunk_size), n, d_pivot + chunk_size * n, d_b_ptrs + chunk_size, n, &info_solve2, chunk_size));

      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);

      std::cout << "Stream 2 Forward and Backward solvers milliseconds: " << milliseconds << std::endl;

      // Async copy back to the host
      CHECK_CUDA(cudaMemcpyAsync(h_x + (chunk_size * n), d_b + (chunk_size * n), vecSizeHalf, cudaMemcpyDeviceToHost, s2));

      // Wait for the stream pipe finish and cleanup
      // The reason we don't synchronize before the copy is
      // because the Stream itself acts like a disciplined "First-In, First-Out" (FIFO) queue.
      // We only call cudaStreamSynchronize(stream) at the end because the CPU needs to be 100% sure
      // the data has actually arrived in h_x before it tries to read it (e.g., for verification or printing).
      cudaStreamSynchronize(s1);
      cudaStreamSynchronize(s2);

      cublasDestroy(handle);
      cudaStreamDestroy(s1);
      cudaStreamDestroy(s2);
      CHECK_CUDA(cudaFree(d_pivot)); CHECK_CUDA(cudaFree(d_info));
      CHECK_CUDA(cudaFree(d_A_ptrs)); CHECK_CUDA(cudaFree(d_b_ptrs));
      CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_b));
      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
}

__host__ void cublasLinearSolverStreams(float* h_A, float* h_x, float* h_b, int n, int batch_size) {
    const int num_streams = 4;
    int chunk_size = batch_size / num_streams;

    // Memory Sizes
    size_t matSizeFull = n * n * batch_size * sizeof(float);
    size_t vecSizeFull = n * batch_size * sizeof(float);
    size_t matSizeChunk = n * n * chunk_size * sizeof(float);
    size_t vecSizeChunk = n * chunk_size * sizeof(float);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 1. Allocate Device Data
    float *d_A, *d_b;
    CHECK_CUDA(cudaMalloc((void**)&d_A, matSizeFull));
    CHECK_CUDA(cudaMalloc((void**)&d_b, vecSizeFull));

    // 2. Prepare the Pointer "Maps" on Host
    float **h_A_ptrs = (float**)malloc(batch_size * sizeof(float*));
    float **h_b_ptrs = (float**)malloc(batch_size * sizeof(float*));
    for (int i = 0; i < batch_size; i++) {
        h_A_ptrs[i] = d_A + (i * n * n);
        h_b_ptrs[i] = d_b + (i * n);
    }

    // 3. Copy "Maps" to Device
    float **d_A_ptrs, **d_b_ptrs;
    CHECK_CUDA(cudaMalloc((void**)&d_A_ptrs, batch_size * sizeof(float*)));
    CHECK_CUDA(cudaMalloc((void**)&d_b_ptrs, batch_size * sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(d_A_ptrs, h_A_ptrs, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_ptrs, h_b_ptrs, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

    int *d_pivot, *d_info;
    CHECK_CUDA(cudaMalloc((void**)&d_pivot, n * batch_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_info, batch_size * sizeof(int)));

    // 4. Create Streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) cudaStreamCreate(&streams[i]);

    // 5. The Dispatch Loop (Non-Blocking!)
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        cublasSetStream(handle, streams[i]);

        // Async Copies (Input)
        CHECK_CUDA(cudaMemcpyAsync(d_A + (offset * n * n), h_A + (offset * n * n), matSizeChunk, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_b + (offset * n), h_b + (offset * n), vecSizeChunk, cudaMemcpyHostToDevice, streams[i]));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0.0f;

        // Batched Solve
        int info = 0;
        cudaEventRecord(start);
        CHECK_CUBLAS(cublasSgetrfBatched(handle, n, d_A_ptrs + offset, n, d_pivot + (offset * n), d_info + offset, chunk_size));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Stream " << i << " LU Factorization milliseconds: " << milliseconds << std::endl;

        cudaEventRecord(start);
        CHECK_CUBLAS(cublasSgetrsBatched(handle, CUBLAS_OP_N, n, 1, (const float**)(d_A_ptrs + offset), n, d_pivot + (offset * n), d_b_ptrs + offset, n, &info, chunk_size));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Stream " << i << " Forward and Backward solvers milliseconds: " << milliseconds << std::endl;

        // Async Copy (Output)
        CHECK_CUDA(cudaMemcpyAsync(h_x + (offset * n), d_b + (offset * n), vecSizeChunk, cudaMemcpyDeviceToHost, streams[i]));
    }

    // 6. Final Sync - This is where the CPU actually waits!
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 7. Cleanup
    for (int i = 0; i < num_streams; i++) cudaStreamDestroy(streams[i]);
    cublasDestroy(handle);
    free(h_A_ptrs); free(h_b_ptrs);
    cudaFree(d_A); cudaFree(d_b);
    cudaFree(d_A_ptrs); cudaFree(d_b_ptrs);
    cudaFree(d_pivot); cudaFree(d_info);
}

int main() {
    const int n = 32;
    const int batch_size = 10000;

    float *h_A, *h_b, *h_x;

    // Use cudaMallocHost instead of malloc for "Pinned" memory
    // This makes cudaMemcpyAsync truly concurrent
    CHECK_CUDA(cudaMallocHost((void**)&h_A, n * n * batch_size * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_b, n * batch_size * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_x, n * batch_size * sizeof(float)));

    // To verify, we need the original A and b
    float* h_A_orig = (float*)malloc(n * n * batch_size * sizeof(float));
    float* h_b_orig = (float*)malloc(n * batch_size * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < n; j++) { // j is column
            for (int i = 0; i < n; i++) { // i is row
                // COLUMN-MAJOR: index = col * n + row
                float val = (float)rand() / RAND_MAX;
                h_A[b * n * n + j * n + i] = val;
                h_A_orig[b * n * n + j * n + i] = val;
            }
        }
        for (int i = 0; i < n; i++) {
            float val = (float)rand() / RAND_MAX;
            h_b[b * n + i] = val;
            h_b_orig[b * n + i] = val;
        }
    }

    cublasLinearSolverStreams(h_A, h_x, h_b, n, batch_size);

    // Final Verification
    double total_max_err = 0;
    int failed = 0;
    for (int b = 0; b < batch_size; b++) {
        float max_err = 0;
        for (int i = 0; i < n; i++) { // Row
            double sum = 0;
            for (int j = 0; j < n; j++) { // Col
                // Column-Major access: j * n + i
                sum += (double)h_A_orig[b * n * n + j * n + i] * h_x[b * n + j];
            }
            double err = fabs(sum - h_b_orig[b * n + i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 1e-3) failed++;
        if (max_err > total_max_err) total_max_err = max_err;
    }

    std::cout << "Verification Result: " << (failed == 0 ? "✅ PASS" : "⚠️ FAIL") << std::endl;
    std::cout << "Max Error: " << total_max_err << std::endl;
    if (failed > 0) std::cout << "Failed matrices: " << failed << std::endl;

    // Use cudaFreeHost for pinned memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_b);
    cudaFreeHost(h_x);

    return 0;
}
