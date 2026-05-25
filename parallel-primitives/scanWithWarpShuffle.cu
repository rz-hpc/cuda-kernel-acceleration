
__global__ void scan_with_warp_shuffle(float* d_in, float* d_out, int n) {
    int tx = threadIdx.x;
    int global_idx = blockDim.x * blockIdx.x + tx;

    if (global_idx >= n)
      return;

    // Register
    float val = d_in[global_idx];

    // Kogge stone scan
    for (int offset = 1; offset < 32; offset <<= 1) {
        float remote = __shfl_up_sync(0xffffffff, val, offset);
        if (tx % 32 >= offset) {
            val += remote;
        }
    }

    d_out[global_idx] = val;
}

// AXPY
// y = ax + y
__global__ void axpy_fma_kernel(float* d_x, float* d_y, float a, int n) {
    // global index
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        d_y[i] = __fmaf_rn(a, d_x[i], d_y[i]);
    }
}
