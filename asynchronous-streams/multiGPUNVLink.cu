
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {

    // Get device count
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        std::cout << "This script requires 2 GPU devices. Found only " << deviceCount << std::endl;
        return 0;
    }

    // Enable Peer-to-Peer access
    // Allow GPU device 0 to talk to GPU 1's memory via NVLink/PCIe directly
    int canAccessPeer;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));

    if (canAccessPeer) {
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(1, 0));
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(0, 0));
        std::cout << "P2P Access Enabled between Device 0 and 1" << std::endl;
    }

    // Allocate memory on both devices
    float *d_0, *d_1;
    size_t size = 1024 * sizeof(float);

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc((void**)&d_0, size));
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc((void**)&d_1, size));

    // Peer-to-Peer copy
    // We are on device 1, but we can command a copy from device 0 memory
    CHECK_CUDA(cudaMemcpyPeerAsync(d_1, 1, d_0, 0, size, 0));

    std::cout << "Successfully initiled P2P memory copy from device 0 to device 1" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaSetDevice(0)); cudaFree(d_0);
    CHECK_CUDA(cudaSetDevice(1)); cudaFree(d_1);

    return 0;
}
