#!/bin/bash
set -e

# 1. Ensure micromamba binary exists locally
if [ ! -f "./bin/micromamba" ]; then
    echo "Downloading micromamba..."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
fi

# 2. Create the environment only if it doesn't already exist
if [ ! -d "./cublasmp_env" ]; then
    echo "Creating micromamba cuBLASMp environment..."
    ./bin/micromamba create -y -p ./cublasmp_env -c nvidia -c conda-forge \
    libcublasmp-dev libcublasmp nccl cuda-cudart
else
    echo "Found existing ./cublasmp_env, skipping creation."
fi

# 3. Automatically detect nvcc path
NVCC_PATH=""
if [ -f "/usr/local/cuda/bin/nvcc" ]; then
    NVCC_PATH="/usr/local/cuda/bin/nvcc"
elif command -v nvcc &> /dev/null; then
    NVCC_PATH="nvcc"
else
    echo "Error: nvcc compiler not found on this system!"
    exit 1
fi

echo "Using nvcc located at: $NVCC_PATH"

# 4. Compile the target code
echo "Compiling summa_cublasmp..."
$NVCC_PATH -Wno-deprecated-gpu-targets -arch=sm_75 summa_cublasmp.cu -o summa_cublasmp \
 -I./cublasmp_env/include \
 -I./cublasmp_env/include/cublasMp \
 -L./cublasmp_env/lib \
 -lcublas -lnccl -lcublasmp -lcudart \
 -ccbin mpicxx

echo "Build complete successfully! Running application..."

# 5. Execute the binary
mpirun --allow-run-as-root --oversubscribe -np 1 \
 -x LD_LIBRARY_PATH=$(pwd)/cublasmp_env/lib:$LD_LIBRARY_PATH \
 -x NCCL_DEBUG=WARN \
 -x NCCL_MULTI_RANK_GPU_ENABLE=1 \
 -x CUBLASMP_LOG_LEVEL=3 \
 ./summa_cublasmp
