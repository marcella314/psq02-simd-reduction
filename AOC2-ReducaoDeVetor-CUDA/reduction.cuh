#pragma once
#include <cstdio>
#include <cstdlib>
#include <algorithm>     
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), file, line);
        std::exit(EXIT_FAILURE);
    }
}

template<typename T, typename Op>
__global__ void reduce_kernel(const T* __restrict__ d_in, T* __restrict__ d_out, size_t N) {
    extern __shared__ T sdata[];  

    Op op;
    T val = Op::identity();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += (size_t)blockDim.x * gridDim.x) {
        val = op(val, d_in[i]);
    }

    sdata[threadIdx.x] = val;
    __syncthreads();


    for (unsigned int s = blockDim.x >> 1; s >= 64; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    T x = sdata[threadIdx.x];
    if (threadIdx.x < 32) {
        unsigned mask = 0xffffffffu;
        if (blockDim.x >= 64) x = op(x, sdata[threadIdx.x + 32]);
        x = op(x, __shfl_down_sync(mask, x, 16));
        x = op(x, __shfl_down_sync(mask, x, 8));
        x = op(x, __shfl_down_sync(mask, x, 4));
        x = op(x, __shfl_down_sync(mask, x, 2));
        x = op(x, __shfl_down_sync(mask, x, 1));
    }

    if (threadIdx.x == 0) d_out[blockIdx.x] = x;
}

template<typename T, typename Op>
T reduce_gpu(const T* d_input, size_t N, int blockSize = 256) {
    if (N == 0) return Op::identity();

    int device;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int gridSize = std::min((int)((N + blockSize - 1) / blockSize), prop.multiProcessorCount * 32);
    gridSize = std::max(gridSize, 1);

    T *d_in = const_cast<T*>(d_input);
    T *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(T) * gridSize));

    size_t n = N;
    int currBlocks = gridSize;
    T result = Op::identity();

    while (true) {
        reduce_kernel<T, Op><<<currBlocks, blockSize, blockSize * sizeof(T)>>>(d_in, d_out, n);
        CUDA_CHECK(cudaPeekAtLastError());

        if (currBlocks == 1) {
            CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost));
            break;
        }

        n = currBlocks;
        currBlocks = std::max(1, (int)((n + blockSize - 1) / blockSize));

        d_in = d_out; 
    }

    CUDA_CHECK(cudaFree(d_out));
    return result;
}
