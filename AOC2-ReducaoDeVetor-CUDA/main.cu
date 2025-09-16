#include <cstdio>
#include <cstdlib>
#include <algorithm>     
#include <vector>
#include <random>
#include <chrono>
#include <cfloat>        
#include "reduction.cuh"
#include "sum_max_ops.cuh"

using clk = std::chrono::high_resolution_clock;


template<typename T, typename Op>
T reduce_cpu(const T* data, size_t N) {
    Op op;
    T acc = Op::identity();
    for (size_t i = 0; i < N; ++i) acc = op(acc, data[i]);
    return acc;
}


template<class F>
double bench_best_ms(F f, int it = 7) {
    double best = 1e100;
    for (int i = 0; i < it; ++i) {
        auto t0 = clk::now();
        volatile float sink = f(); (void)sink; 
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best = std::min(best, ms);
    }
    return best;
}

int main() {
    std::vector<size_t> sizes = {
        1u<<14, 1u<<15, 1u<<16, 1u<<17, 1u<<18, 1u<<19,
        1u<<20, 1u<<21, 1u<<22, 1u<<23, 1u<<24
    };

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);

    std::puts("N,sum_scalar(ms),sum_cuda(ms),speedup_sum,max_scalar(ms),max_cuda(ms),speedup_max");

    for (size_t N : sizes) {
        std::vector<float> h(N);
        for (auto &x : h) x = dist(rng);

        float* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, sizeof(float) * N));
        CUDA_CHECK(cudaMemcpy(d, h.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

        float sum_cpu_val = 0.0f;
        double sum_cpu_ms = bench_best_ms([&] {
            sum_cpu_val = reduce_cpu<float, SumOp<float>>(h.data(), N);
            return sum_cpu_val;
        });

        (void)reduce_gpu<float, SumOp<float>>(d, N);

        cudaEvent_t es, ee;
        CUDA_CHECK(cudaEventCreate(&es));
        CUDA_CHECK(cudaEventCreate(&ee));
        float sum_gpu_val = 0.0f;
        float sum_gpu_best = 1e30f;

        for (int r = 0; r < 7; ++r) {
            CUDA_CHECK(cudaEventRecord(es));
            sum_gpu_val = reduce_gpu<float, SumOp<float>>(d, N);
            CUDA_CHECK(cudaEventRecord(ee));
            CUDA_CHECK(cudaEventSynchronize(ee));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, es, ee));
            sum_gpu_best = std::min(sum_gpu_best, ms);
        }
        CUDA_CHECK(cudaEventDestroy(es));
        CUDA_CHECK(cudaEventDestroy(ee));

        float max_cpu_val = -FLT_MAX;
        double max_cpu_ms = bench_best_ms([&] {
            max_cpu_val = reduce_cpu<float, MaxOp<float>>(h.data(), N);
            return max_cpu_val;
        });

        (void)reduce_gpu<float, MaxOp<float>>(d, N);

        CUDA_CHECK(cudaEventCreate(&es));
        CUDA_CHECK(cudaEventCreate(&ee));
        float max_gpu_val = -FLT_MAX;
        float max_gpu_best = 1e30f;

        for (int r = 0; r < 7; ++r) {
            CUDA_CHECK(cudaEventRecord(es));
            max_gpu_val = reduce_gpu<float, MaxOp<float>>(d, N);
            CUDA_CHECK(cudaEventRecord(ee));
            CUDA_CHECK(cudaEventSynchronize(ee));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, es, ee));
            max_gpu_best = std::min(max_gpu_best, ms);
        }
        CUDA_CHECK(cudaEventDestroy(es));
        CUDA_CHECK(cudaEventDestroy(ee));

        double speedup_sum = (sum_gpu_best > 0.0f) ? (sum_cpu_ms / sum_gpu_best) : 0.0;
        double speedup_max = (max_gpu_best > 0.0f) ? (max_cpu_ms / max_gpu_best) : 0.0;

        std::printf("%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    N,
                    sum_cpu_ms, sum_gpu_best, speedup_sum,
                    max_cpu_ms, max_gpu_best, speedup_max);

        CUDA_CHECK(cudaFree(d));
    }

    return 0;
}
