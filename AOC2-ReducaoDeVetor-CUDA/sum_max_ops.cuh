#pragma once
#include <cfloat>   
#include <climits>  

template<typename T>
struct SumOp {
    __host__ __device__ inline T operator()(const T& a, const T& b) const { return a + b; }
    static __host__ __device__ inline T identity() { return T(0); }
};

template<typename T>
struct MaxOp {
    __host__ __device__ inline T operator()(const T& a, const T& b) const { return (a > b) ? a : b; }
    static __host__ __device__ inline T identity(); 
};

template<>
__host__ __device__ inline float MaxOp<float>::identity() { return -FLT_MAX; }

template<>
__host__ __device__ inline double MaxOp<double>::identity() { return -DBL_MAX; }

template<>
__host__ __device__ inline int MaxOp<int>::identity() { return INT_MIN; }

template<>
__host__ __device__ inline long long MaxOp<long long>::identity() { return LLONG_MIN; }
