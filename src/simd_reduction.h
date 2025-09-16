#pragma once
#include <immintrin.h>
#include <cstddef>
#include <algorithm>

inline float sum_scalar(const float* a, size_t n){
    float s=0.0f; for(size_t i=0;i<n;++i) s+=a[i]; return s;
}
inline float max_scalar(const float* a, size_t n){
    float m=a[0]; for(size_t i=1;i<n;++i) m=std::max(m,a[i]); return m;
}

// AVX2: soma
inline float sum_avx2(const float* a, size_t n){
    const size_t step=8; size_t i=0; __m256 acc=_mm256_setzero_ps();
    for(; i+step<=n; i+=step){ __m256 v=_mm256_loadu_ps(a+i); acc=_mm256_add_ps(acc,v); }
    __m128 lo=_mm256_castps256_ps128(acc), hi=_mm256_extractf128_ps(acc,1);
    __m128 s=_mm_add_ps(lo,hi); s=_mm_hadd_ps(s,s); s=_mm_hadd_ps(s,s);
    float total=_mm_cvtss_f32(s); for(; i<n; ++i) total+=a[i]; return total;
}

// AVX2: máximo
inline float max_avx2(const float* a, size_t n){
    const size_t step=8; size_t i=0; __m256 vmax=_mm256_loadu_ps(a); i+=step;
    for(; i+step<=n; i+=step){ __m256 v=_mm256_loadu_ps(a+i); vmax=_mm256_max_ps(vmax,v); }
    __m128 lo=_mm256_castps256_ps128(vmax), hi=_mm256_extractf128_ps(vmax,1);
    __m128 m=_mm_max_ps(lo,hi); __m128 sh=_mm_movehdup_ps(m); m=_mm_max_ps(m,sh);
    sh=_mm_movehl_ps(sh,m); m=_mm_max_ps(m,sh); sh=_mm_shuffle_ps(m,m,0x55); m=_mm_max_ps(m,sh);
    float res=_mm_cvtss_f32(m); for(; i<n; ++i) res=std::max(res,a[i]); return res;
}
