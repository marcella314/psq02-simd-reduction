#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "simd_reduction.h"

using clk = std::chrono::high_resolution_clock;

template<class F> double bench(F f,int it=7){
    double best=1e100; for(int i=0;i<it;++i){
        auto t0=clk::now(); volatile float sink=f(); (void)sink;
        auto t1=clk::now();
        best=std::min(best,std::chrono::duration<double,std::milli>(t1-t0).count());
    } return best;
}

int main(){
    std::vector<size_t> sizes={
    1<<14, // 16k
    1<<15, // 32k
    1<<16, // 65k
    1<<17, // 131k
    1<<18, // 262k
    1<<19, // 524k
    1<<20, // 1M
    1<<21, // 2M
    1<<22, // 4M
    1<<23, // 8M
    1<<24  // 16M
};
    std::mt19937 rng(123); std::uniform_real_distribution<float> dist(-10.f,10.f);

    std::cout << "N,sum_scalar(ms),sum_avx2(ms),speedup_sum,"
                 "max_scalar(ms),max_avx2(ms),speedup_max\n";

    for(auto n: sizes){
        std::vector<float> a(n); for(auto& x:a) x=dist(rng);

        auto t1=bench([&]{return sum_scalar(a.data(),n);});
        auto t2=bench([&]{return sum_avx2  (a.data(),n);});
        float ref1=sum_scalar(a.data(),n), got1=sum_avx2(a.data(),n);
        if(std::abs(ref1-got1)>1e-3f*std::abs(ref1)) std::cerr<<"Soma divergente!\n";

        auto t3=bench([&]{return max_scalar(a.data(),n);});
        auto t4=bench([&]{return max_avx2  (a.data(),n);});
        float ref2=max_scalar(a.data(),n), got2=max_avx2(a.data(),n);
        if(std::abs(ref2-got2)>1e-6f) std::cerr<<"Max divergente!\n";

        std::cout<< n << "," << t1 << "," << t2 << "," << (t1/t2) << ","
                      << t3 << "," << t4 << "," << (t3/t4) << "\n";
    }
}
