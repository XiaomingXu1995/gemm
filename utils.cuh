#include <sys/time.h>
#include <cuda_runtime.h>     // for cudaError_t, cudaSuccess, cudaGetErrorString
#include <cstdio>             // for std::printf
#include <stdexcept>          // for std::runtime_error


#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

float compute_flops(const int m, const int n, const int k, float time){
  long long int ops = 2L * m * n * k;
  return ops / time;
}

float compute_gflops(const int m, const int n, const int k, float time){
  return compute_flops(m, n, k, time) / 1e9;
}

float compute_tflops(const int m, const int n, const int k, float time){
  return compute_flops(m, n, k, time) / 1e12;
}


