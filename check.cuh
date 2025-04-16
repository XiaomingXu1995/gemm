#include <cstdio>
#include <cmath>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define EPSILON 1e-2

template <typename T>
void checkResult(T* hostRef, T* gpuRef,const int N){
  bool match = 1;

  for(int i = 0; i < N; i++){
    if constexpr (std::is_same<T, float>::value){
      if(abs(hostRef[i] - gpuRef[i]) > EPSILON){
        match = 0;
        printf("Arrays do not match!\n");
        printf("hostRef[%d] = %5.6f, gpuRef[%d] = %5.6f\n", i, hostRef[i], i, gpuRef[i]);
        break;
      }
    }
    else if constexpr(std::is_same<T, __half>::value){
      if(abs(__half2float(hostRef[i]) - __half2float(gpuRef[i])) > EPSILON){
        match = 0;
        printf("Arrays do not match!\n");
        printf("hostRef[%d] = %5.6f, gpuRef[%d] = %5.6f\n", i, __half2float(hostRef[i]), i, __half2float(gpuRef[i]));
        break;
      }
    }
    else if constexpr(std::is_same<T, __nv_bfloat16>::value){
      if(abs(__bfloat162float(hostRef[i]) - __bfloat162float(gpuRef[i])) > EPSILON){
        match = 0;
        printf("Arrays do not match!\n");
        printf("hostRef[%d] = %5.6f, gpuRef[%d] = %5.6f\n", i, __bfloat162float(hostRef[i]), i, __bfloat162float(gpuRef[i]));
        break;
      }
    }
    else{
      if(hostRef[i] != gpuRef[i]){
        match = 0;
        printf("Arrays do not match!\n");
        printf("hostRef[%d] = %d, gpuRef[%d] = %d\n", i, hostRef[i], i, gpuRef[i]);
        break;
      }
    }
  }

  if(match){
    printf("Arrays match!\n");
  }
}
