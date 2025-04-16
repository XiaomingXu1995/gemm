#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define RAND_SEED 0
#define STEP 10
#define MAX_INT8 127


template<typename T>
void initialData(T* ip, size_t size){
  srand(RAND_SEED);
  for(int i = 0; i < size; i++){
    if constexpr (std::is_same<T, int8_t>::value){
      ip[i] = rand() % MAX_INT8;
    }
    else if constexpr (std::is_same<T, __half>::value){
      ip[i] = __float2half(rand() % STEP);
    }
    else if constexpr (std::is_same<T, __nv_bfloat16>::value){
      ip[i] = __float2bfloat16(rand() % STEP);
    }
    else{
      ip[i] = rand() % STEP;
    }
  }
}
