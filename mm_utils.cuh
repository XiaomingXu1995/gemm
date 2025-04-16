#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T, typename U>
__global__ void matrixMulOnGPU(T* A, T* B, U* C, const int m, const int n, const int k){
	// A[m][k], B[k][n], C[m][n];
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
  if constexpr (std::is_same<T, __half>::value && std::is_same<U, __half>::value){
    float val = 0.0f;
    if(row < m && col < n){
      for(int i = 0; i < k; i++){
        val += __half2float(A[row*k+i]) * __half2float(B[i*n+col]); 
      }
      C[row*n+col] = __float2half(val);
    }
  }
  else{
	  U val = 0;
	  if(row < m && col < n){
	  	for(int i = 0; i < k; i++){
       if constexpr (std::is_same<T, __half>::value && std::is_same<U, float>::value){
         val += __half2float(A[row*k+i]) * __half2float(B[i*n+col]); 
       }
       else{
         val += A[row*k+i] * B[i*n+col]; 
       }
	  	}
	  	C[row*n+col] = val;
	  }
  }
}

template<typename T, typename U>
void matrixMulOnHost(T * A, T* B, U * C, const int m, const int n, const int k){
	// A[m][k], B[k][n], C[m][n];
	for(int i = 0; i < m; i++){
	    for(int j  = 0; j < k; j++){
		for(int t = 0; t < n; t++){
				C[i * n + t] += A[i*k + j] * B[j*n + t];
			}
		}
	}
}
