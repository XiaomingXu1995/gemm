#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
//#include <bits/stdc++.h>
#include <sys/time.h>
#include <assert.h>
#include <immintrin.h>


#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#define CHECK(call)                     \
{                                       \
    const cudaError_t error = call;     \
    if(error!=cudaSuccess)              \
    {                                   \
        printf("Error: %s:%d",__FILE__,__LINE__);      \
        std::cout<<"code: "<<error<<" ,reason: "<<cudaGetErrorString(error)<<std::endl;     \
        exit(-10*error);     \
    }                        \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << " - " << err << std::endl; \
        exit(-1); \
    } \
}

using namespace std;
using data_type = float;
using data_type_1 = float;
//using data_type = float;
//using data_type = double;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

template<typename T>
void initialData_int8(T*ip,int size)
{
    int max_int8 = 127;
    srand(0);
    for(int i=0;i<size;i++)
    {
        ip[i] = rand() % max_int8;
    }
}

template<typename T>
void initialData(T*ip,int size, int step)
{
    srand(0);
    for(int i=0;i<size;i++)
    {
				data_type rand_num = rand() % step;
				ip[i] = rand_num / step / 10;
    }
}

template<typename T>
void initialData_fp16(T*ip,int size, int step)
{
    step = 10;
    srand(0);
    for(int i=0;i<size;i++)
    {
				data_type rand_num = rand() % step;
        float f = rand_num;
        ip[i] = __float2half(f);
        if(i < 20){
          printf("ip[%d] = %f\n", i, __half2float(ip[i]));
        }
    }
}


template<typename X>
void checkResult(X* hostRef, X* gpuRef,const int N)
{
  if constexpr (std::is_same<X, float>::value){
    double epsilon = 1.0E-2;
    bool match=1;
    for(int i=0;i<N;i++)
    {
        if(abs(hostRef[i]-gpuRef[i])>epsilon)
        {
            match=0;
            printf("Arrays do not match");
            printf("host %5.6f gpu %5.6f at current %d\n",hostRef[i],gpuRef[i],i);
            break;
        }
    }
    if(match)
      std::cout<<"Arrats match"<<std::endl;
  }
  else if constexpr (std::is_same<X, int32_t>::value){
    bool match = 1;
    for(int i=0;i<N;i++){
      if(hostRef[i] != gpuRef[i]){
        printf("Arrays do not match");
        printf("host %d gpu %d at current %d\n", hostRef[i], gpuRef[i], i);
        match = 0;
        break;
      }
    }
    if(match)
      std::cout<<"Arrats match"<<std::endl;
  }
}

// template<>
// void checkResult<__half>(__half*, __half*, int) = delete;

void checkResult_half(__half* hostRef, __half* gpuRef,const int N)
{
  float epsilon = 1.0E-2;
  bool match=1;
  for(int i=0;i<N;i++)
  {
    float diff = abs(__half2float(hostRef[i])-__half2float(gpuRef[i]));
    if(diff > epsilon)
    {
      match=0;
          printf("Arrays do not match");
          printf("host %5.6f gpu %5.6f at current %d\n",__half2float(hostRef[i]),__half2float(gpuRef[i]),i);
          break;
      }
  }
  if(match)
    std::cout<<"Arrats match"<<std::endl;
}

double get_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}

template<typename T, typename U>
__global__ void matrixMulOnGPU(T* A, T* B, U* C, const int m, const int n, const int k){
	// A[m][k], B[k][n], C[m][n];
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
	U val = 0.0f;
	if(row < m && col < n){
		for(int i = 0; i < k; i++){
			val += A[row*k+i] * B[i*n+col]; 
		}
		C[row*n+col] = val;
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

template<typename T, typename U>
void mm_cublas(int warmup, int repeat, const int m, const int n, const int k){
  if(std::is_same<T, int8_t>::value && std::is_same<U, float>::value){
    cout << "--------------------------------mm_cublas_int8_int8_float--------------------------------" << endl;
  }
  else if(std::is_same<T, float>::value && std::is_same<U, float>::value){
    cout << "--------------------------------mm_cublas_float_float_float--------------------------------" << endl;
  }
  else if(std::is_same<T, __half>::value && std::is_same<U, __half>::value){
    cout << "--------------------------------mm_cublas_fp16_fp16_fp16--------------------------------" << endl;
  }
  else if(std::is_same<T, int8_t>::value && std::is_same<U, int32_t>::value){
    cout << "--------------------------------mm_cublas_int8_int8_int32--------------------------------" << endl;
  }
  long ops = 2L * m * n * k;
  cudaStream_t stream = NULL;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  T *h_A = nullptr;
  T *h_B = nullptr;
  U *h_C = nullptr;
  U *h_C_ref = nullptr;

  h_A = (T *)malloc(m * k * sizeof(T));
  h_B = (T *)malloc(k * n * sizeof(T));
  h_C = (U *)malloc(m * n * sizeof(U));

  h_C_ref = (U *)malloc(m * n * sizeof(U));
  if constexpr(std::is_same<T, int8_t>::value){
    initialData_int8<T>(h_A, m * k);
    initialData_int8<T>(h_B, k * n);
  }
  else if constexpr(std::is_same<T, float>::value){
    initialData<T>(h_A, m * k, 1000);
    initialData<T>(h_B, k * n, 1000);
  }
  else if constexpr(std::is_same<T, __half>::value){
    initialData_fp16<T>(h_A, m * k, 1000);
    initialData_fp16<T>(h_B, k * n, 1000);
  }
  // memset(h_C_ref, 0, m * n * sizeof(U));
  // matrixMulOnHost<T, U>(h_A, h_B, h_C_ref, m, n, k);

  T *d_A = nullptr;
  T *d_B = nullptr;
  U *d_C = nullptr;
  U *d_C_ref = nullptr;

  cublasHandle_t cublasH = NULL;
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  const data_type alpha = 1.0;
  const data_type beta = 0.0;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * m * k));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(T) * k * n));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(U) * m * n));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_ref), sizeof(U) * m * n));

  CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, sizeof(T) * m * k, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, sizeof(T) * k * n, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  dim3 block(32, 32);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  matrixMulOnGPU<T, U><<<grid, block, 0, stream>>>(d_A, d_B, d_C_ref, m, n, k);
  CUDA_CHECK(cudaMemcpyAsync(h_C_ref, d_C_ref, sizeof(U) * m * n, cudaMemcpyDeviceToHost, stream));
  if constexpr(std::is_same<T, int8_t>::value && std::is_same<U, float>::value){
    CHECK_CUBLAS(cublasGemmEx(
      cublasH,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha, 
      d_A, CUDA_R_8I, m,         // A (INT8 matrix)
      d_B, CUDA_R_8I, k,         // B (INT8 matrix)
      &beta,
      d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
      CUDA_R_32F,                // The compute type is FP32
      CUBLAS_GEMM_DEFAULT));     // Use default algorithm
  }
  else if constexpr(std::is_same<T, int8_t>::value && std::is_same<U, int32_t>::value){
    CHECK_CUBLAS(cublasGemmEx(
      cublasH,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha, 
      d_A, CUDA_R_8I, m,         // A (INT8 matrix)
      d_B, CUDA_R_8I, k,         // B (INT8 matrix)
      &beta,
      d_C, CUDA_R_32I, m,        // C (INT32 output matrix)
      CUDA_R_32I,                // The compute type is int32
      CUBLAS_GEMM_DEFAULT));     // Use default algorithm
  }
  else if constexpr(std::is_same<T, float>::value && std::is_same<U, float>::value){
    CHECK_CUBLAS(cublasGemmEx(
      cublasH,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha, 
      d_A, CUDA_R_32F, m,         // A (FP32 matrix)
      d_B, CUDA_R_32F, k,         // B (FP32 matrix)
      &beta,
      d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
      CUDA_R_32F,                // The compute type is FP32
      CUBLAS_GEMM_DEFAULT));     // Use default algorithm
  }
  else if constexpr(std::is_same<T, __half>::value && std::is_same<U, __half>::value){
    // matrixMulOnGPU<__half, __half><<<32, 32, 0, stream>>>(d_A, d_B, d_C, m, n, k);
    CHECK_CUBLAS(cublasGemmEx(
      cublasH,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha, 
      d_A, CUDA_R_16F, m,         // A (FP16 matrix)
      d_B, CUDA_R_16F, k,         // B (FP16 matrix)
      &beta,
      d_C, CUDA_R_16F, m,        // C (FP16 output matrix)
      CUDA_R_16F,                // The compute type is FP16
      CUBLAS_GEMM_DEFAULT));     // Use default algorithm

  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, sizeof(U) * m * n, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());

  if constexpr (std::is_same<U, __half>::value){
    checkResult_half(h_C_ref, h_C, m * n);
    for(int i = 0; i < m * n; i++){
      printf("h_C_ref[%d] = %f, h_C[%d] = %f\n", i, __half2float(h_C_ref[i]), i, __half2float(h_C[i]));
      if(i > 20) break;
    }
  }
  else if constexpr (std::is_same<U, float>::value){
    checkResult<U>(h_C_ref, h_C, m * n);
    for(int i = 0; i < m * n; i++){
      printf("h_C_ref[%d] = %f, h_C[%d] = %f\n", i, h_C_ref[i], i, h_C[i]);
      if(i > 20) break;
    }
  }
  else if constexpr (std::is_same<U, int32_t>::value){
    checkResult<U>(h_C_ref, h_C, m * n);
    for(int i = 0; i < m * n; i++){
      printf("h_C_ref[%d] = %d, h_C[%d] = %d\n", i, h_C_ref[i], i, h_C[i]);
      if(i > 20) break;
    }
  }

  for(int i = 0; i < warmup; i++){
    if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, float>::value){
      CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_8I, m,         // A (INT8 matrix)
        d_B, CUDA_R_8I, k,         // B (INT8 matrix)
        &beta,
        d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
        CUDA_R_32F,                // The compute type is FP32
        CUBLAS_GEMM_DEFAULT));     // Use default algorithm
    }
    else if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, int32_t>::value){
      CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_8I, m,         // A (INT8 matrix)
        d_B, CUDA_R_8I, k,         // B (INT8 matrix)
        &beta,
        d_C, CUDA_R_32I, m,        // C (INT32 output matrix)
        CUDA_R_32I,                // The compute type is int32
        CUBLAS_GEMM_DEFAULT));     // Use default algorithm
    }
    else if constexpr (std::is_same<T, float>::value && std::is_same<U, float>::value){
      CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_32F, m,         // A (FP32 matrix)
        d_B, CUDA_R_32F, k,         // B (FP32 matrix)
        &beta,
        d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
        CUDA_R_32F,                // The compute type is FP32
        CUBLAS_GEMM_DEFAULT));     // Use default algorithm
    }
    else if constexpr (std::is_same<T, __half>::value && std::is_same<U, __half>::value){
      CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_16F, m,         // A (FP16 matrix)
        d_B, CUDA_R_16F, k,         // B (FP16 matrix)
        &beta,
        d_C, CUDA_R_16F, m,        // C (FP16 output matrix)
        CUDA_R_16F,                // The compute type is FP16
        CUBLAS_GEMM_DEFAULT));     // Use default algorithm
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, float>::value){
    std::cout << "----int8_int8_fp32 warmup done" << std::endl;
  }
  else if constexpr (std::is_same<T, float>::value && std::is_same<U, float>::value){
    std::cout << "----fp32_fp32_fp32 warmup done" << std::endl;
  }
  else if constexpr (std::is_same<T, __half>::value && std::is_same<U, __half>::value){
    std::cout << "----fp16_fp16_fp16 warmup done" << std::endl;
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));

  for(int i = 0; i < repeat; i++){
    if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, float>::value){
      CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_8I, m,         // A (INT8 matrix)
        d_B, CUDA_R_8I, k,         // B (INT8 matrix)
        &beta,
        d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
        CUDA_R_32F,                // The compute type is FP32
        CUBLAS_GEMM_DEFAULT));     // Use default algorithm
    }
    else if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, int32_t>::value){
      CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_8I, m,         // A (INT8 matrix)
        d_B, CUDA_R_8I, k,         // B (INT8 matrix)
        &beta,
        d_C, CUDA_R_32I, m,        // C (INT32 output matrix)
        CUDA_R_32I,                // The compute type is int32
        CUBLAS_GEMM_DEFAULT));     // Use default algorithm
    }
    else if constexpr (std::is_same<T, float>::value && std::is_same<U, float>::value){
      CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_32F, m,         // A (FP32 matrix)
        d_B, CUDA_R_32F, k,         // B (FP32 matrix)
        &beta,
        d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
        CUDA_R_32F,                // The compute type is FP32
        CUBLAS_GEMM_DEFAULT));     // Use default algorithm
    }
    else if constexpr (std::is_same<T, __half>::value && std::is_same<U, __half>::value){
      CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_16F, m,         // A (fp16 matrix)
        d_B, CUDA_R_16F, k,         // B (fp16 matrix)
        &beta,
        d_C, CUDA_R_16F, m,        // C (fp16 output matrix)
        CUDA_R_16F,                // The compute type is FP16
        CUBLAS_GEMM_DEFAULT));     // Use default algorithm
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, float>::value){
    std::cout << "----int8_int8_fp32 Time taken: " << milliseconds << " ms" << std::endl;
    std::cout << "----int8_int8_fp32 TFlops: " << (double)ops * 1e-12 / (milliseconds / repeat / 1000) << std::endl;
  }
  else if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, int32_t>::value){
    std::cout << "----int8_int8_int32 Time taken: " << milliseconds << " ms" << std::endl;
    std::cout << "----int8_int8_int32 TFlops: " << (double)ops * 1e-12 / (milliseconds / repeat / 1000) << std::endl;
  }
  else if constexpr (std::is_same<T, float>::value && std::is_same<U, float>::value){
    std::cout << "----fp32_fp32_fp32 Time taken: " << milliseconds << " ms" << std::endl;
    std::cout << "----fp32_fp32_fp32 TFlops: " << (double)ops * 1e-12 / (milliseconds / repeat / 1000) << std::endl;
  }
  else if constexpr (std::is_same<T, __half>::value && std::is_same<U, __half>::value){
    std::cout << "----fp16_fp16_fp16 Time taken: " << milliseconds << " ms" << std::endl;
    std::cout << "----fp16_fp16_fp16 TFlops: " << (double)ops * 1e-12 / (milliseconds / repeat / 1000) << std::endl;
  }

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceReset());
  cout << endl;
}

int main(int argc, char *argv[]) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  int bits = 10;
  int warmup = 30;
  int repeat = 3000;
  if(argc >= 2){
    bits = std::stoi(argv[1]);
  }
  if(argc >= 3){
    warmup = std::stoi(argv[2]);
    repeat = warmup * 100;
  }
  if(argc >= 4){
    repeat = std::stoi(argv[3]);
  }
  //CHECK宏定义检查操作是否正常处理
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using Device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));
  //set up data size of matrix
  int m = 1<<bits; //16384
  int k = 1<<bits; //16384
  int n = 1<<bits; //16384
  mm_cublas<int8_t, float>(warmup, repeat, m, n, k);
  mm_cublas<float, float>(warmup, repeat, m, n, k);
  mm_cublas<__half, __half>(warmup, repeat, m, n, k);
  mm_cublas<int8_t, int32_t>(warmup, repeat, m, n, k);
	return 0;

}
