#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <immintrin.h>


#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "utils.cuh"
#include "init.cuh"
#include "check.cuh"
#include "mm_utils.cuh"

using namespace std;

template<typename T, typename U>
void mm_cublas(int warmup, int repeat, const int m, const int n, const int k){
  if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, float>::value){
    cout << "--------------------------------mm_cublas_int8_int8_float--------------------------------" << endl;
  }
  else if constexpr (std::is_same<T, float>::value && std::is_same<U, float>::value){
    cout << "--------------------------------mm_cublas_float_float_float--------------------------------" << endl;
  }
  else if constexpr (std::is_same<T, __half>::value && std::is_same<U, __half>::value){
    cout << "--------------------------------mm_cublas_fp16_fp16_fp16--------------------------------" << endl;
  }
  else if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, int32_t>::value){
    cout << "--------------------------------mm_cublas_int8_int8_int32--------------------------------" << endl;
  }
  else if constexpr (std::is_same<T, __half>::value && std::is_same<U, float>::value){
    cout << "--------------------------------mm_cublas_fp16_fp16_float--------------------------------" << endl;
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
  initialData<T>(h_A, m * k);
  initialData<T>(h_B, k * n);

  T *d_A = nullptr;
  T *d_B = nullptr;
  U *d_C = nullptr;
  U *d_C_ref = nullptr;

  cublasHandle_t cublasH = NULL;
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  const float alpha = 1.0;
  const float beta = 0.0;

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
    CUBLAS_CHECK(cublasGemmEx(
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
    CUBLAS_CHECK(cublasGemmEx(
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
    CUBLAS_CHECK(cublasGemmEx(
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
    CUBLAS_CHECK(cublasGemmEx(
      cublasH,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha, 
      d_A, CUDA_R_16F, m,         // A (FP16 matrix)
      d_B, CUDA_R_16F, k,         // B (FP16 matrix)
      &beta,
      d_C, CUDA_R_16F, m,        // C (FP16 output matrix)
      CUBLAS_COMPUTE_32F,                // The compute type is FP16
      CUBLAS_GEMM_DEFAULT));     // Use default algorithm

  }
  else if constexpr(std::is_same<T, __half>::value && std::is_same<U, float>::value){
    CUBLAS_CHECK(cublasGemmEx(
      cublasH,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha, 
      d_A, CUDA_R_16F, m,         // A (FP16 matrix)
      d_B, CUDA_R_16F, k,         // B (FP16 matrix)
      &beta,
      d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
      CUBLAS_COMPUTE_32F,                // The compute type is FP32
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));     // Use default algorithm

  }
  
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, sizeof(U) * m * n, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());

  checkResult<U>(h_C_ref, h_C, m * n);

  for(int i = 0; i < warmup; i++){
    if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, float>::value){
      CUBLAS_CHECK(cublasGemmEx(
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
      CUBLAS_CHECK(cublasGemmEx(
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
      CUBLAS_CHECK(cublasGemmEx(
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
      CUBLAS_CHECK(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_16F, m,         // A (FP16 matrix)
        d_B, CUDA_R_16F, k,         // B (FP16 matrix)
        &beta,
        d_C, CUDA_R_16F, m,        // C (FP16 output matrix)
        CUBLAS_COMPUTE_32F,                // The compute type is FP16
        CUBLAS_GEMM_DEFAULT));     // Use default algorithm
    }
    else if constexpr (std::is_same<T, __half>::value && std::is_same<U, float>::value){
      CUBLAS_CHECK(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_16F, m,         // A (FP16 matrix)
        d_B, CUDA_R_16F, k,         // B (FP16 matrix)
        &beta,
        d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
        CUBLAS_COMPUTE_32F,                // The compute type is FP32
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));     // Use default algorithm
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
  else if constexpr (std::is_same<T, __half>::value && std::is_same<U, float>::value){
    std::cout << "----fp16_fp16_fp32 warmup done" << std::endl;
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(start, stream));

  for(int i = 0; i < repeat; i++){
    if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, float>::value){
      CUBLAS_CHECK(cublasGemmEx(
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
      CUBLAS_CHECK(cublasGemmEx(
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
      CUBLAS_CHECK(cublasGemmEx(
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
      CUBLAS_CHECK(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_16F, m,         // A (fp16 matrix)
        d_B, CUDA_R_16F, k,         // B (fp16 matrix)
        &beta,
        d_C, CUDA_R_16F, m,        // C (fp16 output matrix)
        CUBLAS_COMPUTE_32F,                // The compute type is FP16
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));     // Use default algorithm
    }
    else if constexpr (std::is_same<T, __half>::value && std::is_same<U, float>::value){
      CUBLAS_CHECK(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, 
        d_A, CUDA_R_16F, m,         // A (fp16 matrix)
        d_B, CUDA_R_16F, k,         // B (fp16 matrix)
        &beta,
        d_C, CUDA_R_32F, m,        // C (fp32 output matrix)
        CUBLAS_COMPUTE_32F,                // The compute type is FP32
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));     // Use default algorithm
    }
  }

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
  else if constexpr (std::is_same<T, __half>::value && std::is_same<U, float>::value){
    std::cout << "----fp16_fp16_fp32 Time taken: " << milliseconds << " ms" << std::endl;
    std::cout << "----fp16_fp16_fp32 TFlops: " << (double)ops * 1e-12 / (milliseconds / repeat / 1000) << std::endl;
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
  int warmup = 10;
  int repeat = 100;
  if(argc >= 2){
    bits = std::stoi(argv[1]);
  }
  if(argc >= 3){
    warmup = std::stoi(argv[2]);
    repeat = warmup * 10;
  }
  if(argc >= 4){
    repeat = std::stoi(argv[3]);
  }
  //CHECK宏定义检查操作是否正常处理
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using Device %d: %s\n",dev,deviceProp.name);
  CUDA_CHECK(cudaSetDevice(dev));
  //set up data size of matrix
  int m = 1<<bits; //16384
  int k = 1<<bits; //16384
  int n = 1<<bits; //16384
  printf("============m: %d, n: %d, k: %d============\n", m, n, k);
  mm_cublas<int8_t, float>(warmup, repeat, m, n, k);
  mm_cublas<float, float>(warmup, repeat, m, n, k);
  mm_cublas<int8_t, int32_t>(warmup, repeat, m, n, k);
  mm_cublas<__half, __half>(warmup, repeat, m, n, k);
  mm_cublas<__half, float>(warmup, repeat, m, n, k);
	return 0;

}
