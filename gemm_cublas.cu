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

void initialData_int8(int8_t *ip,int size)
{
    int max_int8 = 127;
    srand(0);
    for(int i=0;i<size;i++)
    {
        ip[i] = rand() % max_int8;
    }
}
void initialData(data_type *ip,int size, int step)
{
    //generate different seed for random number
    // time_t t;
    // srand((unsigned)time(&t));
    srand(0);
    for(int i=0;i<size;i++)
    {
				data_type rand_num = rand() % step;
				ip[i] = rand_num / step / 1000;
    }
}

//hostRef传入CPU端计算的矩阵加法结果，gpuRef传入GPU端计算的矩阵加法结果
//对比争取输出"Arrats match"
void checkResult(data_type *hostRef,data_type *gpuRef,const int N)
{
    //double epsilon = 1.0E-8;
    //double epsilon = 1.0E-5;
    double epsilon = 1.0E-2;
		//double rate = 0.2;
    bool match=1;
    for(int i=0;i<N;i++)
    {
        if(abs(hostRef[i]-gpuRef[i])>epsilon)
        //if(abs(hostRef[i]-gpuRef[i]) > rate * abs(hostRef[i]))
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

double get_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
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

void mm_int8_int8_fp32(int warmup, int repeat, const int m, const int n, const int k){
  cout << "--------------------------------mm_int8_int8_fp32--------------------------------" << endl;
  long ops = 2L * m * n * k;
  cudaStream_t stream = NULL;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  int8_t *h_A = nullptr;
  int8_t *h_B = nullptr;
  float *h_C = nullptr;
  float *h_C_ref = nullptr;

  h_A = (int8_t *)malloc(m * k * sizeof(int8_t));
  h_B = (int8_t *)malloc(k * n * sizeof(int8_t));
  h_C = (float *)malloc(m * n * sizeof(float));
  h_C_ref = (float *)malloc(m * n * sizeof(float));
  initialData_int8(h_A, m * k);
  initialData_int8(h_B, k * n);
  memset(h_C_ref, 0, m * n * sizeof(float));
  matrixMulOnHost<int8_t, float>(h_A, h_B, h_C_ref, m, n, k);

  int8_t *d_A = nullptr;
  int8_t *d_B = nullptr;
  float *d_C = nullptr;

  cublasHandle_t cublasH = NULL;
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  const data_type alpha = 1.0;
  const data_type beta = 0.0;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(int8_t) * m * k));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(int8_t) * k * n));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * m * n));

  CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, sizeof(int8_t) * m * k, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, sizeof(int8_t) * k * n, cudaMemcpyHostToDevice, stream));

  CHECK_CUBLAS(cublasGemmEx(
    cublasH,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    m, n, k,
    &alpha, d_A, CUDA_R_8I, m,  // A (INT8 matrix)
    d_B, CUDA_R_8I, k,         // B (INT8 matrix)
    &beta,
    d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
    CUDA_R_32F,                // The compute type is FP32
    CUBLAS_GEMM_DEFAULT));     // Use default algorithm

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());

  checkResult(h_C_ref, h_C, m * n);

  for(int i = 0; i < warmup; i++){
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, 
        CUBLAS_OP_N, 
        m, n, k, 
        &alpha, d_A, CUDA_R_8I, m, 
        d_B, CUDA_R_8I, k, 
        &beta, 
        d_C, CUDA_R_32F, m, 
        CUDA_R_32F, 
        CUBLAS_GEMM_DEFAULT));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "----int8_int8_fp32 warmup done" << std::endl;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));

  for(int i = 0; i < repeat; i++){
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, d_A, CUDA_R_8I, m,
        d_B, CUDA_R_8I, k,
        &beta,
        d_C, CUDA_R_32F, m,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT));
  }

  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "----int8_int8_fp32 Time taken: " << milliseconds << " ms" << std::endl;
  std::cout << "----int8_int8_fp32 TFlops: " << (double)ops * 1e-12 / (milliseconds / repeat / 1000) << std::endl;


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
}

void mm_fp32_fp32_fp32(int warmup, int repeat, const int m, const int n, const int k){
  cout << "--------------------------------mm_fp32_fp32_fp32--------------------------------" << endl;
  long ops = 2L * m * n * k;
  cudaStream_t stream = NULL;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  float *h_A = nullptr;
  float *h_B = nullptr;
  float *h_C = nullptr;
  float *h_C_ref = nullptr;

  h_A = (float *)malloc(m * k * sizeof(float));
  h_B = (float *)malloc(k * n * sizeof(float));
  h_C = (float *)malloc(m * n * sizeof(float));
  h_C_ref = (float *)malloc(m * n * sizeof(float));
  initialData(h_A, m * k, 1000);
  initialData(h_B, k * n, 1000);
  memset(h_C_ref, 0, m * n * sizeof(float));
  matrixMulOnHost<float, float>(h_A, h_B, h_C_ref, m, n, k);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  cublasHandle_t cublasH = NULL;
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  const data_type alpha = 1.0;
  const data_type beta = 0.0;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * k));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * k * n));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * m * n));

  CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, sizeof(float) * m * k, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, sizeof(float) * k * n, cudaMemcpyHostToDevice, stream));

  CHECK_CUBLAS(cublasGemmEx(
    cublasH,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    m, n, k,
    &alpha, d_A, CUDA_R_32F, m,  // A (FP32 matrix)
    d_B, CUDA_R_32F, k,         // B (FP32 matrix)
    &beta,
    d_C, CUDA_R_32F, m,        // C (FP32 output matrix)
    CUDA_R_32F,                // The compute type is FP32
    CUBLAS_GEMM_DEFAULT));     // Use default algorithm

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());

  checkResult(h_C_ref, h_C, m * n);

  for(int i = 0; i < warmup; i++){
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, 
        CUBLAS_OP_N, 
        m, n, k, 
        &alpha, d_A, CUDA_R_32F, m, 
        d_B, CUDA_R_32F, k, 
        &beta, 
        d_C, CUDA_R_32F, m, 
        CUDA_R_32F, 
        CUBLAS_GEMM_DEFAULT));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "----fp32_fp32_fp32 warmup done" << std::endl;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));

  for(int i = 0; i < repeat; i++){
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha, d_A, CUDA_R_32F, m,
        d_B, CUDA_R_32F, k,
        &beta,
        d_C, CUDA_R_32F, m,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT));
  }

  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "----fp32_fp32_fp32 Time taken: " << milliseconds << " ms" << std::endl;
  std::cout << "----fp32_fp32_fp32 TFlops: " << (double)ops * 1e-12 / (milliseconds / repeat / 1000) << std::endl;


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
}


int main(int argc, char *argv[]) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  int bits = 10;
  if(argc >= 2){
    bits = std::stoi(argv[1]);
  }
  //CHECK宏定义检查操作是否正常处理
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using Device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));
  //set up data size of matrix
  int m = 1<<bits; //16384
  int k = 1<<bits; //16384
  int n = 1<<bits; //16384
  mm_int8_int8_fp32(10, 300, m, n, k);
  mm_fp32_fp32_fp32(10, 300, m, n, k);
	return 0;

}
