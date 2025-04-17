#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <assert.h>
#include <immintrin.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utils.cuh"
#include "init.cuh"
#include "check.cuh"
#include "mm_utils.cuh"

#include "mma.cuh"
#include "shm_kernel.cuh"

template<const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int TILES_COMPUTE_ROW, const int TILES_COMPUTE_COL>
void mm_mma_int8(int warmup, int repeat, int m, int n, int k){
  int8_t * d_A;
  int8_t * d_B;
  int32_t * d_C;
  int32_t * d_C_ref;
  int8_t * h_A;
  int8_t * h_B;
  int32_t * h_C;
  int32_t * h_C_ref;

  CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_C_ref, m * n * sizeof(int32_t)));

  CUDA_CHECK(cudaMallocHost(&h_A, m * k * sizeof(int8_t)));
  CUDA_CHECK(cudaMallocHost(&h_B, k * n * sizeof(int8_t)));
  CUDA_CHECK(cudaMallocHost(&h_C, m * n * sizeof(int32_t)));
  CUDA_CHECK(cudaMallocHost(&h_C_ref, m * n * sizeof(int32_t)));
  printf("initial data\n");
  initialData<int8_t>(h_A, m * k);
  initialData<int8_t>(h_B, k * n);
  CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(int8_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(int8_t), cudaMemcpyHostToDevice));

  printf("gemm_mma\n");
  dim3 blockDim(BLOCK_SIZE_N / TILES_COMPUTE_COL, BLOCK_SIZE_M / TILES_COMPUTE_ROW);
  dim3 gridDim((n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
  printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
  printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);

  gemm_mma<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, TILES_COMPUTE_ROW, TILES_COMPUTE_COL><<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);

  printf("matrixMulOnGPU\n");
  dim3 blockDim_ref(32, 32);
  dim3 gridDim_ref((n + blockDim_ref.x - 1) / blockDim_ref.x, (m + blockDim_ref.y - 1) / blockDim_ref.y);
  matrixMulOnGPU<int8_t, int32_t><<<gridDim_ref, blockDim_ref>>>(d_A, d_B, d_C_ref, m, n, k);

  CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost));
  checkResult<int32_t>(h_C_ref, h_C, m * n);

  for(int i = 0; i < warmup; i++){
    gemm_mma<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, TILES_COMPUTE_ROW, TILES_COMPUTE_COL><<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
  }
  printf("warmup done\n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for(int i = 0; i < repeat; i++){
    gemm_mma<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, TILES_COMPUTE_ROW, TILES_COMPUTE_COL><<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
  }
  cudaEventRecord(stop, 0); 
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("mma time: %f ms\n", milliseconds / 1000 / repeat);
  printf("mma gflops: %f\n", compute_gflops(m, n, k, milliseconds / 1000 / repeat));
  printf("mma tflops: %f\n", compute_tflops(m, n, k, milliseconds / 1000 / repeat));

} 

template<const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int TILES_COMPUTE_ROW, const int TILES_COMPUTE_COL>
void mm_mma_float(int warmup, int repeat, int m, int n, int k){
  float * d_A;
  float * d_B;
  float * d_C;
  float * d_C_ref;
  float * h_A;
  float * h_B;
  float * h_C;
  float * h_C_ref;

  CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C_ref, m * n * sizeof(float)));

  CUDA_CHECK(cudaMallocHost(&h_A, m * k * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_B, k * n * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_C, m * n * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_C_ref, m * n * sizeof(float)));
  printf("initial data\n");
  initialData<float>(h_A, m * k);
  initialData<float>(h_B, k * n);
  CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

  printf("MatrixMulCUDA6\n");
  dim3 blockDim(BLOCK_SIZE_N / TILES_COMPUTE_COL, BLOCK_SIZE_M / TILES_COMPUTE_ROW);
  dim3 gridDim((n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
  printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
  printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);

  MatrixMulCUDA6<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, TILES_COMPUTE_ROW, TILES_COMPUTE_COL, false><<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n, 1, 0);

  printf("matrixMulOnGPU\n");
  dim3 blockDim_ref(32, 32);
  dim3 gridDim_ref((n + blockDim_ref.x - 1) / blockDim_ref.x, (m + blockDim_ref.y - 1) / blockDim_ref.y);
  matrixMulOnGPU<float, float><<<gridDim_ref, blockDim_ref>>>(d_A, d_B, d_C_ref, m, n, k);

  CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, m * n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
  checkResult<float>(h_C_ref, h_C, m * n);

  for(int i = 0; i < warmup; i++){
    MatrixMulCUDA6<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, TILES_COMPUTE_ROW, TILES_COMPUTE_COL, false><<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n, 1, 0);
  }
  printf("warmup done float\n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for(int i = 0; i < repeat; i++){
    MatrixMulCUDA6<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, TILES_COMPUTE_ROW, TILES_COMPUTE_COL, false><<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n, 1, 0);
  }
  cudaEventRecord(stop, 0); 
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("float mma time: %f ms\n", milliseconds / 1000 / repeat);
  printf("float mma gflops: %f\n", compute_gflops(m, n, k, milliseconds / 1000 / repeat));
  printf("float mma tflops: %f\n", compute_tflops(m, n, k, milliseconds / 1000 / repeat));

} 







int main(int argc, char ** argv){
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
  printf("============warmup: %d, repeat: %d============\n", warmup, repeat);



  // allocate memory
  //mm_mma_int8<64, 64, 128, 4, 4>(warmup, repeat, m, n, k);
  mm_mma_int8<128, 64, 128, 8, 4>(warmup, repeat, m, n, k);
  //mm_mma_float<96, 32, 64, 6, 4>(warmup, repeat, m, n, k);
  //mm_mma_float<64, 64, 64, 4, 4>(warmup, repeat, m, n, k);

  // cudaFuncAttributes attr;
  // CUDA_CHECK(cudaFuncGetAttributes(&attr, gemm_mma<64, 64, 128, 4, 4>));  // 注意是 kernel 函数名本身（不加括号）
  // printf("shared memory size = %ld bytes\n", attr.sharedSizeBytes);
  // printf("shared memory size = %ld KB\n", attr.sharedSizeBytes / 1024);



}