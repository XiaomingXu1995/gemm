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


// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template<
  const int BLOCK_SIZE_M, 
  const int BLOCK_SIZE_N, 
  const int BLOCK_SIZE_K >
__global__ void gemm_mma(
  int8_t * __restrict__ A, 
  int8_t * __restrict__ B, 
  int32_t * __restrict__ C, 
  const int M, 
  const int N, 
  const int K) {

  // for loading A and B to shared memory 
  assert(BLOCK_SIZE_M % blockDim.y == 0);
  assert(BLOCK_SIZE_K % blockDim.x == 0);
  assert(BLOCK_SIZE_K % blockDim.y == 0);
  assert(BLOCK_SIZE_N % blockDim.x == 0);
  const int tiles_read_a_row = BLOCK_SIZE_M / blockDim.y;
  const int tiles_read_a_col = BLOCK_SIZE_K / blockDim.x;
  const int tiles_read_b_row = BLOCK_SIZE_K / blockDim.y;
  const int tiles_read_b_col = BLOCK_SIZE_N / blockDim.x;
  const int tiles_compute_row = BLOCK_SIZE_M / blockDim.y;
  const int tiles_compute_col = BLOCK_SIZE_N / blockDim.x;

  __shared__ int8_t smem_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ int8_t smem_b[BLOCK_SIZE_K][BLOCK_SIZE_N];

  int32_t c_accum[BLOCK_SIZE_M][BLOCK_SIZE_N] = {0};

  for (int k = 0; k < K; k += BLOCK_SIZE_K){

    // load A to shared memory smem_a
    # pragma unroll
    for (int i = 0; i < tiles_read_a_row; i++){
      # pragma unroll
      for (int j = 0; j < tiles_read_a_col; j++){
        const int inner_row = i * blockDim.y + threadIdx.y;
        const int inner_col = j * blockDim.x + threadIdx.x;
        const int row_a = blockIdx.y * BLOCK_SIZE_M + inner_row;
        const int col_a = k + inner_col;
        if (blockIdx.y == gridDim.y -1 || blockIdx.x == gridDim.x -1){
          smem_a[inner_row][inner_col] = (row_a < M && col_a < K) ? A[OFFSET(row_a, col_a, K)] : 0;
        }
        else{
          smem_a[inner_row][inner_col] = A[OFFSET(row_a, col_a, K)];
        }
      }
    }

    // load B to shared memory smem_b
    # pragma unroll
    for (int i = 0; i < tiles_read_b_row; i++){
      # pragma unroll
      for (int j = 0; j < tiles_read_b_col; j++){
        const int inner_row = i * blockDim.y + threadIdx.y;
        const int inner_col = j * blockDim.x + threadIdx.x;
        const int row_b = k + inner_row;
        const int col_b = blockIdx.x * BLOCK_SIZE_N + inner_col;
        if (blockIdx.y == gridDim.y -1 || blockIdx.x == gridDim.x -1){
          smem_b[inner_row][inner_col] = (row_b < K && col_b < N) ? B[OFFSET(row_b, col_b, N)] : 0;
        }
        else{
          smem_b[inner_row][inner_col] = B[OFFSET(row_b, col_b, N)];
        }
      }
    }

    __syncthreads();

    // do mma
    # pragma unroll
    for (int k = 0; k < BLOCK_SIZE_K; k++){
      # pragma unroll
      for(int i = 0; i < tiles_compute_row; i++){
        # pragma unroll
        for(int j = 0; j < tiles_compute_col; j++){
          c_accum[i * blockDim.y + threadIdx.y][j * blockDim.x + threadIdx.x] += smem_a[i * blockDim.y + threadIdx.y][k] * smem_b[k][j * blockDim.x + threadIdx.x];
        }
      }
    }
    __syncthreads();

  }

  // store C to global memory
  # pragma unroll
  for(int i = 0; i < tiles_compute_row; i++){
    # pragma unroll
    for(int j = 0; j < tiles_compute_col; j++){
      const int row_c = blockIdx.y * BLOCK_SIZE_M + i * blockDim.y + threadIdx.y;
      const int col_c = blockIdx.x * BLOCK_SIZE_N + j * blockDim.x + threadIdx.x;
      if (row_c < M && col_c < N){
        C[OFFSET(row_c, col_c, N)] = c_accum[i * blockDim.y + threadIdx.y][j * blockDim.x + threadIdx.x];
      }
    }
  }
}

template<const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K>
void mm_mma(int warmup, int repeat, int m, int n, int k){
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
  dim3 blockDim(32, 32);
  dim3 gridDim((n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
  gemm_mma<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K><<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);

  printf("matrixMulOnGPU\n");
  dim3 blockDim_ref(32, 32);
  dim3 gridDim_ref((n + blockDim_ref.x - 1) / blockDim_ref.x, (m + blockDim_ref.y - 1) / blockDim_ref.y);
  matrixMulOnGPU<int8_t, int32_t><<<gridDim_ref, blockDim_ref>>>(d_A, d_B, d_C_ref, m, n, k);

  CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost));
  checkResult<int32_t>(h_C, h_C_ref, m * n);

  for(int i = 0; i < warmup; i++){
    gemm_mma<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K><<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
  }
  printf("warmup done\n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for(int i = 0; i < repeat; i++){
    gemm_mma<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K><<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
  }
  cudaEventRecord(stop, 0); 
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("mma time: %f ms\n", milliseconds / 1000 / repeat);
  printf("mma gflops: %f\n", compute_gflops(m, n, k, milliseconds / 1000 / repeat));
  printf("mma tflops: %f\n", compute_tflops(m, n, k, milliseconds / 1000 / repeat));

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
  mm_mma<64, 64, 128>(warmup, repeat, m, n, k);

  cudaFuncAttributes attr;
  CUDA_CHECK(cudaFuncGetAttributes(&attr, gemm_mma<64, 64, 128>));  // 注意是 kernel 函数名本身（不加括号）
  printf("shared memory size = %ld bytes\n", attr.sharedSizeBytes);
  printf("shared memory size = %ld KB\n", attr.sharedSizeBytes / 1024);



}