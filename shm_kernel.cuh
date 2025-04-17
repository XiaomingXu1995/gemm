#ifndef SHM_KERNEL_CUH
#define SHM_KERNEL_CUH

#include <cuda_runtime.h>
#include <cassert>  

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void MatrixMulCUDA6( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int K,
    const int N,
    float alpha,
    float beta
    ) {
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = threadIdx.y * bszx + threadIdx.x;

    // shared memory

    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;

    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    const int A_S = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int B_S = BLOCK_SIZE_N / THREAD_SIZE_X;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {
        // load A from global memory to shared memory

        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                As[i + A_TILE_ROW ][A_TILE_COL] = row < M && col < K ? A[OFFSET(
                    row, // row
                    col, // col
                    K )] : 0;
            } else {
                As[i + A_TILE_ROW ][A_TILE_COL] = A[OFFSET(
                    row, // row
                    col, // col
                    K )];
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            const int row = tile_idx + i + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * blockIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                Bs[i + B_TILE_ROW][B_TILE_COL] = row < K && col < N ? B[OFFSET(
                    row, // row
                    col, // col
                    N )] : 0;
            } else {
                Bs[i + B_TILE_ROW][B_TILE_COL] = B[OFFSET(
                    row, // row
                    col, // col
                    N )];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    // accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                    accum[thread_y][thread_x] += As[thread_y * A_S + threadIdx.y][k] * Bs[k][thread_x * B_S + threadIdx.x];
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            const int row = BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y;
            const int col = BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET(row, col, N)] = C[OFFSET(row, col, N)] * beta + accum[thread_y][thread_x] * alpha;
                }
            } else {
                C[OFFSET(row, col, N)] = C[OFFSET(row, col, N)] * beta + accum[thread_y][thread_x] * alpha;
            }
        }
    }
}

template<
  const int BLOCK_SIZE_M, 
  const int BLOCK_SIZE_N, 
  const int BLOCK_SIZE_K,
  const int TILES_COMPUTE_ROW,
  const int TILES_COMPUTE_COL
  >
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

  __shared__ int8_t smem_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ int8_t smem_b[BLOCK_SIZE_K][BLOCK_SIZE_N];

  int32_t c_accum_1[TILES_COMPUTE_ROW][TILES_COMPUTE_COL] = {0};

  for (int k = 0; k < K; k += BLOCK_SIZE_K){

    // load A to shared memory smem_a
    # pragma unroll
    for (int i = 0; i < tiles_read_a_row; i++){
      # pragma unroll
      for (int j = 0; j < tiles_read_a_col; j++){
        const int row_a = blockIdx.y * BLOCK_SIZE_M + i * blockDim.y + threadIdx.y;
        const int col_a = k + j * blockDim.x + threadIdx.x;
        if (blockIdx.y == gridDim.y -1 || blockIdx.x == gridDim.x -1){
          smem_a[i * blockDim.y + threadIdx.y][j * blockDim.x + threadIdx.x] = (row_a < M && col_a < K) ? A[OFFSET(row_a, col_a, K)] : 0;
        }
        else{
          smem_a[i * blockDim.y + threadIdx.y][j * blockDim.x + threadIdx.x] = A[OFFSET(row_a, col_a, K)];
        }
      }
    }

    // load B to shared memory smem_b
    # pragma unroll
    for (int i = 0; i < tiles_read_b_row; i++){
      # pragma unroll
      for (int j = 0; j < tiles_read_b_col; j++){
        const int row_b = k + i * blockDim.y + threadIdx.y;
        const int col_b = blockIdx.x * BLOCK_SIZE_N + j * blockDim.x + threadIdx.x;
        if (blockIdx.y == gridDim.y -1 || blockIdx.x == gridDim.x -1){
          smem_b[i * blockDim.y + threadIdx.y][j * blockDim.x + threadIdx.x] = (row_b < K && col_b < N) ? B[OFFSET(row_b, col_b, N)] : 0;
        }
        else{
          smem_b[i * blockDim.y + threadIdx.y][j * blockDim.x + threadIdx.x] = B[OFFSET(row_b, col_b, N)];
        }
      }
    }

    __syncthreads();

    // do mma
    # pragma unroll
    for (int k = 0; k < BLOCK_SIZE_K; k++){
      # pragma unroll
      for(int i = 0; i < TILES_COMPUTE_ROW; i++){
        # pragma unroll
        for(int j = 0; j < TILES_COMPUTE_COL; j++){
          c_accum_1[i][j] += smem_a[i * blockDim.y + threadIdx.y][k] * smem_b[k][j * blockDim.x + threadIdx.x];
        }
      }
    }
    __syncthreads();

  }

  // store C to global memory
  # pragma unroll
  for(int i = 0; i < TILES_COMPUTE_ROW; i++){
    # pragma unroll
    for(int j = 0; j < TILES_COMPUTE_COL; j++){
      const int row_c = blockIdx.y * BLOCK_SIZE_M + i * blockDim.y + threadIdx.y;
      const int col_c = blockIdx.x * BLOCK_SIZE_N + j * blockDim.x + threadIdx.x;
      if (blockIdx.y == gridDim.y -1 || blockIdx.x == gridDim.x -1){
        if (row_c < M && col_c < N){
          C[OFFSET(row_c, col_c, N)] = c_accum_1[i][j];
        }
      }
      else{
        C[OFFSET(row_c, col_c, N)] = c_accum_1[i][j];
      }
    }
  }
}

#endif
