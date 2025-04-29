#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
#include<cuda_fp16.h>

#include "init.cuh"
#include "utils.cuh"
#include "mm_utils.cuh"

// int8 tensor core

#define WARP_SIZE 32
#define PACK_SIZE 16 // as if it is int8, 128/8=16
#define PACK_SIZE_INT32 4 // as if it is int32, 128/32=4
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

#define div_ceil(M, N) (((M) + (N)-1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

using b128_t = uint4;


__device__ __forceinline__ uint32_t get_warp_id()
{
  return threadIdx.y;
}

__device__ __forceinline__ uint32_t get_lane_id()
{
  return threadIdx.x;
}

template <uint32_t stride>
__device__ __forceinline__ uint32_t get_permuted_offset(const uint32_t &i, const uint32_t &j) {
  return i * stride + j;
}


// smem offset on b128_t level
template <uint32_t step_size, uint32_t stride>
static __device__ __forceinline__ uint32_t advance_offset_by_row(const uint32_t &offset) {
  return offset + step_size * stride;
}
  template <uint32_t step_size>
static __device__ __forceinline__ uint32_t advance_offset_by_column(const uint32_t &offset) {
  return offset + step_size;
}

template <uint32_t step_size>
static __device__ __forceinline__ uint32_t advance_offset_by_column(const uint32_t &offset, const uint32_t &iter) {
  return offset + step_size;
}

template <typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
}

template <typename T>
__device__ __forceinline__ void load_128b_async(const uint32_t &offset, const T* gptr, const b128_t * smem_base_ptr) {
  b128_t* smem_ptr = const_cast<b128_t*>(smem_base_ptr) + offset;
  load_128b(smem_ptr, reinterpret_cast<const b128_t*>(gptr));
}

template <typename T>
__device__ __forceinline__ void store_128b(const uint32_t &offset, T* gptr, const b128_t * smem_base_ptr) {
  //static_assert(reinterpret_cast<uintptr_t>(gptr) % 16 == 0, "gptr must be 16-byte aligned"); // gptr必须16字节对齐
  *reinterpret_cast<b128_t*>(gptr) = *(smem_base_ptr + offset);
  // *(reinterpret_cast<int32_t*>(gptr)+ 0) = *(reinterpret_cast<const int32_t*>(smem_base_ptr + offset)+ 0); // out
  // *(reinterpret_cast<int32_t*>(gptr)+ 1) = *(reinterpret_cast<const int32_t*>(smem_base_ptr + offset)+ 1); 
  // *(reinterpret_cast<int32_t*>(gptr)+ 2) = *(reinterpret_cast<const int32_t*>(smem_base_ptr + offset)+ 2);
  // *(reinterpret_cast<int32_t*>(gptr)+ 3) = *(reinterpret_cast<const int32_t*>(smem_base_ptr + offset)+ 3);
}

__device__ __forceinline__ void commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <size_t n>
__device__ __forceinline__ void wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// from xxm: in global memory, each element is 8-bit, so the advance_offset of global memory need to multiply pack_size;
// in shared memory, each element is 128-bit, so the advance_offset of shared memory is global_to_shared_line_lanes on column and global_to_shared_copy_lines_per_warp_per_iter*stride on row;
template <uint32_t global_to_shared_line_lanes, uint32_t global_to_shared_copy_lines_per_warp_per_iter, 
          uint32_t smem_iters_row, uint32_t smem_iters_col, uint32_t stride, uint32_t CTA, typename T>
__device__ __forceinline__ void load_global_to_share(T **lane_ptr, uint32_t &smem_offset,
                                                    const uint32_t &gmem_stride,
                                                    const b128_t * smem_base_ptr)
{
  static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
  static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);
  constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++)
    {
      load_128b_async<T>(smem_offset, *lane_ptr, smem_base_ptr);
      *lane_ptr += (global_to_shared_line_lanes * pack_size);
      smem_offset = advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
    }

    smem_offset = advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter, stride>(smem_offset - smem_iters_row * global_to_shared_line_lanes);
    *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size));
  }
  smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  *lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

template <typename T>
__device__ __forceinline__ void mma_ldmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
             : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
             : "r"(smem_int_ptr));
}

__device__ __forceinline__ void ldmatrix_m8n8x4(const uint32_t &offset, uint32_t* R, const b128_t* smem_base_ptr) {
  b128_t* smem_ptr = const_cast<b128_t*>(smem_base_ptr) + offset;
  mma_ldmatrix_m8n8x4<b128_t>(R, smem_ptr);
}

__device__ __forceinline__ void mma_sync_m16n16k32_row_col_s8s8s32_init(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
        "r"(0), "r"(0));
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
        "r"(0), "r"(0));
}

__device__ __forceinline__ void mma_sync_m16n16k32_row_col_s8s8s32_update(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
        "r"(C[2]), "r"(C[3]));
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]),
        "r"(C[6]), "r"(C[7]));
}

template < uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_qk_inner, uint32_t stride>
__device__ __forceinline__ void compute_int_qk(const b128_t *smem_Q, const b128_t *smem_K, int32_t RS[][num_tiles_k][8], uint32_t &offset_Q, uint32_t &offset_K)
{
  uint32_t RQ[num_tiles_q][4];
  uint32_t RK[4];

  // the first iteration, mma mode is kInit
#pragma unroll
  for (uint32_t iter = 0; iter < 1; iter++)
  {
    // load RQ
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      ldmatrix_m8n8x4(offset_Q, RQ[fq], smem_Q);

      // each row in shared memory has 128 Bytes, and each element is 128 bits (b128_t *smem_base_ptr), so the stride = 128B/128-bit = 8
      // ldmatrix_m8n8x4 load 8x8x4 16-bit elements, equal to 16x32 int8 elements (for mma::m16n16k32). 16x32 int8 is equal to 16x2 128-bit, so advance_offse_by_col<2>. //assert stride == 2 * num_tiles_qk_inner  
      // so offset_Q need to advance 16 rows.
      offset_Q = advance_offset_by_row<16, stride>(offset_Q); 
    }
    // ! using permutation invariance
    offset_Q = advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      // load RK
      //smem_K.ldmatrix_m8n8x4(offset_K, RK);
      ldmatrix_m8n8x4(offset_K, RK, smem_K);
      offset_K = advance_offset_by_row<16, stride>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        mma_sync_m16n16k32_row_col_s8s8s32_init(RS[fq][fk], RQ[fq], RK);
      }
    }
    offset_K = advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

  // following iteration, mma mode is kInplace
#pragma unroll
  for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++)
  {
    // load RQ
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      ldmatrix_m8n8x4(offset_Q, RQ[fq], smem_Q);
      offset_Q = advance_offset_by_row<16, stride>(offset_Q);
    }
    offset_Q = advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      // load RK
      ldmatrix_m8n8x4(offset_K, RK, smem_K);
      offset_K = advance_offset_by_row<16, stride>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        mma_sync_m16n16k32_row_col_s8s8s32_update(RS[fq][fk], RQ[fq], RK);
      }
    }
    offset_K = advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

  offset_Q -= (2 * num_tiles_qk_inner);
  offset_K -= (2 * num_tiles_qk_inner);
}



// (M,K) * (K,N) = (M,N) A * B = C
template<int CTA_Q, int CTA_K, int WARP_Q, int WARP_K, int HEAD_DIM>
__global__ void gemm_mma_kernel(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K) {
  // if(blockIdx.x == 0 && threadIdx.x == 0) {
  //   printf("CTA_Q: %d, CTA_K: %d, WARP_Q: %d, WARP_K: %d, HEAD_DIM: %d\n", CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM);
  // }
  static_assert(HEAD_DIM == 128, "k (headdim) must be 128");
  constexpr int num_warps_q = CTA_Q / WARP_Q; // 128/32 = 4
  constexpr int num_warps_k = CTA_K / WARP_K; // 64/64 = 1
  constexpr int num_warps = num_warps_q * num_warps_k;
  constexpr int num_tiles_q = WARP_Q / MMA_QK_M; // 32/16 = 2
  constexpr int num_tiles_k = WARP_K / MMA_QK_N; // 64/16 = 4
  constexpr int num_tiles_qk_inner =  HEAD_DIM / MMA_QK_K; // 128/32 = 4
  // constexpr int num_tiles_qk_inner = (DTypeQK == DataType::kInt8) ? (head_dim / MMA_QK_K) : (head_dim / 2 / MMA_QK_K);

  // constexpr size_t smem_size = CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_Q * CTA_K * sizeof(int32_t);
  // __shared__ int8_t smem[smem_size];
  extern __shared__ int8_t smem[];

  const uint32_t bx = blockIdx.x;
  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // RS for the register result of int32 accumulator
  int32_t RS[num_tiles_q][num_tiles_k][8];

  constexpr uint32_t global_to_shared_line_lanes = 8; // HEAD_DIM / PACK_SIZE = 128 / 16 = 8
  constexpr uint32_t global_to_shared_copy_lines_per_warp = 4; // WARP_Q / global_to_shared_line_lanes = 32 / 8 = 4
  constexpr uint32_t smem_stride = HEAD_DIM / PACK_SIZE; // 128 / 16 = 8
  constexpr uint32_t AB_smem_iters_row = HEAD_DIM / (global_to_shared_line_lanes * PACK_SIZE);
  constexpr uint32_t A_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp);
  constexpr uint32_t B_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp);
  
  // C_smem is [128, 64] uint32_t
  constexpr uint32_t global_to_shared_line_lanes_C = CTA_K / PACK_SIZE_INT32; // 64 / 4 = 16
  constexpr uint32_t global_to_shared_copy_lines_per_warp_C = WARP_SIZE / global_to_shared_line_lanes_C; // 32 / 16 = 2
  constexpr uint32_t smem_stride_C = CTA_K / PACK_SIZE_INT32; // 64 / 4 = 16
  constexpr uint32_t C_smem_iters_row = CTA_K / (global_to_shared_line_lanes_C * PACK_SIZE_INT32); // 64 / (16 * 4) = 1
  constexpr uint32_t C_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_C); // 128 / (4 * 2) = 16

  b128_t * smem_A = reinterpret_cast<b128_t *>(smem);
  b128_t * smem_B = reinterpret_cast<b128_t *>(smem + CTA_Q * K);
  b128_t * smem_C = reinterpret_cast<b128_t *>(smem + CTA_Q * K + CTA_K * K);

  int8_t * A_lane_base_ptr = A + bx * CTA_Q * HEAD_DIM + warp_id * WARP_Q * HEAD_DIM + (lane_id / global_to_shared_line_lanes) * HEAD_DIM + (lane_id % global_to_shared_line_lanes) * PACK_SIZE;
  int8_t * B_lane_base_ptr = B + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes) * HEAD_DIM + (lane_id % global_to_shared_line_lanes) * PACK_SIZE;
  int32_t * C_lane_base_ptr = C + bx * CTA_Q * N + warp_id * WARP_Q * N + (lane_id / global_to_shared_line_lanes_C) * N;

  uint32_t A_smem_offset_load = get_permuted_offset<smem_stride>(warp_id * global_to_shared_copy_lines_per_warp * A_smem_iters_col + lane_id / global_to_shared_line_lanes, lane_id % global_to_shared_line_lanes);
  uint32_t B_smem_offset_load = get_permuted_offset<smem_stride>(warp_id * global_to_shared_copy_lines_per_warp * B_smem_iters_col + lane_id / global_to_shared_line_lanes, lane_id % global_to_shared_line_lanes);
  uint32_t C_smem_offset_store = get_permuted_offset<smem_stride_C>(warp_id * global_to_shared_copy_lines_per_warp_C * C_smem_iters_col + lane_id / global_to_shared_line_lanes_C, lane_id % global_to_shared_line_lanes_C);

  const uint32_t num_iterations = div_ceil(N,CTA_K);

  uint32_t A_smem_offset_mma = get_permuted_offset<smem_stride>(warp_id * global_to_shared_copy_lines_per_warp * A_smem_iters_col + lane_id / global_to_shared_line_lanes, lane_id % global_to_shared_line_lanes);
  uint32_t B_smem_offset_mma = get_permuted_offset<smem_stride>(warp_id * global_to_shared_copy_lines_per_warp * B_smem_iters_col + lane_id / global_to_shared_line_lanes, lane_id % global_to_shared_line_lanes);

  //TODO A_smem_offset_mma, B_smem_offset_mma

  // load A
  load_global_to_share<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, AB_smem_iters_row, A_smem_iters_col, smem_stride, CTA_Q, int8_t>(&A_lane_base_ptr, A_smem_offset_load, K, smem_A);
  commit_group();

  //uint32_t RQ[num_tiles_q][4];  

  // load B
  // load_global_to_share<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, AB_smem_iters_row, B_smem_iters_col, smem_stride, CTA_K, int8_t>(&B_lane_base_ptr, B_smem_offset_load, K, smem_B);
  // commit_group();
  // wait_group<0>();
  // _syncthreads();

  // TODO: mma compute
  #pragma unroll
  for (uint32_t iter = 0; iter < num_iterations; iter++)
  {
    // load B without predicate
    load_global_to_share<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, AB_smem_iters_row, B_smem_iters_col, smem_stride, CTA_K, int8_t>(
    &B_lane_base_ptr, B_smem_offset_load, K, smem_B);
    commit_group();
  
    // ensure K is ready
    wait_group<0>();
    __syncthreads();

    // compute QK^T
    compute_int_qk<num_tiles_q, num_tiles_k, num_tiles_qk_inner, smem_stride>(
        reinterpret_cast<const b128_t*>(smem_A), reinterpret_cast<const b128_t*>(smem_B), RS, A_smem_offset_mma, B_smem_offset_mma);

    //K_idx_lane_base += CTA_K;

    // if(threadIdx.x == 0 && blockIdx.x == 0 && iter == 0 && warp_id == 0){
    //   printf("RS[0][0][0]: %d, RS[0][0][1]: %d, RS[0][0][2]: %d, RS[0][0][3]: %d\n", RS[0][0][0], RS[0][0][1], RS[0][0][2], RS[0][0][3]);
    //   printf("RS[0][1][0]: %d, RS[0][1][1]: %d, RS[0][1][2]: %d, RS[0][1][3]: %d\n", RS[0][1][0], RS[0][1][1], RS[0][1][2], RS[0][1][3]);
    //   printf("RS[1][0][0]: %d, RS[1][0][1]: %d, RS[1][0][2]: %d, RS[1][0][3]: %d\n", RS[1][0][0], RS[1][0][1], RS[1][0][2], RS[1][0][3]);
    //   printf("RS[1][1][0]: %d, RS[1][1][1]: %d, RS[1][1][2]: %d, RS[1][1][3]: %d\n", RS[1][1][0], RS[1][1][1], RS[1][1][2], RS[1][1][3]);
    // }

    // store C from the register result to shared memory
    // int32_t RS[num_tiles_q][num_tiles_k][8]; 
    //num_tiles_q = 2, num_tiles_k = 4, A[32, 128] x B[128, 64] = C[32, 64] = 2 x 4 x [16, 16](m16n16k32) 
    //[16, 16] = 32 x 8 (32 threads per warp, 8 elements per thread)

    // the elements is in int32_t format

    #pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      #pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        uint32_t C_smem_row = warp_id * (num_tiles_q * MMA_QK_M) + fq * MMA_QK_M + lane_id / 4;
        uint32_t C_smem_col = fk * MMA_QK_N + (lane_id % 4) * 2;
        // (uint32_t*)smem_C[C_smem_row][C_smem_col] = RS[fq][fk][0];
        *((uint32_t*)smem_C + OFFSET(C_smem_row, C_smem_col, MMA_QK_N * num_tiles_k)) = RS[fq][fk][0];
        *((uint32_t*)smem_C + OFFSET(C_smem_row, C_smem_col + 1, MMA_QK_N * num_tiles_k)) = RS[fq][fk][1];
        *((uint32_t*)smem_C + OFFSET(C_smem_row + 8, C_smem_col, MMA_QK_N * num_tiles_k)) = RS[fq][fk][2];
        *((uint32_t*)smem_C + OFFSET(C_smem_row + 8, C_smem_col + 1, MMA_QK_N * num_tiles_k)) = RS[fq][fk][3];

        *((uint32_t*)smem_C + OFFSET(C_smem_row, C_smem_col + 8, MMA_QK_N * num_tiles_k)) = RS[fq][fk][4];
        *((uint32_t*)smem_C + OFFSET(C_smem_row, C_smem_col + 8 + 1, MMA_QK_N * num_tiles_k)) = RS[fq][fk][5];
        *((uint32_t*)smem_C + OFFSET(C_smem_row + 8, C_smem_col + 8, MMA_QK_N * num_tiles_k)) = RS[fq][fk][6];
        *((uint32_t*)smem_C + OFFSET(C_smem_row + 8, C_smem_col + 8 + 1, MMA_QK_N * num_tiles_k)) = RS[fq][fk][7];
      }
    }
    __syncwarp();

    // store C from shared memory to global memory
    // int8_t * A_lane_base_ptr = A + bx * CTA_Q * HEAD_DIM + warp_id * WARP_Q * HEAD_DIM + (lane_id / global_to_shared_line_lanes) * HEAD_DIM + (lane_id % global_to_shared_line_lanes) * PACK_SIZE;
    // int8_t * B_lane_base_ptr = B + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes) * HEAD_DIM + (lane_id % global_to_shared_line_lanes) * PACK_SIZE;
    //int32_t * C_lane_base_ptr = C + bx * CTA_Q * N + warp_id * WARP_Q * N + (lane_id / global_to_shared_line_lanes_C) * N + (iter * WARP_K + lane_id % global_to_shared_line_lanes_C) * PACK_SIZE_INT32;
    //int32_t C_smem_offset_store = get_permuted_offset<smem_stride_C>(warp_id * global_to_shared_copy_lines_per_warp_C * C_smem_iters_col + lane_id / global_to_shared_line_lanes_C, lane_id % global_to_shared_line_lanes_C);
    C_lane_base_ptr += (iter * WARP_K + lane_id % global_to_shared_line_lanes_C) * PACK_SIZE_INT32;

    #pragma unroll
    for (uint32_t inner_row = 0; inner_row < C_smem_iters_col; inner_row++)
    {
      #pragma unroll
      for (uint32_t inner_col = 0; inner_col < C_smem_iters_row; inner_col++)
      {
        store_128b(C_smem_offset_store, C_lane_base_ptr, smem_C);
        C_lane_base_ptr += global_to_shared_line_lanes_C * PACK_SIZE_INT32;
        C_smem_offset_store = advance_offset_by_column<smem_stride_C>(C_smem_offset_store);
      }
      C_smem_offset_store = advance_offset_by_row<global_to_shared_copy_lines_per_warp_C, smem_stride_C>(C_smem_offset_store - C_smem_iters_row * global_to_shared_line_lanes_C);
      C_lane_base_ptr += ((global_to_shared_copy_lines_per_warp_C * N) - (C_smem_iters_row * global_to_shared_line_lanes_C * PACK_SIZE_INT32));
    }
    C_smem_offset_store -= (C_smem_iters_col * global_to_shared_copy_lines_per_warp_C * smem_stride_C);
    C_lane_base_ptr += (CTA_Q - C_smem_iters_col * global_to_shared_copy_lines_per_warp_C) * N;

    __syncthreads();

    // B_smem_offset_load += CTA_K;

  }

  // store C from the register result to shared memory
  // int32_t RS[num_tiles_q][num_tiles_k][8]; 
  //num_tiles_q = 2, num_tiles_k = 4, A[32, 128] x B[128, 64] = C[32, 64] = 2 x 4 x [16, 16](m16n16k32) 
  //[16, 16] = 32 x 8 (32 threads per warp, 8 elements per thread)



}



int main() {
  int num_warmup = 5;
  int num_repeat = 20;

  int seq_len = 4096;
  const int headdim = 128;
  int m = seq_len;
  int n = seq_len; 
  int k = headdim;

  constexpr int CTA_Q = 64;
  constexpr int CTA_K = 64;
  constexpr int WARP_Q = 32;
  constexpr int WARP_K = 64;
  constexpr int HEAD_DIM = headdim;

  int8_t *a, *b;
  int32_t *c;
  int32_t *c_ref;

  long long int flops = 2LL * m * n * k;
  printf("flops Number: %lld\n", flops);

  //cudaMalloc(reinterpret_cast<void**>(&q), batch * seq_len * head * headdim * sizeof(int8_t));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a), m * k * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b), k * n * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c), m * n * sizeof(int32_t)));
  c_ref = (int32_t *)malloc(m * n * sizeof(int32_t));
  memset(c_ref, 0, m * n * sizeof(int32_t));
  CUDA_CHECK(cudaMemcpy(c, c_ref, m * n * sizeof(int32_t), cudaMemcpyHostToDevice));

  int8_t *a_ref = (int8_t *)malloc(m * k * sizeof(int8_t));
  init_int8_array<int8_t>(a_ref, m * k, -95, 95);
  CUDA_CHECK(cudaMemcpy(a, a_ref, m * k * sizeof(int8_t), cudaMemcpyHostToDevice));
  int8_t *b_ref = (int8_t *)malloc(k * n * sizeof(int8_t));
  init_int8_array<int8_t>(b_ref, k * n, -95, 95);
  CUDA_CHECK(cudaMemcpy(b, b_ref, k * n * sizeof(int8_t), cudaMemcpyHostToDevice));

  matrixMulOnHost<int8_t, int32_t>(a_ref, b_ref, c_ref, m, n, k);
  for(int i = 0; i < 10; i++){
    printf("c_ref[%d]: %d\n", i*seq_len, c_ref[i*seq_len]);
  }


  //size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t), CTA_Q * HEAD_DIM * sizeof(half));
  size_t smem_size = CTA_Q * k * sizeof(int8_t) + CTA_K * k * sizeof(int8_t) + CTA_Q * CTA_K * sizeof(int32_t);
  auto kernel_func = gemm_mma_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  dim3 block(32, (CTA_Q/WARP_Q) * (CTA_K/WARP_K));
  dim3 grid(div_ceil(m, CTA_Q));
  printf("block: %d, %d, grid: %d, %d\n", block.x, block.y, grid.x, grid.y);

  for(int i = 0; i < num_warmup; i++) {
    kernel_func<<<grid, block, smem_size>>>(a, b, c, m, n, k);
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));
  for(int i = 0; i < num_repeat; i++) {
    kernel_func<<<grid, block, smem_size>>>(a, b, c, m, n, k);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float time_mma;
  CUDA_CHECK(cudaEventElapsedTime(&time_mma, start, stop));
  time_mma /= num_repeat;
  time_mma /= 1000;
  printf("mma time: %f s\n", time_mma);
  //printf("mma flops: %lld\n", flops);
  printf("mma gflops: %f\n", flops / time_mma / 1e12);

  CUDA_CHECK(cudaMemcpy(c_ref, c, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost));
  for(int i = 0; i < 10; i++){
    printf("c_ref[%d]: %d\n", i*seq_len, c_ref[i*seq_len]);
  }

  return 0;





}