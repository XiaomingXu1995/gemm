#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
#include<cuda_fp16.h>

#include "sage_dir/qattn/qk_int_sv_f8_cuda_sm89.cuh"
#include "init.cuh"




int main() {
    // 设置输入参数
    int batch = 4;
    int head = 32;
    const int headdim= 128;
    int seq_len = 1024;

    int num_iters = 1;

    int seq_len_list[] = {1024, 2048, 4096, 8192, 16384, 32768};
    // int seq_len_list[] = {4096};

    const bool is_causal = false;
    printf("is_causal: %d\n", is_causal);
    const bool per_warp_quant = true;
    const bool per_thread_quant = false;

    constexpr int CTA_Q = 128;
    constexpr int CTA_K = 64;
    constexpr int WARP_Q = 32;
    constexpr int WARP_K = 64;
    constexpr int HEAD_DIM = headdim;



    for (int i = 0; i < 1; i++) {
        seq_len = seq_len_list[i];
        size_t flops = 4LL * head * batch * headdim * seq_len * seq_len;
        // printf("flops: %zu\n", flops);

        int8_t *q, *k, *v;
        __half * o;


        cudaMalloc(reinterpret_cast<void**>(&q), batch * seq_len * head * headdim * sizeof(int8_t));
        cudaMalloc(reinterpret_cast<void**>(&k), batch * seq_len * head * headdim * sizeof(int8_t));
        cudaMalloc(reinterpret_cast<void**>(&v), batch * headdim * head * seq_len * sizeof(int8_t));
        cudaMalloc(reinterpret_cast<void**>(&o), batch * seq_len * head * headdim * sizeof(__half));

        int8_t * tmp = (int8_t *)malloc(batch * seq_len * head * headdim * sizeof(int8_t));
        init_int8_array<int8_t>(tmp, batch * seq_len * head * headdim, -95, 95);
        cudaMemcpy(q, tmp, batch * seq_len * head * headdim * sizeof(int8_t), cudaMemcpyHostToDevice);
        init_int8_array<int8_t>(tmp, batch * seq_len * head * headdim, -95, 95);
        cudaMemcpy(k, tmp, batch * seq_len * head * headdim * sizeof(int8_t), cudaMemcpyHostToDevice);
        init_int8_array<int8_t>(tmp, batch * headdim * head * seq_len, -127, 127);
        cudaMemcpy(v, tmp, batch * headdim * head * seq_len * sizeof(int8_t), cudaMemcpyHostToDevice);
        
        float* q_scale, *k_scale;

        if(per_warp_quant) {
          cudaMalloc(reinterpret_cast<void**>(&q_scale), batch * head * seq_len / WARP_Q * sizeof(float));
          cudaMalloc(reinterpret_cast<void**>(&k_scale), batch * head * seq_len / WARP_K * sizeof(float));
        } else if(per_thread_quant) {
          cudaMalloc(reinterpret_cast<void**>(&q_scale), batch * head * seq_len / WARP_Q * 8 * sizeof(float));
          cudaMalloc(reinterpret_cast<void**>(&k_scale), batch * head * seq_len / WARP_K * 4 * sizeof(float));
        }

        const int qk_quant_gran = per_warp_quant ? 2 : 3;
        constexpr MaskMode mask_mode = is_causal ? MaskMode::kCausal : MaskMode::kNone;
        const bool return_lse = 0;

        float sm_scale = 1 / sqrt(headdim);
        size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t), CTA_Q * HEAD_DIM * sizeof(half));
        // printf("smem_max: %zu\n", smem_max);
        auto kernel_func = qk_int_sv_f8_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, static_cast<QuantGranularity>(qk_quant_gran), static_cast<QuantGranularity>(qk_quant_gran),
                                                        float, false, __half, ComputeUnit::kCudaCore, mask_mode, return_lse, false, false>;

        cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

        dim3 grid(div_ceil(seq_len, CTA_Q), head, batch);
        dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
        const int num_kv_groups = head / head;
        const int stride_bz_q = seq_len * head * headdim;
        const int stride_bz_k = seq_len * head * headdim;
        const int stride_bz_v = seq_len * head * headdim;
        const int stride_bz_o = seq_len * head * headdim;

        const int stride_seq_q = head * headdim;
        const int stride_h_q = headdim;

        const int stride_seq_k = head * headdim;
        const int stride_h_k = headdim;

        const int stride_h_v = headdim;
        const int stride_d_v = head * headdim;

        const int stride_seq_o = head * headdim;
        const int stride_h_o = headdim;
        // printf("stride_bz_q: %d\n", stride_bz_q);
        // printf("stride_seq_q: %d\n", stride_seq_q);
        // printf("stride_h_q: %d\n", stride_h_q);
        // printf("stride_bz_k: %d\n", stride_bz_k);
        // printf("stride_seq_k: %d\n", stride_seq_k);
        // printf("stride_h_k: %d\n", stride_h_k);
        // printf("stride_bz_v: %d\n", stride_bz_v);
        
      
      

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        

        for (int i = 0; i < num_iters; i++) {
            kernel_func<<<grid, block, smem_max>>>(
              q, 
              k,
              v,
              o,
              nullptr,
              q_scale,
              k_scale,
              nullptr,
              nullptr,
              seq_len,
              seq_len,
              num_kv_groups,
              stride_bz_q, stride_seq_q, stride_h_q,
              stride_bz_k, stride_seq_k, stride_h_k,
              stride_bz_v, stride_h_v, stride_d_v,
              stride_bz_o, stride_seq_o, stride_h_o,
              sm_scale);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Kernel launch failed: %s\n", cudaGetErrorString(err));
            }

        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float seconds = milliseconds / 1000;
        float avg_seconds = seconds / num_iters;
        // printf("Time taken: %f s\n", avg_seconds);
        printf("seq_len: %d, TFLOPS: %f\n", seq_len, flops / avg_seconds * 1e-12);

        __half *o_ref = (__half *)malloc(batch * seq_len * head * headdim * sizeof(__half));
        cudaMemcpy(o_ref, o, batch * seq_len * head * headdim * sizeof(__half), cudaMemcpyDeviceToHost);
        for(int i = 0; i < 10; i++){
          printf("o[%d]: %f\n", i, __half2float(o_ref[i]));
        }
        
  }
}




