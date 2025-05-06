#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
#include<cuda_fp16.h>

#include "sage_dir/qattn/qk_int_sv_f8_cuda_sm89.cuh"
#include "init.cuh"

//#define O_TYPE __nv_bfloat16
#define O_TYPE __half

#define div_ceil(a, b) ((a + b - 1) / b)

template<typename T>
void fread_tensor(T * buf, size_t num_elements, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    size_t read_elements = fread(buf, sizeof(T), num_elements, fp);
    if (read_elements != num_elements) {
        std::cerr << "Error reading file: " << filename << std::endl;
        std::cerr << "Expected: " << num_elements << ", but got: " << read_elements << std::endl;
    }
    fclose(fp);
}

template<typename T>
void fwrite_tensor(const T* buf, size_t num_elements, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    size_t written_elements = fwrite(buf, sizeof(T), num_elements, fp);
    if (written_elements != num_elements) {
        std::cerr << "Error writing file: " << filename << std::endl;
        std::cerr << "Expected: " << num_elements << ", but got: " << written_elements << std::endl;
    }
    fclose(fp);
}



int main() {
    // 设置输入参数

    // int batch = 4;
    // int head = 32;
    // const int headdim= 128;
    // int seq_len = 1024;

    int batch = 1;
    int head = 24;
    const int headdim= 64;
    int seq_len = 1105;

    int num_iters = 1;

    int seq_len_list[] = {1024, 2048, 4096, 8192, 16384, 32768};
    // int seq_len_list[] = {4096};

    const bool is_causal = false;
    printf("is_causal: %d\n", is_causal);
    const bool per_warp_quant = true;
    const bool per_thread_quant = false;

    constexpr int CTA_Q = 128;
    constexpr int CTA_K = 64;
    constexpr int BLKQ = 128;
    constexpr int BLKK = 64;
    constexpr int WARP_Q = 32;
    constexpr int WARP_K = 64;
    constexpr int HEAD_DIM = headdim;



    for (int i = 0; i < 1; i++) {
        seq_len = seq_len_list[i];
        seq_len = 1105;
        size_t flops = 4LL * head * batch * headdim * seq_len * seq_len;
        // printf("flops: %zu\n", flops);

        int8_t *q, *k, *v;
        O_TYPE * o;

        int qk_size = batch * head * seq_len * headdim;
        int v_size = batch * head * headdim * ((seq_len + 63) / 64 * 64);


        cudaMalloc(reinterpret_cast<void**>(&q), qk_size * sizeof(int8_t));
        cudaMalloc(reinterpret_cast<void**>(&k), qk_size * sizeof(int8_t));
        cudaMalloc(reinterpret_cast<void**>(&v), v_size * sizeof(int8_t));
        cudaMalloc(reinterpret_cast<void**>(&o), qk_size * sizeof(O_TYPE));

        int8_t * tmp_q = (int8_t *)malloc(qk_size * sizeof(int8_t));
        int8_t * tmp_k = (int8_t *)malloc(qk_size * sizeof(int8_t));
        uint8_t * tmp_v = (uint8_t *)malloc(v_size * sizeof(uint8_t));

        fread_tensor<int8_t>(tmp_q, qk_size, "/root/xxm/data/q_int8.bin");
        fread_tensor<int8_t>(tmp_k, qk_size, "/root/xxm/data/k_int8.bin");
        fread_tensor<uint8_t>(tmp_v, v_size, "/root/xxm/data/v_int8.bin");

        cudaMemcpy(q, tmp_q, qk_size * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(k, tmp_k, qk_size * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(v, tmp_v, v_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

        //init_int8_array<int8_t>(tmp, batch * seq_len * head * headdim, -127, 127);

        // for(int i = 0; i < qk_size; i++){
        //   printf("q[%d]: %d\n", i, tmp[i]);
        // }


        // for(int i = 0; i < qk_size; i++){
        //   printf("k[%d]: %d\n", i, tmp[i]);
        // }

        // for(int i = 0; i < v_size; i++){
        //   printf("v[%d]: %d\n", i, tmp_v[i]);
        // }
        
        int q_scale_size = batch * head * ((seq_len + BLKQ - 1) / BLKQ) * (BLKQ / WARP_Q);
        int k_scale_size = batch * head * ((seq_len + BLKK - 1) / BLKK);
        int v_scale_size = batch * head * headdim;
        // printf("batch: %d\n", batch);
        // printf("head: %d\n", head);
        // printf("seq_len: %d\n", seq_len);
        // printf("BLKQ: %d\n", BLKQ); 
        // printf("WARP_Q: %d\n", WARP_Q);
        // printf("BLKK: %d\n", BLKK);
        // printf("q_scale_size: %d\n", q_scale_size);
        // printf("k_scale_size: %d\n", k_scale_size);

        // printf("the size of q_scale: %zu\n", batch * head * (seq_len + BLKQ - 1) / BLKQ * (BLKQ / WARP_Q) );
        float * q_scale_ref = (float *)malloc(q_scale_size * sizeof(float));
        float * k_scale_ref = (float *)malloc(k_scale_size * sizeof(float));
        float * v_scale_ref = (float *)malloc(v_scale_size * sizeof(float));
        fread_tensor<float>(q_scale_ref, q_scale_size, "/root/xxm/data/q_scale.bin");
        fread_tensor<float>(k_scale_ref, k_scale_size, "/root/xxm/data/k_scale.bin");
        fread_tensor<float>(v_scale_ref, v_scale_size, "/root/xxm/data/v_scale.bin");

        // for(int i = 0; i < q_scale_size; i++) {
        //     printf("q_scale[%d]: %f\n", i, q_scale_ref[i]);
        // }
        // for(int i = 0; i < k_scale_size; i++) {
        //     printf("k_scale[%d]: %f\n", i, k_scale_ref[i]);
        // }
        // for(int i = 0; i < v_scale_size; i++) {
        //     printf("v_scale[%d]: %f\n", i, v_scale_ref[i]);
        // }

        float *q_scale, *k_scale, *v_scale;
        if(per_warp_quant) {
          // cudaMalloc(reinterpret_cast<void**>(&q_scale), batch * head * seq_len / WARP_Q * sizeof(float));
          // cudaMalloc(reinterpret_cast<void**>(&k_scale), batch * head * seq_len / WARP_K * sizeof(float));
          cudaMalloc(reinterpret_cast<void**>(&q_scale), q_scale_size * sizeof(float));
          cudaMalloc(reinterpret_cast<void**>(&k_scale), k_scale_size * sizeof(float));
        } else if(per_thread_quant) {
          cudaMalloc(reinterpret_cast<void**>(&q_scale), batch * head * seq_len / WARP_Q * 8 * sizeof(float));
          cudaMalloc(reinterpret_cast<void**>(&k_scale), batch * head * seq_len / WARP_K * 4 * sizeof(float));
        }
        cudaMalloc(reinterpret_cast<void**>(&v_scale), v_scale_size * sizeof(float));
        cudaMemcpy(q_scale, q_scale_ref, q_scale_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(k_scale, k_scale_ref, k_scale_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(v_scale, v_scale_ref, v_scale_size * sizeof(float), cudaMemcpyHostToDevice);

        const int qk_quant_gran = per_warp_quant ? 2 : 3;
        constexpr MaskMode mask_mode = is_causal ? MaskMode::kCausal : MaskMode::kNone;
        const bool return_lse = 0;

        float sm_scale = 1 / sqrt(headdim);
        size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t), CTA_Q * HEAD_DIM * sizeof(half));
        // printf("smem_max: %zu\n", smem_max);
        // auto kernel_func = qk_int_sv_f8_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, static_cast<QuantGranularity>(qk_quant_gran), static_cast<QuantGranularity>(qk_quant_gran),
        //                                                 float, false, __nv_bfloat16, ComputeUnit::kCudaCore, mask_mode, return_lse, false, false>;
        auto kernel_func = qk_int_sv_f8_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8, static_cast<QuantGranularity>(qk_quant_gran), static_cast<QuantGranularity>(qk_quant_gran),
                                                        float, false, __half, ComputeUnit::kCudaCore, mask_mode, return_lse, true, false>;

        cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

        dim3 grid(div_ceil(seq_len, CTA_Q), head, batch);
        dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
        const int num_kv_groups = head / head;
        const int stride_bz_q = seq_len * head * headdim;
        const int stride_bz_k = seq_len * head * headdim;
        const int stride_bz_v = (seq_len + 63) / 64 * 64 * head * headdim; //batch * head * headdim * ((seq_len + 63) / 64 * 64)
        const int stride_bz_o = seq_len * head * headdim;

        const int stride_seq_q = headdim;
        const int stride_h_q = headdim * seq_len;

        const int stride_seq_k = headdim;
        const int stride_h_k = headdim * seq_len;

        const int stride_h_v = div_ceil(seq_len, 64) * 64 * headdim; //batch * head * headdim * ((seq_len + 63) / 64 * 64)
        const int stride_d_v = div_ceil(seq_len, 64) * 64; //(seq_len + 63) / 64 * 64;

        const int stride_seq_o = headdim;
        const int stride_h_o = headdim * seq_len;
        // printf("stride_bz_q: %d\n", stride_bz_q);
        // printf("stride_seq_q: %d\n", stride_seq_q);
        // printf("stride_h_q: %d\n", stride_h_q);
        // printf("stride_bz_k: %d\n", stride_bz_k);
        // printf("stride_seq_k: %d\n", stride_seq_k);
        // printf("stride_h_k: %d\n", stride_h_k);
        // printf("stride_bz_v: %d\n", stride_bz_v);

        printf("grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
        printf("block: (%d, %d)\n", block.x, block.y);
        printf("smem_max: %zu\n", smem_max);

        // printf("qk_quant_gran: %d\n", qk_quant_gran);
        // printf("mask_mode: %d\n", mask_mode);
        // printf("return_lse: %d\n", return_lse);

        // printf("qo_len: %d\n", seq_len);
        // printf("kv_len: %d\n", seq_len);
        // printf("num_kv_groups: %d\n", num_kv_groups);
        // printf("stride_bz_q: %d\n", stride_bz_q);
        // printf("stride_seq_q: %d\n", stride_seq_q);
        // printf("stride_h_q: %d\n", stride_h_q);
        // printf("stride_bz_k: %d\n", stride_bz_k);
        // printf("stride_seq_k: %d\n", stride_seq_k);
        // printf("stride_h_k: %d\n", stride_h_k);
        // printf("stride_bz_v: %d\n", stride_bz_v);
        // printf("stride_h_v: %d\n", stride_h_v);
        // printf("stride_d_v: %d\n", stride_d_v);
        // printf("stride_bz_o: %d\n", stride_bz_o);
        // printf("stride_seq_o: %d\n", stride_seq_o);
        // printf("stride_h_o: %d\n", stride_h_o);
        // printf("sm_scale: %f\n", sm_scale);

      
      

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
              v_scale,
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

        O_TYPE *o_ref = (O_TYPE *)malloc(batch * seq_len * head * headdim * sizeof(O_TYPE));
        cudaMemcpy(o_ref, o, batch * seq_len * head * headdim * sizeof(__half), cudaMemcpyDeviceToHost);
        fwrite_tensor<O_TYPE>(o_ref, batch * seq_len * head * headdim, "o_fp16.bin");
        //size_t out_size = batch * seq_len * head * headdim;
        for(size_t i = 0; i < qk_size; i++){
          // union {
          //   __nv_bfloat16 b;
          //   uint16_t u;
          // } u_h;

          // u_h.b = o_ref[i];
          // uint16_t res = u_h.u;
          // uint16_t exp_raw = (res >> 7) & 0xFF;
          // uint16_t mas = res & 0x7F;
          // printf("o[%zu]: %f, exp_raw: %04x, mantissa: %04x\n", i, exp_raw, mas);


          //printf("o[%zu]: %f\n", i, __half2float(o_ref[i]));
        }
        // for(int i = 0; i < 20; i++){
        //   printf("o[%d]: %f\n", i, __half2float(o_ref[i]));
        // }
        
  }
}




