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

using namespace std;
using data_type = float;
const string src_file0 = "billion0.float.random";
const string src_file1 = "billion1.float.random";

double get_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}

void gemm_base_naive(data_type* a, data_type* b, data_type* c, int m, int n, int l){
	for(int i = 0; i < m; i++){
			for(int k = 0; k < l; k++){
		for(int j = 0; j < n; j++){
				c[i * n + j] += a[i * l + k] * b[k * n + j];
			}
		}
	}
}


int main(int argc, char *argv[]) {
	double t0 = get_sec();
	if(argc < 5){
		cerr << "error, run as: <exe_gemm_base m n l res.file> to compute the (M,K) multiple (K,N) matrix, and store the result into res.file" << endl;
		return 1;
	}

	int m = stoi(argv[1]);
	int n = stoi(argv[2]);
	int l = stoi(argv[3]);
	string res_file = argv[4];
	
	data_type* a = (data_type*)_mm_malloc(m * l * sizeof(data_type), 64);
	data_type* b = (data_type*)_mm_malloc(l * n * sizeof(data_type), 64);
	data_type* c1 = (data_type*)_mm_malloc(m * n * sizeof(data_type), 64);
	data_type* c2 = (data_type*)_mm_malloc(m * n * sizeof(data_type), 64);

	FILE* fp0 = fopen(src_file0.c_str(), "rb");
	FILE* fp1 = fopen(src_file1.c_str(), "rb");
	assert(fp0 != NULL);
	assert(fp1 != NULL);

	int read_a = fread(a, sizeof(data_type), m * l, fp0);
	assert(read_a == m * l);
	fclose(fp0);

	int read_b = fread(b, sizeof(data_type), l * n, fp1);
	assert(read_b == l * n);
	fclose(fp1);

	for(int i = 0; i < m * n; i++){
		c1[i] = 0.0;
		c2[i] = 0.0;
	}

	long num_ops = 2L * m * n * l;

	//cout << "the matrix a is: " << endl;
	//print_matrix(m, l, a, m);

	//cout << "the matrix b is: " << endl;
	//print_matrix(l, n, b, l);

	double t1 = get_sec();
	cerr << "======time of init matrix is: " << t1 - t0 << endl;

	//gemm_base_naive(a, b, c1, m, n, l);

	//cout << "the matrix c1 is: " << endl;
	//print_matrix(m, n, c1, m);

	double t2 = get_sec();
	cerr << "======time of gemm_base_naive is: " << t2 - t1 << endl;
	double gflops = (double)num_ops * 1e-9 / (t2-t1);
	fprintf(stderr, "the gflops of gemm_base_naive matrix %d %d %d is: %lf\n", m, n, l, gflops); 
	
	//----------------------for cublas-----------------
	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;

	const int lda = m;
	const int ldb = l;
	const int ldc = m;

  const data_type alpha = 1.0;
  const data_type beta = 0.0;

  data_type *d_A = nullptr;
  data_type *d_B = nullptr;
  data_type *d_C = nullptr;

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_T;

  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

	double t3 = get_sec();
	cerr << "time of init cuda parameter is: " << t3 - t2 << endl;


  /* step 2: copy data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * m * l));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * l * n));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * m * n));

	double t4 = get_sec();
	cerr << "time of init cuda malloc is: " << t4 - t3 << endl;

  CUDA_CHECK(cudaMemcpyAsync(d_A, a, sizeof(data_type) * m * l, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, b, sizeof(data_type) * l * n, cudaMemcpyHostToDevice,
                             stream));

	double t5 = get_sec();
	cerr << "time of init cuda memcpy Matrix_a and Matrix_b is: " << t5 - t4 << endl;

  /* step 3: compute */
  CUBLAS_CHECK(
      cublasSgemm(cublasH, transa, transb, m, n, l, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

  /* step 4: copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(c2, d_C, sizeof(data_type) * m * n, cudaMemcpyDeviceToHost,
                             stream));
	//cout << "the matrix c2 is: " << endl;
	//print_matrix(m, n, c2, m);

	double t6 = get_sec();
	cerr << "======time of cublasSgemm is: " << t6 - t5 << endl;
	double gflops_2 = (double)num_ops * 1e-9 / (t6-t5);
	fprintf(stderr, "the gflops of cublasSgemm matrix %d %d %d is: %lf\n", m, n, l, gflops_2); 

  CUDA_CHECK(cudaStreamSynchronize(stream));

	string res_file_base0 = res_file + ".base0";
	FILE* fp_res = fopen(res_file_base0.c_str(), "wb");
	assert(fp_res != NULL);
	int err_num = 0;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			if(abs(c1[i*n+j] - c2[j*m+i]) > 1e-5 * abs(c1[i*n+j])){
			//if(abs(c1[i*n+j] - c2[j*m+i]) > 1e-5){
				err_num++;
				fprintf(fp_res, "%d\t%d\t%f\t%f\n", i, j, c1[i*n+j], c2[j*m+i]);
			}
		}
	}

	fclose(fp_res);

	double t7 = get_sec();
	cerr << "======time of check result gemm_base is: " << t7 - t6 << endl;
	cerr << "-----the error number is: " << err_num << endl;


  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());


	_mm_free(a);
	_mm_free(b);
	_mm_free(c1);
	_mm_free(c2);
	return 0;

}
