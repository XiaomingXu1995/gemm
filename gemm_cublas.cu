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

using namespace std;
using data_type = float;
//using data_type = double;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialData(data_type *ip,int size, int step)
{
    //generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for(int i=0;i<size;i++)
    {
				data_type rand_num = rand() % step;
				ip[i] = rand_num / step;
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

void matrixMulOnHost(data_type* A, data_type* B, data_type* C, const int m, const int n, const int k){
	// A[m][k], B[k][n], C[m][n];
	for(int i = 0; i < m; i++){
			for(int j  = 0; j < k; j++){
		for(int t = 0; t < n; t++){
				C[i * n + t] += A[i*k + j] * B[j*n + t];
			}
		}
	}
}


int main(int argc, char *argv[]) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  //CHECK宏定义检查操作是否正常处理
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using Device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));
  //set up data size of matrix
  int m = 1<<12; //16384
  int k = 1<<12; //16384
  int n = 1<<12; //16384
	int rand_step = 1000;
	long compute_time = 2L * m * n * k;
  //int nxy = nx*ny;
  //int nBytes = nxy*sizeof(data_type);
  printf("Matrix A size: m %d k %d\n",m,k);
  printf("Matrix B size: k %d n %d\n",k,n);
  printf("Matrix C size: m %d n %d\n",m,n);
  //malloc host memory
  data_type *h_A,*h_B,*hostRef,*gpuRef;
  h_A = (data_type*)malloc(m * k * sizeof(data_type));
  h_B = (data_type*)malloc(k * n * sizeof(data_type));
  hostRef = (data_type*)malloc(m * n * sizeof(data_type));
  gpuRef = (data_type*)malloc(m * n * sizeof(data_type));
  //init data at host side
	double t0 = get_sec();
  initialData(h_A,m * k, rand_step);
  initialData(h_B,k * n, rand_step);
  memset(hostRef,0, m * n * sizeof(data_type));
  memset(gpuRef,0, m * n * sizeof(data_type));

	double t1 = get_sec();
	std::cout << "time of init is: " << t1 - t0 << std::endl;
  //sumMatrixOnHost(h_A,h_B,hostRef,nx,ny);
	//assert nx == ny == nz, since the malloc is the nx and ny
	matrixMulOnHost(h_A, h_B, hostRef, m, n, k);
	double t2 = get_sec();
  std::cout<<"sumMatrixOnHost cost "<<t2 -t1<<"sec\n";
	double gflopsCPU = compute_time * 1e-9 / (t2 - t1);
	std::cout<<"gflops of cpu host is: " << gflopsCPU << std::endl;

	//----------------------for cublas-----------------
	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;

	const int lda = m;
	const int ldb = k;
	const int ldc = n;

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
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * m * k));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * k * n));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * m * n));

	double t4 = get_sec();
	cerr << "time of init cuda malloc is: " << t4 - t3 << endl;

  CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, sizeof(data_type) * m * k, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, sizeof(data_type) * k * n, cudaMemcpyHostToDevice,
                             stream));

	double t5 = get_sec();
	cerr << "time of init cuda memcpy Matrix_a and Matrix_b is: " << t5 - t4 << endl;

  /* step 3: compute */
  CUBLAS_CHECK(
      cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
      //cublasDgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

	//cout << "the matrix c2 is: " << endl;
	//print_matrix(m, n, c2, m);

	double t6 = get_sec();
	cerr << "======time of cublasSgemm is: " << t6 - t5 << endl;
	double gflops_2 = (double)compute_time * 1e-9 / (t6-t5);
	fprintf(stderr, "the gflops of cublasSgemm matrix %d %d %d is: %lf\n", m, n, k, gflops_2); 

  /* step 4: copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(gpuRef, d_C, sizeof(data_type) * m * n, cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));


	double t7 = get_sec();
	cerr << "======time of check result gemm_base is: " << t7 - t6 << endl;
  checkResult(hostRef,gpuRef, m*n);


  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());

  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);
	return 0;

}
