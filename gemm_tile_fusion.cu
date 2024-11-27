#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
//#include <bits/stdc++.h>
#include <sys/time.h>
#include <assert.h>


#include <cuda_runtime.h>

#include <omp.h>

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
using data_type = int;
//using data_type = double;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialData(data_type *ip, int size, int step)
{
    //time_t t;
    //srand((unsigned)time(&t));
    srand(0);
    // for no overflow of 4096 * 4096 matrix, set step < = 8
    step = min(step, 8);
    for(int i=0;i<size;i++)
    {
		data_type rand_num = rand() % step;
		ip[i] = rand_num;
    }
}

//hostRef传入CPU端计算的矩阵加法结果，gpuRef传入GPU端计算的矩阵加法结果
//对比争取输出"Arrats match"
void checkResult(data_type *hostRef,data_type *gpuRef,const int N)
{
    //double epsilon = 1.0E-8;
    //double epsilon = 1.0E-5;
    //double epsilon = 1.0E-2;
		//double rate = 0.2;
    bool match=1;
    for(int i=0;i<N;i++)
    {
        //if(abs(hostRef[i]-gpuRef[i])>epsilon)
        //if(abs(hostRef[i]-gpuRef[i]) > rate * abs(hostRef[i]))
        if(hostRef[i] != gpuRef[i])
        {
            match=0;
            printf("Arrays do not match");
            printf("host %d gpu %d at current %d\n",hostRef[i],gpuRef[i], i);
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
    int num_procs = omp_get_num_procs();
    #pragma omp parallel for num_threads(num_procs / 2)
	for(int i = 0; i < m; i++){
			for(int j  = 0; j < k; j++){
		for(int t = 0; t < n; t++){
				C[i * n + t] += A[i*k + j] * B[j*n + t];
			}
		}
	}
}

//cuda Gemm 

__global__ void cuda_gemm(data_type* A, data_type* B, data_type* C, int m, int n, int k) {
    // 每个线程处理 C 的一个元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        data_type value = 0;
        for (int i = 0; i < k; ++i) {
            value += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = value;
    }
}


int main(int argc, char* argv[]){
    // Q = N * d;
    // K = d * N; 
    // V = N * d;
    int N = 4096;
    int d = 4096;
    if(argc >= 3){
        N = stoi(argv[1]);
        d = stoi(argv[2]);
    }
    size_t ops = 0;
    double t0 = get_sec();
    data_type * h_Q = (data_type*)malloc(N * d *sizeof(data_type)); // Q[N][d]
    data_type * h_K = (data_type*)malloc(N * d *sizeof(data_type)); // K[d][N]
    data_type * h_V = (data_type*)malloc(N * d *sizeof(data_type)); // V[N][d]
    data_type * h_S = (data_type*)malloc(N * N *sizeof(data_type)); // S = Q * K; S[N][N]
    data_type * h_O = (data_type*)malloc(N * d *sizeof(data_type)); // O = S * V; O[N][d]
    memset(h_S, 0, N * N * sizeof(data_type));
    memset(h_O, 0, N * d * sizeof(data_type));
    initialData(h_Q, N*d, 4);
    initialData(h_K, N*d, 4);
    initialData(h_V, N*d, 4);

    //---------------gemm on host-------------------
    double t1 = get_sec();
    cout << "time of initData is: " << t1-t0 << endl;

    matrixMulOnHost(h_Q, h_K, h_S, N, N, d);
    ops += 2L * N * N * d;
    
    matrixMulOnHost(h_S, h_V, h_O, N, d, N);
    ops += 2L * N * N * d;

    double t2 = get_sec();
    cout << "time of gemm on host is: " << t2-t1 << endl;
    double gflops_host = (double)ops * 1e-9 / (t2-t1);
    cout << "ops is: " << ops << endl;
    cout << "the gops of host_gemm is: " << gflops_host << endl;

    data_type *d_Q, *d_K, *d_S, *d_V, *d_O;
    data_type *res_S = (data_type*)malloc(N * N * sizeof(data_type));
    data_type *res_O = (data_type*)malloc(N * d * sizeof(data_type));

    cudaMalloc(&d_Q, N * d * sizeof(data_type));
    cudaMalloc(&d_K, N * d * sizeof(data_type));
    cudaMalloc(&d_V, N * d * sizeof(data_type));
    cudaMalloc(&d_S, N * N * sizeof(data_type));
    cudaMalloc(&d_O, N * d * sizeof(data_type));

    cudaMemcpy(d_Q, h_Q, N * d * sizeof(data_type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, N * d * sizeof(data_type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N * d * sizeof(data_type), cudaMemcpyHostToDevice);
    cudaMemset(d_S, 0, N * N * sizeof(data_type));
    cudaMemset(d_O, 0, N * d * sizeof(data_type));
    double t3 = get_sec();
    cout << "time of memset h_S,h_O and memcpy d_Q,d_K,d_V is: " << t3-t2 << endl;
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid0((N+threadsPerBlock.x-1)/threadsPerBlock.x, (N+threadsPerBlock.y-1)/threadsPerBlock.y);
    dim3 blocksPerGrid1((d+threadsPerBlock.x-1)/threadsPerBlock.x, (N+threadsPerBlock.y-1)/threadsPerBlock.y);
    cuda_gemm<<<blocksPerGrid0, threadsPerBlock>>>(d_Q, d_K, d_S, N, N, d);
    cuda_gemm<<<blocksPerGrid1, threadsPerBlock>>>(d_S, d_V, d_O, N, d, N);
    //cudaMemcpy(res_S, d_S, N * N * sizeof(data_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_O, d_O, N * d * sizeof(data_type), cudaMemcpyDeviceToHost);

    double t4 = get_sec();
    cout << "time of cuda_gemm is: " << t4-t3 << endl;
    double gops_cuda_gemm = (double)ops * 1e-9 / (t4-t3);
    cout << "the gpos of cuda_gemm is: " << gops_cuda_gemm << endl;

    //// check the gemm result
    // checkResult(h_S, res_S, N * N);
    checkResult(h_O, res_O, N * d);



    free(h_Q);
    free(h_K);
    free(h_S);
    free(h_V);
    free(h_O);
    free(res_S);
    free(res_O);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_S);
    cudaFree(d_O);
    return 0;
}






