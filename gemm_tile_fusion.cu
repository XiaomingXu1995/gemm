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
using data_type = float;
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
            printf("host %f gpu %f at current %d\n",hostRef[i],gpuRef[i], i);
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

// 定义tile大小
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

__global__ void cuda_gemm_tile(data_type* A, data_type* B, data_type* C, int m, int n, int k) {
    // 声明共享内存
    __shared__ data_type As[TILE_HEIGHT][TILE_WIDTH];
    __shared__ data_type Bs[TILE_WIDTH][TILE_WIDTH];
    
    // 计算线程索引
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算全局索引
    int row = by * TILE_HEIGHT + ty;
    int col = bx * TILE_WIDTH + tx;
    
    data_type value = 0;
    
    // 检查边界
    if (row < m && col < n) {
        // 对每个tile进行计算
        for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
            // 加载数据到共享内存
            if (row < m && t * TILE_WIDTH + tx < k) {
                As[ty][tx] = A[row * k + t * TILE_WIDTH + tx];
            } else {
                As[ty][tx] = 0;
            }
            
            if (t * TILE_WIDTH + ty < k && col < n) {
                Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * n + col];
            } else {
                Bs[ty][tx] = 0;
            }
            
            // 同步所有线程
            __syncthreads();
            
            // 计算tile内的点积
            for (int i = 0; i < TILE_WIDTH; i++) {
                value += As[ty][i] * Bs[i][tx];
            }
            
            // 同步所有线程
            __syncthreads();
        }
        
        // 写入结果
        C[row * n + col] = value;
    }
}

int main(int argc, char* argv[]){
    // Q = N * d;
    // K = d * N; 
    // V = N * d;
    int N = 4096;
    int d = 4096;
    string mm_type = "v1";
    if(argc >= 2){
        mm_type = argv[1];
    }
    if(argc >= 3){
        N = stoi(argv[2]);
    }
    if(argc >= 4){
        d = stoi(argv[3]);
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
    printf("time of memset h_S,h_O and memcpy d_Q,d_K,d_V is: %f\n", t3-t2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 设置线程块和网格大小
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid0((N+threadsPerBlock.x-1)/threadsPerBlock.x, (N+threadsPerBlock.y-1)/threadsPerBlock.y);
    dim3 blocksPerGrid1((d+threadsPerBlock.x-1)/threadsPerBlock.x, (N+threadsPerBlock.y-1)/threadsPerBlock.y);

    printf("-----------------------------------\n");
    printf("N: %d, d: %d, mm_type: %s\n", N, d, mm_type.c_str());
    printf("-----------------------------------\n");

    // 测试原始版本
    if (mm_type == "v1"){
        cout << "\nTesting original cuda_gemm:" << endl;
        // 第一轮计算并检查精度
        cudaEventRecord(start);
        cuda_gemm<<<blocksPerGrid0, threadsPerBlock>>>(d_Q, d_K, d_S, N, N, d);
        cuda_gemm<<<blocksPerGrid1, threadsPerBlock>>>(d_S, d_V, d_O, N, d, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    
        // 检查精度
        cudaMemcpy(res_S, d_S, N * N * sizeof(data_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(res_O, d_O, N * d * sizeof(data_type), cudaMemcpyDeviceToHost);
        cout << "Checking first computation accuracy:" << endl;
        checkResult(h_S, res_S, N * N);
        checkResult(h_O, res_O, N * d);

        // 30轮预热
        cout << "Starting 30 rounds warm-up..." << endl;
        for(int i = 0; i < 30; i++) {
            cuda_gemm<<<blocksPerGrid0, threadsPerBlock>>>(d_Q, d_K, d_S, N, N, d);
            cuda_gemm<<<blocksPerGrid1, threadsPerBlock>>>(d_S, d_V, d_O, N, d, N);
        }
        cudaDeviceSynchronize();

        // 300轮性能测试
        cout << "Starting 300 rounds performance test..." << endl;
        cudaEventRecord(start);
        for(int i = 0; i < 300; i++) {
            cuda_gemm<<<blocksPerGrid0, threadsPerBlock>>>(d_Q, d_K, d_S, N, N, d);
            cuda_gemm<<<blocksPerGrid1, threadsPerBlock>>>(d_S, d_V, d_O, N, d, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        double avg_time_original = milliseconds / 1000.0 / 300.0;  // 每轮平均时间
        double tflops_original = (double)ops * 1e-12 / avg_time_original;  // 转换为TFLOPS
        cout << "Original version - Average time per round: " << avg_time_original << " seconds" << endl;
        cout << "Original version - Performance: " << tflops_original << " TFLOPS" << endl;
    }
    else if (mm_type == "v2"){

        // 测试tile版本
        cout << "\nTesting tiled cuda_gemm_tile:" << endl;
        // 第一轮计算并检查精度
        cudaEventRecord(start);
        cuda_gemm_tile<<<blocksPerGrid0, threadsPerBlock>>>(d_Q, d_K, d_S, N, N, d);
        cuda_gemm_tile<<<blocksPerGrid1, threadsPerBlock>>>(d_S, d_V, d_O, N, d, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    
        // 检查精度
        cudaMemcpy(res_S, d_S, N * N * sizeof(data_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(res_O, d_O, N * d * sizeof(data_type), cudaMemcpyDeviceToHost);
        cout << "Checking first computation accuracy:" << endl;
        checkResult(h_S, res_S, N * N);
        checkResult(h_O, res_O, N * d);

        // 30轮预热
        cout << "Starting 30 rounds warm-up..." << endl;
        for(int i = 0; i < 30; i++) {
            cuda_gemm_tile<<<blocksPerGrid0, threadsPerBlock>>>(d_Q, d_K, d_S, N, N, d);
            cuda_gemm_tile<<<blocksPerGrid1, threadsPerBlock>>>(d_S, d_V, d_O, N, d, N);
        }
        cudaDeviceSynchronize();

        // 300轮性能测试
        cout << "Starting 300 rounds performance test..." << endl;
        cudaEventRecord(start);
        for(int i = 0; i < 300; i++) {
            cuda_gemm_tile<<<blocksPerGrid0, threadsPerBlock>>>(d_Q, d_K, d_S, N, N, d);
            cuda_gemm_tile<<<blocksPerGrid1, threadsPerBlock>>>(d_S, d_V, d_O, N, d, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        double avg_time_tiled = milliseconds / 1000.0 / 300.0;  // 每轮平均时间
        double tflops_tiled = (double)ops * 1e-12 / avg_time_tiled;  // 转换为TFLOPS
        cout << "Tiled version - Average time per round: " << avg_time_tiled << " seconds" << endl;
        cout << "Tiled version - Performance: " << tflops_tiled << " TFLOPS" << endl;
    }

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






