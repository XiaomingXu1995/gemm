#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<sys/time.h>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define CUDA_CHECK(call)                     \
{                                       \
    const cudaError_t error = call;     \
    if(error!=cudaSuccess)              \
    {                                   \
        printf("Error: %s:%d",__FILE__,__LINE__);      \
        std::cout<<"code: "<<error<<" ,reason: "<<cudaGetErrorString(error)<<std::endl;     \
        exit(-10*error);     \
    }                        \
}
//使用gettimeofday会获取自1970年1月1日0点以来到现在的秒数
//timeval是一个结构体，其中有成员 tv_sec:秒 tv_usec:微秒
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
//初始化数组
template<typename T>    
void initialData(T *ip,int size, int step)
{
    //generate different seed for random number
    if constexpr (std::is_same<T, int8_t>::value){
      srand(23);
      step = 127;
      for(int i=0;i<size;i++)
      {
        int rand_num = rand() % step;
        ip[i] = rand_num;
      }
    }
    else if constexpr (std::is_same<T, float>::value){
      srand(23);
      for(int i=0;i<size;i++)
      {
        int rand_num = rand() % step;
        ip[i] = rand_num / step;
      }
    }
    else if constexpr (std::is_same<T, __half>::value){
      srand(23);
      for(int i=0;i<size;i++)
      {
        int rand_num = rand() % step;
        ip[i] = __float2half(rand_num / step);
      }
    }
}
//hostRef传入CPU端计算的矩阵加法结果，gpuRef传入GPU端计算的矩阵加法结果
//对比争取输出"Arrats match"
void checkResult(float *hostRef,float *gpuRef,const int N)
{
    //double epsilon = 1.0E-8;
    //double epsilon = 1.0E-5;
    double epsilon = 1.0E-2;
    bool match=1;
    for(int i=0;i<N;i++)
    {
				//printf("%d\t%5.2f\t%5.2f\n",i, hostRef[i], gpuRef[i]);
        if(abs(hostRef[i]-gpuRef[i])>epsilon)
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

void matrixMulOnHost(float* A, float* B, float* C, const int m, const int n, const int k){
	// A[m][k], B[k][n], C[m][n];
	for(int i = 0; i < m; i++){
			for(int j  = 0; j < k; j++){
		for(int t = 0; t < n; t++){
				C[i * n + t] += A[i*k + j] * B[j*n + t];
			}
		}
	}
}

template<typename T, typename U>
__global__ void matrixMulOnGPU(T* A, T* B, U* C, const int n){
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
	U val = 0.0f;
	if(row < n && col < n){
		for(int i = 0; i < n; i++){
			val += A[row*n+i] * B[i*n+col]; 
		}
		C[row*n+col] = val;
	}
}

template<typename T, typename U>    
__global__ void matrixMulOnGPU(T* A, T* B, U* C, const int m, const int n, const int k){
	// A[m][k], B[k][n], C[m][n];
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
	U val = 0.0f;
	if(row < m && col < n){
		for(int i = 0; i < k; i++){
			val += A[row*k+i] * B[i*n+col]; 
		}
		C[row*n+col] = val;
	}
}

// Tile-based cached Matrix Multiplication
#define TILE_WIDTH 32

template <typename T>
__global__ void matrixMul(T* A, T* B, T* C, int N) {
    __shared__ T tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ T tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    T Pvalue = 0;

    for (int ph = 0; ph < N / TILE_WIDTH; ++ph) {
        // Load tiles into shared memory
        tile_A[ty][tx] = A[row * N + ph * TILE_WIDTH + tx];
        tile_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];
        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += tile_A[ty][k] * tile_B[k][tx];
        }
        __syncthreads();
    }

    C[row * N + col] = Pvalue;
}

template <typename T, typename U>
__global__ void matrixMul(T* A, T* B, U* C, int M, int K, int N) {
    __shared__ T tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ T tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    U Pvalue = 0;

    // Loop over the tiles of the input matrices
    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        // Load tiles into shared memory, with bounds checking
        if (row < M && (ph * TILE_WIDTH + tx) < K) {
            tile_A[ty][tx] = A[row * K + ph * TILE_WIDTH + tx];
        } else {
            tile_A[ty][tx] = 0.0;
        }

        if (col < N && (ph * TILE_WIDTH + ty) < K) {
            tile_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];
        } else {
            tile_B[ty][tx] = 0.0;
        }

        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += tile_A[ty][k] * tile_B[k][tx];
        }
        __syncthreads();
    }

    // Store the result in the output matrix
    if (row < M && col < N) {
        C[row * N + col] = Pvalue;
    }
}



void sumMatrixOnHost(float *A,float *B,float *C,const int nx ,const int ny)
{
    float *ia=A;
    float *ib=B;
    float *ic=C;
    for(int iy=0;iy<ny;iy++)
    {
        for(int ix=0;ix<nx;ix++)
        {
            ic[ix]=ia[ix]+ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}
//cuda核函数计算矩阵加法
__global__ void sumMatrixOnGPU(float *MatA,float *MatB,float *MatC,int nx,int ny)
{
    //使用前问中的线程全局索引的计算方式
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;
    if(ix<nx && iy<ny)
    {
        //这种线程的全局索引方式正好是与按行优先的存储的矩阵的索引方式是一致的
        //所以线程的全局索引可以与矩阵中元素的索引很好的对应
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

template <typename T, typename U>
void mm_cuda(int warmup, int repeat, int m, int n, int k) {
    std::cout << "-------------------------Matrix Multiplication--------------" << std::endl;
    long ops = 2L * m * n * k;

    T *h_A, *h_B;
    T * d_A, *d_B;
    U *h_C, *d_C;
    h_A = (T*)malloc(m * k * sizeof(T));
    h_B = (T*)malloc(k * n * sizeof(T));
    h_C = (U*)malloc(m * n * sizeof(U));
    initialData<T>(h_A, m * k, 1000);
    initialData<T>(h_B, k * n, 1000);
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(U)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(T), cudaMemcpyHostToDevice));
    dim3 block(32, 32);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    double iStart = cpuSecond();
    for (int i = 0; i < warmup; i++) {
      matrixMulOnGPU<T, U><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    printf("Warmup & Kernel: %f sec\n", iElaps);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < repeat; i++) {
      matrixMulOnGPU<T, U><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    printf("Kernel: %f sec\n", iElaps);   
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, int32_t>::value){
      std::cout << "----int8_int8_int32 Time taken: " << milliseconds << " ms" << std::endl;
      std::cout << "----int8_int8_int32 TFlops: " << (double)ops * 1e-12 / (milliseconds / repeat / 1000) << std::endl;
    }
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

template <typename T, typename U>
void mm_cuda_tile(int warmup, int repeat, int m, int n, int k) {
    std::cout << "-------------------------Tile-based cached Matrix Multiplication--------------" << std::endl;
    long ops = 2L * m * n * k;

    T *h_A, *h_B;
    T * d_A, *d_B;
    U *h_C, *d_C;
    h_A = (T*)malloc(m * k * sizeof(T));
    h_B = (T*)malloc(k * n * sizeof(T));
    h_C = (U*)malloc(m * n * sizeof(U));
    initialData<T>(h_A, m * k, 1000);
    initialData<T>(h_B, k * n, 1000);
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(U)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(T), cudaMemcpyHostToDevice));
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    double iStart = cpuSecond();
    for (int i = 0; i < warmup; i++) {
      matrixMul<T, U><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    printf("Warmup & Kernel: %f sec\n", iElaps);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < repeat; i++) {
      matrixMul<T, U><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    printf("Kernel: %f sec\n", iElaps);   
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    if constexpr (std::is_same<T, int8_t>::value && std::is_same<U, int32_t>::value){
      std::cout << "----int8_int8_int32_tile Time taken: " << milliseconds << " ms" << std::endl;
      std::cout << "----int8_int8_int32_tile TFlops: " << (double)ops * 1e-12 / (milliseconds / repeat / 1000) << std::endl;
    }
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc,char **argv)
{
    int bits = 12;
    int warmup = 30;
    int repeat = 3000;
    if(argc >= 2){
      bits = std::stoi(argv[1]);
    }
    if(argc >= 3){
      warmup = std::stoi(argv[2]);
      repeat = warmup * 100;
    }
    int dev = 0;
    cudaDeviceProp deviceProp;
    //CHECK宏定义检查操作是否正常处理
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d: %s\n",dev,deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));
    //set up data size of matrix
    int m = 1<<bits; //16384
    int k = 1<<bits; //16384
    int n = 1<<bits; //16384
    //int nxy = nx*ny;
    //int nBytes = nxy*sizeof(float);
    printf("Matrix A size: m %d k %d\n",m,k);
    printf("Matrix B size: k %d n %d\n",k,n);
    printf("Matrix C size: m %d n %d\n",m,n);

    mm_cuda<int8_t, int32_t>(warmup, repeat, m, n, k);
    mm_cuda_tile<int8_t, int32_t>(warmup, repeat, m, n, k);
    return 0;
}
