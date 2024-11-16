#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<sys/time.h>
#include <iostream>
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
//使用gettimeofday会获取自1970年1月1日0点以来到现在的秒数
//timeval是一个结构体，其中有成员 tv_sec:秒 tv_usec:微秒
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
//初始化数组
void initialData(float *ip,int size, int step)
{
    //generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for(int i=0;i<size;i++)
    {
				float rand_num = rand() % step;
				ip[i] = rand_num / step;
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

__global__ void matrixMulOnGPU(float* A, float* B, float* C, const int n){
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
	float val = 0.0f;
	if(row < n && col < n){
		for(int i = 0; i < n; i++){
			val += A[row*n+i] * B[i*n+col]; 
		}
		C[row*n+col] = val;
	}
}

__global__ void matrixMulOnGPU(float* A, float* B, float* C, const int m, const int n, const int k){
	// A[m][k], B[k][n], C[m][n];
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
	float val = 0.0f;
	if(row < m && col < n){
		for(int i = 0; i < k; i++){
			val += A[row*k+i] * B[i*n+col]; 
		}
		C[row*n+col] = val;
	}
}

// Tile-based cached Matrix Multiplication
#define TILE_WIDTH 32

__global__ void matrixMul(float* A, float* B, float* C, int N) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

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

__global__ void matrixMul(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

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
int main(int argc,char **argv)
{
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
    //int nBytes = nxy*sizeof(float);
    printf("Matrix A size: m %d k %d\n",m,k);
    printf("Matrix B size: k %d n %d\n",k,n);
    printf("Matrix C size: m %d n %d\n",m,n);
    //malloc host memory
    float *h_A,*h_B,*hostRef,*gpuRef;
    h_A = (float*)malloc(m * k * sizeof(float));
    h_B = (float*)malloc(k * n * sizeof(float));
    hostRef = (float*)malloc(m * n * sizeof(float));
    gpuRef = (float*)malloc(m * n * sizeof(float));
    //init data at host side
    double iStart = cpuSecond();
    initialData(h_A,m * k, rand_step);
    initialData(h_B,k * n, rand_step);
    memset(hostRef,0, m * n * sizeof(float));
    memset(gpuRef,0, m * n * sizeof(float));
    double iElaps = cpuSecond() - iStart;
		std::cout << "time of init is: " << iElaps << std::endl;
    iStart = cpuSecond();
    //sumMatrixOnHost(h_A,h_B,hostRef,nx,ny);
		//assert nx == ny == nz, since the malloc is the nx and ny
		matrixMulOnHost(h_A, h_B, hostRef, m, n, k);
    iElaps = cpuSecond() - iStart; //cpu 端耗时
    std::cout<<"sumMatrixOnHost cost "<<iElaps<<"sec\n";
		double gflopsCPU = compute_time * 1e-9 / iElaps;
		std::cout<<"gflops of cpu host is: " << gflopsCPU << std::endl;
    //malloc device global memory
    //GPU 申请GPU端空间
    float *d_MatA,*d_MatB,*d_MatC;
    cudaMalloc((void**)&d_MatA, m * k * sizeof(float));
    cudaMalloc((void**)&d_MatB, k * n * sizeof(float));
    cudaMalloc((void**)&d_MatC, m * n * sizeof(float));

		// transpose the matrix B
		//float* h_B_trans = (float*)malloc(k * n * sizeof(float));
		//for(int i = 0; i < n; i++){
		//	for(int j = 0; j < k; j++){
		//		h_B_trans[i * k + j] = h_B[j * n + i];
		//	}
		//}
    //transfer data from host to device
    //数据传输
    cudaMemcpy(d_MatA,h_A, m*k*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB,h_B, k*n*sizeof(float),cudaMemcpyHostToDevice);
    //cudaMemcpy(d_MatB,h_B_trans, k*n*sizeof(float),cudaMemcpyHostToDevice);
    //invoke kernel at host side
    int dimx = TILE_WIDTH;
    int dimy = TILE_WIDTH;
    //block size = (32,32) 32 = 2^5
    //也就是每个block中有32*32个线程（结构是二维）
    dim3 block(dimx,dimy);
    //grid size = (512,512) 512 = 2^9 = 2^(14-5)
    //也就是每个grid中有512*512个block （结构是二维）
    dim3 grid((m+block.x-1)/block.x,((n+block.y-1)/block.y));
    iStart = cpuSecond();//gpu初始时间
    //sumMatrixOnGPU<<<grid,block>>>(d_MatA,d_MatB,d_MatC,nx,ny);//以上述配置线程层级结构的方式启动核函数
		//matrixMulOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx);
		matrixMulOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, m, n, k);
		//matrixMul<<<grid, block>>>(d_MatA, d_MatB, d_MatC, n);
		//matrixMul<<<grid, block>>>(d_MatA, d_MatB, d_MatC, m, k, n);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("MulMatrixOnGPU<<<(%d,%d),(%d,%d)>>>elapsed %f sec\n",grid.x,grid.y,block.x,block.y,iElaps);
		double gflopsGPU = compute_time * 1e-9 / iElaps;
		std::cout<<"gflops of GPU is: " << gflopsGPU << std::endl;
    //copy kernel result back to host side
    //再把GPU计算的结果拷贝会cpu端
    cudaMemcpy(gpuRef,d_MatC,m*n*sizeof(float),cudaMemcpyDeviceToHost);
    //check device res
    checkResult(hostRef,gpuRef, m*n);
    
    //释放gpu中申请的内存
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    //释放主机端内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    //reset device 
    cudaDeviceReset();
    return 0;
}
