#include <iostream>
#include <vector>
#include <sys/time.h>
#include <assert.h>
#include <immintrin.h>
#include <pthread.h>
#include <sys/mman.h>



using namespace std;

//const string src_file0 = "million0.file";
//const string src_file1 = "million1.file";
const string src_file0 = "billion0.float.random";
const string src_file1 = "billion1.float.random";

double get_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}

static void thread_bind(int cpu){
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(cpu, &cpu_set);
	if(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) != 0){
		fprintf(stderr, "Error: cpu[%d] bind failed.\n", cpu);
		exit(0);
	}
}

static void *page_alloc(size_t size)
{
    void *data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    if (data == (void*)-1)
    {
        fprintf(stderr, "Error(MemData::Construction): mmap failed.\n");
        exit(0);
    }
    return data;
}

void gemm_base_naive(float* a, float* b, float* c, int m, int n, int l){
	for(int i = 0; i < m; i++){
			for(int k = 0; k < l; k++){
		for(int j = 0; j < n; j++){
				c[i * n + j] += a[i * l + k] * b[k * n + j];
			}
		}
	}
}


void gemm_base_vec(float* a, float* b, float* c, int m, int n, int l){
	__m256 v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;
	//n == 24
	
	int s_i = 0;
	int s_k = 0;
	//n = 24;
	//int k = l;

	//#pragma omp parallel for num_threads(10)
	for(int s_j = 0; s_j < n; s_j+=24){
		for(int s_i = 0; s_i < m; s_i+=4){
			__m256 v0 = _mm256_set1_ps(0.0);
			__m256 v1 = v0; 
			__m256 v2 = v0; 
			__m256 v3 = v0; 
			__m256 v4 = v0; 
			__m256 v5 = v0; 
			__m256 v6 = v0; 
			__m256 v7 = v0; 
			__m256 v8 = v0; 
			__m256 v9 = v0; 
			__m256 v10 = v0; 
			__m256 v11 = v0; 
			for(int kk = 0; kk < l; kk++){
				__m256 v12 = _mm256_load_ps(&b[kk*n+s_j + 0]);
				__m256 v13 = _mm256_load_ps(&b[kk*n+s_j + 8]);
				__m256 v14 = _mm256_load_ps(&b[kk*n+s_j + 16]);

				__m256 v15 = _mm256_set1_ps(a[(s_i+0)*l+kk]);
				// v0 = _mm256_add_ps(v0, _mm256_mul_ps(v15, v12));
				// v1 = _mm256_add_ps(v1, _mm256_mul_ps(v15, v13));
				// v2 = _mm256_add_ps(v2, _mm256_mul_ps(v15, v14));
				v0 = _mm256_fmadd_ps(v15, v12, v0);
				v1 = _mm256_fmadd_ps(v15, v13, v1);
				v2 = _mm256_fmadd_ps(v15, v14, v2);

				v15 = _mm256_set1_ps(a[(s_i+1)*l+kk]);
				// v3 = _mm256_add_ps(v3, _mm256_mul_ps(v15, v12));
				// v4 = _mm256_add_ps(v4, _mm256_mul_ps(v15, v13));
				// v5 = _mm256_add_ps(v5, _mm256_mul_ps(v15, v14));
				v3 = _mm256_fmadd_ps(v15, v12, v3);
				v4 = _mm256_fmadd_ps(v15, v13, v4);
				v5 = _mm256_fmadd_ps(v15, v14, v5);

				v15 = _mm256_set1_ps(a[(s_i+2)*l+kk]);
				// v6 = _mm256_add_ps(v6, _mm256_mul_ps(v15, v12));
				// v7 = _mm256_add_ps(v7, _mm256_mul_ps(v15, v13));
				// v8 = _mm256_add_ps(v8, _mm256_mul_ps(v15, v14));
				v6 = _mm256_fmadd_ps(v15, v12, v6);
				v7 = _mm256_fmadd_ps(v15, v13, v7);
				v8 = _mm256_fmadd_ps(v15, v14, v8);

				v15 = _mm256_set1_ps(a[(s_i+3)*l+kk]);
				// v9 = _mm256_add_ps(v9, _mm256_mul_ps(v15, v12));
				// v10 = _mm256_add_ps(v10, _mm256_mul_ps(v15, v13));
				// v11 = _mm256_add_ps(v11, _mm256_mul_ps(v15, v14));
				v9 = _mm256_fmadd_ps(v15, v12, v9);
				v10 = _mm256_fmadd_ps(v15, v13, v10);
				v11 = _mm256_fmadd_ps(v15, v14, v11);
			}
			_mm256_store_ps(&c[(s_i+0)*n+s_j+0], v0);
			_mm256_store_ps(&c[(s_i+0)*n+s_j+8], v1);
			_mm256_store_ps(&c[(s_i+0)*n+s_j+16], v2);

			_mm256_store_ps(&c[(s_i+1)*n+s_j+0], v3);
			_mm256_store_ps(&c[(s_i+1)*n+s_j+8], v4);
			_mm256_store_ps(&c[(s_i+1)*n+s_j+16], v5);

			_mm256_store_ps(&c[(s_i+2)*n+s_j+0], v6);
			_mm256_store_ps(&c[(s_i+2)*n+s_j+8], v7);
			_mm256_store_ps(&c[(s_i+2)*n+s_j+16], v8);

			_mm256_store_ps(&c[(s_i+3)*n+s_j+0], v9);
			_mm256_store_ps(&c[(s_i+3)*n+s_j+8], v10);
			_mm256_store_ps(&c[(s_i+3)*n+s_j+16], v11);
		}
	}

}


int main(int argc, char * argv[]){
	double t0 = get_sec();
	if(argc < 5){
		cerr << "error, run as: <exe_gemm_base m n l res.file> to compute the (M,K) multiple (K,N) matrix, and store the result into res.file" << endl;
		return 1;
	}
	int m = stoi(argv[1]);
	int n = stoi(argv[2]);
	int l = stoi(argv[3]);
	string res_file = argv[4];

	thread_bind(1);

	//float* a = new float[m * l];
	//float* b = new float[l * n];
	//float * c1 = new float[m * n];
	//float * c2 = new float[m * n];
	
	// float* a = (float*)_mm_malloc(m * l * sizeof(float), 64);
	// float* b = (float*)_mm_malloc(l * n * sizeof(float), 64);
	// float* c1 = (float*)_mm_malloc(m * n * sizeof(float), 64);
	// float* c2 = (float*)_mm_malloc(m * n * sizeof(float), 64);

	float* a = (float*)page_alloc(m * l * sizeof(float));
	float* b = (float*)page_alloc(l * n * sizeof(float));
	float* c1 = (float*)page_alloc(m * n * sizeof(float));
	float* c2 = (float*)page_alloc(m * n * sizeof(float));
	
	FILE* fp0 = fopen(src_file0.c_str(), "rb");
	FILE* fp1 = fopen(src_file1.c_str(), "rb");
	assert(fp0 != NULL);
	assert(fp1 != NULL);

	int read_a = fread(a, sizeof(float), m * l, fp0);
	assert(read_a == m * l);
	fclose(fp0);
	//for(int i = 0; i < m * l; i++){
	//	cout << a[i] << endl;
	//}
	//cout << "----------------" << endl;

	int read_b = fread(b, sizeof(float), l * n, fp1);
	assert(read_b == l * n);
	fclose(fp1);
	//for(int i = 0; i < n * l; i++){
	//	cout << b[i] << endl;
	//}
	//exit(0);

	for(int i = 0; i < m * n; i++){
		c1[i] = 0.0;
		c2[i] = 0.0;
	}

	long num_ops = 2L * m * n * l;
	int loop_time = (int)(2e11 / num_ops);

	double t1 = get_sec();
	cerr << "======time of init matrix is: " << t1 - t0 << endl;

	for(int i = 0; i < loop_time; i++){
		gemm_base_vec(a, b, c2, m, n, l);
	}

	//gemm_base(matrix0, matrix1, res, m, n, l);

	double t2 = get_sec();
	cerr << "======time of warming up gemm_base_vec is: " << t2 - t1 << endl;
	double gflops = (double)num_ops * 1e-9 / ((t2-t1)/loop_time);
	fprintf(stderr, "the gflops of warming up gemm_base_vec matrix %d %d %d is: %lf\n", m, n, l, gflops); 

	for(int i = 0; i < loop_time; i++){
		gemm_base_vec(a, b, c2, m, n, l);
	}

	double t2_2 = get_sec();
	cerr << "======time of gemm_base_vec is: " << t2_2 - t2 << endl;
	double gflops_2 = (double)num_ops * 1e-9 / ((t2_2-t2)/loop_time);
	fprintf(stderr, "the gflops of gemm_base_vec matrix %d %d %d is: %lf\n", m, n, l, gflops_2); 

	gemm_base_naive(a, b, c1, m, n, l);

	string res_file_base0 = res_file + ".base0";
	FILE* fp_res = fopen(res_file_base0.c_str(), "wb");
	assert(fp_res != NULL);
	int err_num = 0;
	for(int i = 0; i < m * n; i++){
		if(abs(c1[i]-c2[i]) > 1e-5){
			err_num++;
			fprintf(fp_res, "%d\t%f\t%f\n", i, c1[i], c2[i]);
		}
	}
	fclose(fp_res);

	double t3 = get_sec();
	cerr << "======time of check result gemm_base is: " << t3 - t2_2 << endl;
	cerr << "-----the error number is: " << err_num << endl;

	cerr << "finished!" << endl;

	return 0;
}
