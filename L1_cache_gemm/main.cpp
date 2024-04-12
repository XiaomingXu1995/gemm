#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <immintrin.h>

using namespace std;

static double get_time(struct timespec *start, struct timespec *end)
{
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

static void thread_bind(int cpu)
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu, &cpu_set);
    if (pthread_setaffinity_np(pthread_self(),
            sizeof(cpu_set_t), &cpu_set) != 0)
    {
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

static void page_free(void *mem, size_t size)
{
    munmap(mem, size);
}

void sgemm_naive(float *a, float *b, float *c, int m, int n, int k)
{
    int i, j, kk;
    for (i = 0; i < m; i++)
    {
            for (kk = 0; kk < k; kk++)
        for (j = 0; j < n; j++)
        {
            {
                c[i * n + j] += a[i * k + kk] * b[kk * n + j];
            }
        }
    }
}

void vec_sgemm(float* a, float* b, float* c, int m, int n, int k){
	__m256 v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;
	//n == 24
	
	int s_i = 0;
	int s_k = 0;
	n = 24;

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
		for(int kk = 0; kk < k; kk++){
			__m256 v12 = _mm256_load_ps(&b[kk*n + 0]);
			__m256 v13 = _mm256_load_ps(&b[kk*n + 8]);
			__m256 v14 = _mm256_load_ps(&b[kk*n + 16]);

			__m256 v15 = _mm256_set1_ps(a[(s_i+0)*k+kk]);
			v0 = _mm256_add_ps(v0, _mm256_mul_ps(v15, v12));
			v1 = _mm256_add_ps(v1, _mm256_mul_ps(v15, v13));
			v2 = _mm256_add_ps(v2, _mm256_mul_ps(v15, v14));

			v15 = _mm256_set1_ps(a[(s_i+1)*k+kk]);
			v3 = _mm256_add_ps(v3, _mm256_mul_ps(v15, v12));
			v4 = _mm256_add_ps(v4, _mm256_mul_ps(v15, v13));
			v5 = _mm256_add_ps(v5, _mm256_mul_ps(v15, v14));

			v15 = _mm256_set1_ps(a[(s_i+2)*k+kk]);
			v6 = _mm256_add_ps(v6, _mm256_mul_ps(v15, v12));
			v7 = _mm256_add_ps(v7, _mm256_mul_ps(v15, v13));
			v8 = _mm256_add_ps(v8, _mm256_mul_ps(v15, v14));

			v15 = _mm256_set1_ps(a[(s_i+3)*k+kk]);
			v9 = _mm256_add_ps(v9, _mm256_mul_ps(v15, v12));
			v10 = _mm256_add_ps(v10, _mm256_mul_ps(v15, v13));
			v11 = _mm256_add_ps(v11, _mm256_mul_ps(v15, v14));
		}
		_mm256_store_ps(&c[(s_i+0)*n+0], v0);
		_mm256_store_ps(&c[(s_i+0)*n+8], v1);
		_mm256_store_ps(&c[(s_i+0)*n+16], v2);

		_mm256_store_ps(&c[(s_i+1)*n+0], v3);
		_mm256_store_ps(&c[(s_i+1)*n+8], v4);
		_mm256_store_ps(&c[(s_i+1)*n+16], v5);

		_mm256_store_ps(&c[(s_i+2)*n+0], v6);
		_mm256_store_ps(&c[(s_i+2)*n+8], v7);
		_mm256_store_ps(&c[(s_i+2)*n+16], v8);

		_mm256_store_ps(&c[(s_i+3)*n+0], v9);
		_mm256_store_ps(&c[(s_i+3)*n+8], v10);
		_mm256_store_ps(&c[(s_i+3)*n+16], v11);
	}

}

int main(int argc, char* argv[]){
	int i;
	if (argc != 3)
	{
	    fprintf(stderr, "Usage: %s m k\n", argv[0]);
	    return 0;
	}
	
	int m = atoi(argv[1]);
	int k = atoi(argv[2]);
	long comp = 2L * m * k * 24L;
  int loop_time = (int)(2e11 / comp);

  struct timespec start, end;
  double t, gflops;

  thread_bind(0);

  float *a = (float*)page_alloc(m * k * sizeof(float));
  float *b = (float*)page_alloc(k * 24 * sizeof(float));
  float *c1 = (float*)page_alloc(m * 24 * sizeof(float));
  float *c2 = (float*)page_alloc(m * 24 * sizeof(float));

  srand(time(NULL));

  for (i = 0; i < m * k; i++)
  {
      a[i] = (float)rand() / (float)RAND_MAX;
  }
  for (i = 0; i < k * 24; i++)
  {
      b[i] = (float)rand() / (float)RAND_MAX;
  }

	sgemm_naive(a, b, c1, m, 24, k);
	vec_sgemm(a, b, c2, m, 24, k);
	string check_file = "file.diff";
	FILE* fp = fopen(check_file.c_str(), "w");
	for(int i = 0; i < m * 24; i++){
		if(abs(c1[i]-c2[i]) > 1e-5){
		//if(c1[i] != c2[i]){
			fprintf(fp, "%d\t%f\t%f\n", i, c1[i], c2[i]);
		}
	}
	fclose(fp);
	cerr << "finished the check!" << endl;

  // fma-tuned version
  // warm up
  for (i = 0; i < loop_time; i++)
  {
      //sgemm_kernel_x64_fma_m4n24(a, b, c2, m, k);
			vec_sgemm(a, b, c2, m, 24, k);
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  for (i = 0; i < loop_time; i++)
  {
		sgemm_naive(a, b, c1, m, 24, k);
			//vec_sgemm(a, b, c2, m, 24, k);
      //sgemm_kernel_x64_fma_m4n24(a, b, c2, m, k);
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  t = get_time(&start, &end) / loop_time;
  gflops = (double)comp / t * 1e-9;

  printf("sgemm_kernel_x64_fma(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, 24, k, t * 1e6, gflops);
  page_free(a, m * k * sizeof(float));
  page_free(b, k * 24 * sizeof(float));
  page_free(c1, m * 24 * sizeof(float));
  page_free(c2, m * 24 * sizeof(float));



}
