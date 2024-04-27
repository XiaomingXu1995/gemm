#include <iostream>
#include <vector>
#include <sys/time.h>
#include <assert.h>
#include <immintrin.h>

using namespace std;

const string src_file0 = "billion0.int.random";
const string src_file1 = "billion1.int.random";

double get_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}

void gemm_base_naive(int* a, int* b, int* c, int m, int n, int l){
	for(int i = 0; i < m; i++){
			for(int k = 0; k < l; k++){
		for(int j = 0; j < n; j++){
				c[i * n + j] += a[i * l + k] * b[k * n + j];
			}
		}
	}
}

void print_avx2(__m256i v){
	int arr[8];
	_mm256_store_si256((__m256i*)arr, v);
	for(int i = 0; i < 8; i++){
		cout << arr[i] << '\t';
	}
	cout << endl;
}

void gemm_base_vec(int* a, int* b, int* c, int m, int n, int l){
	for(int s_j = 0; s_j < n; s_j+=24){
		for(int s_i = 0; s_i < m; s_i+=4){
			__m256i v0 = _mm256_set1_epi32(0);
			__m256i v1 = v0; 
			__m256i v2 = v0; 
			__m256i v3 = v0; 
			__m256i v4 = v0; 
			__m256i v5 = v0; 
			__m256i v6 = v0; 
			__m256i v7 = v0; 
			__m256i v8 = v0; 
			__m256i v9 = v0; 
			__m256i v10 = v0; 
			__m256i v11 = v0; 
			for(int kk = 0; kk < l; kk++){
				// need avx512 CPU flags
				//__m256i v12 = _mm256_load_epi32(&b[kk*n+s_j + 0]);
				//__m256i v13 = _mm256_load_epi32(&b[kk*n+s_j + 8]);
				//__m256i v14 = _mm256_load_epi32(&b[kk*n+s_j + 16]);
				
				__m256i v12 = _mm256_load_si256((__m256i*)&b[kk*n+s_j + 0]);
				__m256i v13 = _mm256_load_si256((__m256i*)&b[kk*n+s_j + 8]);
				__m256i v14 = _mm256_load_si256((__m256i*)&b[kk*n+s_j + 16]);


				__m256i v15 = _mm256_set1_epi32(a[(s_i+0)*l+kk]);
				v0 = _mm256_add_epi32(v0, _mm256_mullo_epi32(v15, v12));
				v1 = _mm256_add_epi32(v1, _mm256_mullo_epi32(v15, v13));
				v2 = _mm256_add_epi32(v2, _mm256_mullo_epi32(v15, v14));

				v15 = _mm256_set1_epi32(a[(s_i+1)*l+kk]);
				v3 = _mm256_add_epi32(v3, _mm256_mullo_epi32(v15, v12));
				v4 = _mm256_add_epi32(v4, _mm256_mullo_epi32(v15, v13));
				v5 = _mm256_add_epi32(v5, _mm256_mullo_epi32(v15, v14));

				v15 = _mm256_set1_epi32(a[(s_i+2)*l+kk]);
				v6 = _mm256_add_epi32(v6, _mm256_mullo_epi32(v15, v12));
				v7 = _mm256_add_epi32(v7, _mm256_mullo_epi32(v15, v13));
				v8 = _mm256_add_epi32(v8, _mm256_mullo_epi32(v15, v14));

				v15 = _mm256_set1_epi32(a[(s_i+3)*l+kk]);
				v9 = _mm256_add_epi32(v9, _mm256_mullo_epi32(v15, v12));
				v10 = _mm256_add_epi32(v10, _mm256_mullo_epi32(v15, v13));
				v11 = _mm256_add_epi32(v11, _mm256_mullo_epi32(v15, v14));
			}
			_mm256_store_si256((__m256i*)&c[(s_i+0)*n+s_j+0], v0);
			_mm256_store_si256((__m256i*)&c[(s_i+0)*n+s_j+8], v1);
			_mm256_store_si256((__m256i*)&c[(s_i+0)*n+s_j+16], v2);

			_mm256_store_si256((__m256i*)&c[(s_i+1)*n+s_j+0], v3);
			_mm256_store_si256((__m256i*)&c[(s_i+1)*n+s_j+8], v4);
			_mm256_store_si256((__m256i*)&c[(s_i+1)*n+s_j+16], v5);

			_mm256_store_si256((__m256i*)&c[(s_i+2)*n+s_j+0], v6);
			_mm256_store_si256((__m256i*)&c[(s_i+2)*n+s_j+8], v7);
			_mm256_store_si256((__m256i*)&c[(s_i+2)*n+s_j+16], v8);

			_mm256_store_si256((__m256i*)&c[(s_i+3)*n+s_j+0], v9);
			_mm256_store_si256((__m256i*)&c[(s_i+3)*n+s_j+8], v10);
			_mm256_store_si256((__m256i*)&c[(s_i+3)*n+s_j+16], v11);
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

	//int* a = new int[m * l];
	//int* b = new int[l * n];
	//int * c1 = new int[m * n];
	//int * c2 = new int[m * n];
	int* a = (int*)_mm_malloc(m * l * sizeof(int), 64);
	int* b = (int*)_mm_malloc(l * n * sizeof(int), 64);
	int* c1 = (int*)_mm_malloc(m * n * sizeof(int), 64);
	int* c2 = (int*)_mm_malloc(m * n * sizeof(int), 64);

	FILE* fp0 = fopen(src_file0.c_str(), "rb");
	FILE* fp1 = fopen(src_file1.c_str(), "rb");
	assert(fp0 != NULL);
	assert(fp1 != NULL);

	int read_a = fread(a, sizeof(int), m * l, fp0);
	assert(read_a == m * l);
	fclose(fp0);

	int read_b = fread(b, sizeof(int), l * n, fp1);
	assert(read_b == l * n);
	fclose(fp1);

	for(int i = 0; i < m * n; i++){
		c1[i] = 0;
		c2[i] = 0;
	}
	

	long num_ops = 2L * m * n * l;

	double t1 = get_sec();
	cerr << "======time of init matrix is: " << t1 - t0 << endl;

	gemm_base_naive(a, b, c1, m, n, l);

	double t2 = get_sec();
	cerr << "======time of gemm_base_naive is: " << t2 - t1 << endl;
	double mips = (double)num_ops * 1e-6 / (t2-t1);
	fprintf(stderr, "the MIPS(Million Instructions Per Second) of gemm_base_naive matrix %d %d %d is: %lf\n", m, n, l, mips); 

	gemm_base_vec(a, b, c2, m, n, l);
	double t3 = get_sec();
	cerr << "======time of gemm_base_vec is: " << t3 - t2 << endl;
	double mips_2 = (double)num_ops * 1e-6 / (t3-t2);
	fprintf(stderr, "the MIPS(Million Instructions Per Second) of gemm_base_vec matrix %d %d %d is: %lf\n", m, n, l, mips_2); 

	string res_file_base0 = res_file + ".base0";
	FILE* fp_res = fopen(res_file_base0.c_str(), "wb");
	assert(fp_res != NULL);
	int err_num = 0;
	for(int i = 0; i < m * n; i++){
		if(c1[i] != c2[i]){
			err_num++;
			fprintf(fp_res, "%d\t%d\t%d\n", i, c1[i], c2[i]);
		}
	}
	fclose(fp_res);

	double t4 = get_sec();
	cerr << "======time of check result gemm_base is: " << t4 - t3 << endl;
	cerr << "-----the error number is: " << err_num << endl;

	delete [] a;
	delete [] b;
	delete [] c1;
	delete [] c2;


	cerr << "finished!" << endl;

	return 0;
}
