#include <iostream>
#include <vector>
#include <sys/time.h>
#include <assert.h>

using namespace std;

//const string src_file0 = "million0.file";
//const string src_file1 = "million1.file";
const string src_file0 = "billion0.random";
const string src_file1 = "billion1.random";

double get_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
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

	float* a = new float[m * l];
	float* b = new float[l * n];
	float * c1 = new float[m * n];
	float * c2 = new float[m * n];

	FILE* fp0 = fopen(src_file0.c_str(), "rb");
	FILE* fp1 = fopen(src_file1.c_str(), "rb");
	assert(fp0 != NULL);
	assert(fp1 != NULL);

	int read_a = fread(a, sizeof(float), m * l, fp0);
	assert(read_a == m * l);
	fclose(fp0);

	int read_b = fread(b, sizeof(float), l * n, fp1);
	assert(read_b == l * n);
	fclose(fp1);

	for(int i = 0; i < m * n; i++){
		c1[i] = 0.0;
		c2[i] = 0.0;
	}
	
	//int** res = new int*[m];
	//for(int i = 0; i < m; i++){
	//	res[i] = new int[n];
	//	for(int j = 0; j < n; j++){
	//		res[i][j] = 0;
	//	}
	//}

	long num_ops = 2L * m * n * l;

	double t1 = get_sec();
	cerr << "======time of init matrix is: " << t1 - t0 << endl;

	gemm_base_naive(a, b, c1, m, n, l);
	//gemm_base(matrix0, matrix1, res, m, n, l);

	double t2 = get_sec();
	cerr << "======time of gemm_base is: " << t2 - t1 << endl;
	double gflops = (double)num_ops * 1e-9 / (t2-t1);
	fprintf(stderr, "the gflops of matrix %d %d %d is: %lf\n", m, n, l, gflops); 

	string res_file_base0 = res_file + ".base0";
	FILE* fp_res = fopen(res_file_base0.c_str(), "wb");
	assert(fp_res != NULL);
	int written = fwrite(c1, sizeof(float), m * n, fp_res);
	assert(written == m * n);
	fclose(fp_res);

	double t3 = get_sec();
	cerr << "======time of save result gemm_base is: " << t3 - t2 << endl;

	cerr << "finished!" << endl;

	return 0;
}
