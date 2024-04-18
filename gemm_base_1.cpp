#include <iostream>
#include <vector>
#include <sys/time.h>
#include <assert.h>

using namespace std;

//const string src_file0 = "million0.file";
//const string src_file1 = "million1.file";
const string src_file0 = "billion0.int.random";
const string src_file1 = "billion1.int.random";

double get_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}

void gemm_base(int** matrix0, int** matrix1, int** res, int m, int n, int l){
	for(int i = 0; i < m; i++){
			for(int k = 0; k < l; k++){
		for(int j = 0; j < n; j++){
				res[i][j] += matrix0[i][k] * matrix1[k][j];
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

	int ** matrix0 = new int*[m]; //m * l
	int ** matrix1 = new int*[l]; //l * n
	FILE* fp0 = fopen(src_file0.c_str(), "rb");
	FILE* fp1 = fopen(src_file1.c_str(), "rb");
	assert(fp0 != NULL);
	assert(fp1 != NULL);
	
	for(int i = 0; i < m; i++){
		matrix0[i] = new int[l];
		int read = fread(matrix0[i], sizeof(int), l, fp0);
		assert(read == l);
	}
	fclose(fp0);

	for(int i = 0; i < l; i++){
		matrix1[i] = new int[n];
		int read = fread(matrix1[i], sizeof(int), n, fp1);
		assert(read == n);
	}
	fclose(fp1);

	int** res = new int*[m];
	for(int i = 0; i < m; i++){
		res[i] = new int[n];
		for(int j = 0; j < n; j++){
			res[i][j] = 0;
		}
	}

	long num_ops = 2L * m * n * l;

	double t1 = get_sec();
	cerr << "======time of init matrix is: " << t1 - t0 << endl;

	gemm_base(matrix0, matrix1, res, m, n, l);

	double t2 = get_sec();
	cerr << "======time of gemm_base is: " << t2 - t1 << endl;
	double gflops = (double)num_ops * 1e-9 / (t2-t1);
	fprintf(stderr, "the gflops of matrix %d %d %d is: %lf\n", m, n, l, gflops); 

	string res_file_base0 = res_file + ".base0";
	FILE* fp_res = fopen(res_file_base0.c_str(), "wb");
	assert(fp_res != NULL);
	for(int i = 0; i < m; i++){
		int written = fwrite(res[i], sizeof(int), n, fp_res);
		assert(written == n);
	}
	fclose(fp_res);

	double t3 = get_sec();
	cerr << "======time of save result gemm_base is: " << t3 - t2 << endl;


	




	cerr << "finished!" << endl;

	return 0;
}
