#include <iostream>
#include <sys/stat.h>
#include <bits/stdc++.h>

using namespace std;

size_t get_file_size(const char* file){
	struct stat st;
	if(stat(file, &st) == 0){
		return st.st_size;
	}
	else{
		return -1;
	}
}

int main(int argc, char* argv[]){
	if(argc < 3){
		cerr << "run as: <check file0 file> to check the diffence between file0 and file1" << endl;
		return 1;
	}

	string file0 = argv[1];
	string file1 = argv[2];

	FILE* fp0 = fopen(file0.c_str(), "rb");
	assert(fp0 != NULL);
	FILE* fp1 = fopen(file1.c_str(), "rb");
	assert(fp1 != NULL);

	size_t file_size_0 = get_file_size(file0.c_str());
	size_t file_size_1 = get_file_size(file1.c_str());
	assert(file_size_0 == file_size_1);
	size_t num = file_size_0 / sizeof(int);
	int * arr0 = new int[num];
	int * arr1 = new int[num];
	int num_0 = fread(arr0, sizeof(int), num, fp0);
	assert(num_0 == num);
	int num_1 = fread(arr1, sizeof(int), num, fp1);
	assert(num_1 == num);
	cerr << "start check:" << endl;
	for(int i = 0; i < num; i++){
		if(arr0[i] != arr1[i]){
			cout << i << '\t' << arr0[i] << '\t' << arr1[i] << endl;
		}
	}

	cerr << "finished!" << endl;
	return 0;
}
