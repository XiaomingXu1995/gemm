#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

template <typename T>
void get_random_int(int nums, string res_file){
  // 创建随机数引擎，这里使用了默认的随机数引擎
	std::random_device rd;
  std::default_random_engine generator(rd());

  // 创建一个分布，这里使用的是均匀分布，指定了生成随机数的范围为1到100
  std::uniform_int_distribution<int> distribution(1, 20);
  //std::uniform_real_distribution<T> distribution(0.0f, 20.0f);

	vector<T> res(nums, 0);
	for(int i = 0; i < nums; i++){
  	T randomNumber = distribution(generator);
		res[i] = randomNumber;
	}

	FILE * fp = fopen(res_file.c_str(), "wb");
	if(!fp){
		cerr << "error open the file: " << res_file << endl;
		exit(1);
	}
	size_t written = fwrite(res.data(), sizeof(T), nums, fp); 
	if(written != nums){
		cerr << "failed to write all data into file" << endl;
		fclose(fp);
		exit(2);
	}
	cerr << "save the result in: " << res_file << endl;
	fclose(fp);

}

template <typename T>
void get_random_float(int nums, string res_file){
  // 创建随机数引擎，这里使用了默认的随机数引擎
	std::random_device rd;
  std::default_random_engine generator(rd());

  // 创建一个分布，这里使用的是均匀分布，指定了生成随机数的范围为1到100
  //std::uniform_int_distribution<int> distribution(1, 20);
  std::uniform_real_distribution<T> distribution(0.0f, 20.0f);

	vector<T> res(nums, 0);
	for(int i = 0; i < nums; i++){
  	T randomNumber = distribution(generator);
		res[i] = randomNumber;
	}

	FILE * fp = fopen(res_file.c_str(), "wb");
	if(!fp){
		cerr << "error open the file: " << res_file << endl;
		exit(1);
	}
	size_t written = fwrite(res.data(), sizeof(T), nums, fp); 
	if(written != nums){
		cerr << "failed to write all data into file" << endl;
		fclose(fp);
		exit(2);
	}
	cerr << "save the result in: " << res_file << endl;
	fclose(fp);

}

int main(int argc, char* argv[]) {
	if(argc < 4){
		cerr << "run as: <exe_generate_random int 10000 res.file> to generate 10000 random integer and store into res.file" << endl;
		return 1;
	}
	string type = argv[1];
	int nums = stoi(argv[2]);
	string res_file = argv[3];

	if(type == "int"){
		get_random_int<int>(nums, res_file);
	}
	else if(type == "float"){
		get_random_float<float>(nums, res_file);
	}
	else{
		cerr << "error type, need <float> or <int>, run as <exe_generate_random int 10000 res.file> to generate 10000 random integer and store into res.file" << endl;
		return 1;
	}
	
	cout << "finished!" << endl;

  return 0;
}

