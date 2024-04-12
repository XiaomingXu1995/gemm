#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
	if(argc < 3){
		cerr << "run as: <exe_generate_random 10000 res.file> to generate 10000 random integer and store into res.file" << endl;
		return 1;
	}
	int nums = stoi(argv[1]);
	string res_file = argv[2];
  // 创建随机数引擎，这里使用了默认的随机数引擎
	
	std::random_device rd;
  std::default_random_engine generator(rd());

  // 创建一个分布，这里使用的是均匀分布，指定了生成随机数的范围为1到100
  std::uniform_int_distribution<int> distribution(1, 20);

	vector<int> res(nums, 0);
	for(int i = 0; i < nums; i++){
  	int randomNumber = distribution(generator);
		res[i] = randomNumber;
	}

	FILE * fp = fopen(res_file.c_str(), "wb");
	if(!fp){
		cerr << "error open the file: " << res_file << endl;
		return 2;
	}
	size_t written = fwrite(res.data(), sizeof(int), nums, fp); 
	if(written != nums){
		cerr << "failed to write all data into file" << endl;
		fclose(fp);
		return 3;
	}
	fclose(fp);


	cout << "finished!" << endl;

  // 生成随机数

  // 输出随机数
  //std::cout << "Random Number: " << randomNumber << std::endl;

  return 0;
}

