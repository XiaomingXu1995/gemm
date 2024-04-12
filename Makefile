all: exe1 exe2 

exe1: exe_generate_random exe_check 
exe2: exe_gemm_base exe_gemm_base_1 exe_gemm_base_2 exe_gemm_base_3

exe_generate_random: generate_random.cpp
	g++ -O3 -g generate_random.cpp -o exe_generate_random
exe_gemm_base: gemm_base.cpp
	g++ -O3 -g gemm_base.cpp -o exe_gemm_base
exe_gemm_base_1: gemm_base_1.cpp
	g++ -O3  -g gemm_base_1.cpp -o exe_gemm_base_1
exe_check: check.cpp
	g++ -O3 -g check.cpp -o exe_check
exe_gemm_base_2: gemm_base_2.cpp
	g++ -O3 -g gemm_base_2.cpp -o exe_gemm_base_2 -fopenmp
exe_gemm_base_3: gemm_base_3.cpp
	g++ -O3 -g gemm_base_3.cpp -o exe_gemm_base_3 -fopenmp -mavx2

clean:
	rm -rf exe_*
