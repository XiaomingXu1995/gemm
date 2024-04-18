all: exe1 exe2 exe3

exe1: exe_generate_random exe_check 
exe2: exe_gemm_base exe_gemm_base_1 exe_gemm_base_2 exe_gemm_base_3
exe3: exe_gemm_float exe_gemm_float_mem_align exe_gemm_float_multiple

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
exe_gemm_float: gemm_float.cpp
	g++ -O3 -g gemm_float.cpp -o exe_gemm_float -mavx2 -fopenmp -ffast-math -fstrict-aliasing -mfma
exe_gemm_float_mem_align: gemm_float_mem_align.cpp
	g++ -O3 -g gemm_float_mem_align.cpp -o exe_gemm_float_mem_align -mavx2 -fopenmp
exe_gemm_float_multiple: gemm_float_multiple.cpp
	g++ -O3 -g gemm_float_multiple.cpp -o exe_gemm_float_multiple -mavx2 -lpthread -mfma

clean:
	rm -rf exe_*
