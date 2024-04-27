all: exe1 exe2 exe3

exe1: exe_generate_random exe_check 
exe2: exe_gemm_int 
exe3: exe_gemm_float exe_gemm_float_multiple

exe_generate_random: generate_random.cpp
	g++ -O3 -g generate_random.cpp -o exe_generate_random
exe_check: check.cpp
	g++ -O3 -g check.cpp -o exe_check
exe_gemm_float: gemm_float.cpp
	g++ -O3 -g gemm_float.cpp -o exe_gemm_float -mavx2 -fopenmp -ffast-math -fstrict-aliasing -mfma
exe_gemm_float_multiple: gemm_float_multiple.cpp
	g++ -O3 -g gemm_float_multiple.cpp -o exe_gemm_float_multiple -mavx2 -lpthread -mfma
exe_gemm_int: gemm_int.cpp
	g++ -O3 -g gemm_int.cpp -o exe_gemm_int -mavx2 -fopenmp -ffast-math -fstrict-aliasing -mfma

clean:
	rm -rf exe_*
