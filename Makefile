all: exe1 exe2 exe3 exe4

exe1: exe_generate_random exe_check 
exe2: exe_gemm_int 
exe3: exe_gemm_float exe_gemm_float_multiple
exe4: exe_gemm_cublas exe_gemm_cuda exe_gemm_tile_fusion



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
exe_gemm_cublas: gemm_cublas.cu
	nvcc -O3 -g gemm_cublas.cu -o exe_gemm_cublas -lcublas
exe_gemm_cuda: gemm_cuda.cu
	nvcc -O3 -g gemm_cuda.cu -o exe_gemm_cuda 
exe_gemm_tile_fusion: gemm_tile_fusion.cu
	nvcc -O3 -g gemm_tile_fusion.cu -o exe_gemm_tile_fusion -Xcompiler -fopenmp

clean:
	rm -rf exe_*
