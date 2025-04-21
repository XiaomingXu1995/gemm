all: mkdir_build exe1 exe2 exe3 exe4 exe5

exe1: build/generate_random build/check
exe2: build/gemm_int 
exe3: build/gemm_float build/gemm_float_multiple
exe4: build/gemm_cublas build/gemm_cuda build/gemm_tile_fusion build/gemm_shared_memory #build/gemm_cublas_int8
exe5: build/sage_attention_test

mkdir_build:
	mkdir -p build

build/generate_random: generate_random.cpp
	g++ -O3 -g generate_random.cpp -o build/generate_random
build/check: check.cpp
	g++ -O3 -g check.cpp -o build/check
build/gemm_float: gemm_float.cpp
	g++ -O3 -g gemm_float.cpp -o build/gemm_float -mavx2 -fopenmp -ffast-math -fstrict-aliasing -mfma
build/gemm_float_multiple: gemm_float_multiple.cpp
	g++ -O3 -g gemm_float_multiple.cpp -o build/gemm_float_multiple -mavx2 -lpthread -mfma
build/gemm_int: gemm_int.cpp
	g++ -O3 -g gemm_int.cpp -o build/gemm_int -mavx2 -fopenmp -ffast-math -fstrict-aliasing -mfma
build/gemm_cublas: gemm_cublas.cu
	nvcc -O3 -g gemm_cublas.cu -o build/gemm_cublas -lcublas -std=c++17 -gencode arch=compute_89,code=sm_89
build/gemm_cuda: gemm_cuda.cu
	nvcc -O3 -g gemm_cuda.cu -o build/gemm_cuda -std=c++17
build/gemm_tile_fusion: gemm_tile_fusion.cu
	nvcc -O3 -g gemm_tile_fusion.cu -o build/gemm_tile_fusion -Xcompiler -fopenmp
build/gemm_cublas_int8: gemm_cublas_int8.cu
	nvcc -O3 -g gemm_cublas_int8.cu -o build/gemm_cublas_int8 -lcublas
build/gemm_shared_memory: gemm_shared_memory.cu
	nvcc -Xptxas=-v -lineinfo -O3 -g gemm_shared_memory.cu -o build/gemm_shared_memory -std=c++17 -gencode arch=compute_89,code=sm_89
build/sage_attention_test: sage_attention_test.cu
	nvcc -Xptxas=-v -lineinfo -O3 -g sage_attention_test.cu -o build/sage_attention_test -I sage_dir/ -I sage_dir/qattn/ -std=c++17 -gencode arch=compute_89,code=sm_89

clean:
	rm -rf build/*
