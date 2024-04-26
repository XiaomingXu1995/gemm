set -x
./exe_gemm_float 2400 2400 2400 res0
./exe_gemm_float_multiple 24 24 64 res1
./exe_gemm_int 2400 2400 2400 res2
