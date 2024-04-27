# Overview
The vectorization idea of gemm by SIMD instructions comes from the ***zhihu*** (https://zhuanlan.zhihu.com/p/383115932) and the (https://github.com/pigirons/sgemm_hsw).

[zhihu](https://github.com/pigirons/sgemm_hsw) gives a detailed description of the methods with perspicuous pictures. 

# Build
`make -j8`

# Init the input 
`./init.sh`

This is used for initialization of the input matrix.
Input matrices of `A[m][n]` and `B[n][k]` are read from the `*.random` files.
Make sure the `m*n` and `n*k` are less than the element number of `.random` files.

# Run the gemm
`./exe_gemm_float m n k res` It means that  `C[m][k]=A[m][n]xB[n][k]`.
As is shown in [zhihu](https://github.com/pigirons/sgemm_hsw), the `n` should be a multiple of `24` to fully use of the `16` `ymm` logical register.

For example:  
`./exe_gemm_float 2400 2400 2400 res`

`./exe_gemm_float_multiple 24 24 64 res`

`./run.sh`
