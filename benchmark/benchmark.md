# develop benchmark log



## gemm shared memory 分析：(将文件改名为gemm_shared_memory.cu)

### 问题：有时候性能极差，有时候结果不对
gemm_mma 中，写的shared_memory 方式，计算gemm，有一个问题：
```cuda
mm_mma_int8<64, 64, 128, 4, 4>(warmup, repeat, m, n, k);
mm_mma_float<96, 32, 64, 6, 4>(warmup, repeat, m, n, k);
```
当设置的blockDim 中的线程数目变化的时候，结果会不正确。（划分方式不同）这是什么问题导致的呢？

之前的实现，有些问题，开的accum_c的数据过多，导致性能变慢。但是现在的操作，有时候，结果不稳定，这个需要明确一下。

第二天对于结果不正确的问题进一步明确：
* 结果不正确的具体表现为：TFLOPS极高，kernel直接跳过；输出结果矩阵都为0.
* 因此说明kennel没有执行。

后续的分析如下：


### ***结果已经明确：***
TL DL:
一个kernel中分配的shared memory或者每个线程使用的寄存器过多，导致kernel不启动或者将数据放在栈上（太慢）。

在编译时加上`-Xptxas=-v`参数，可以获取kernel的shared memory，寄存器以及offloading的具体情况。

例如：
之前的分配变量方式：
```cuda
  int32_t c_accum[BLOCK_SIZE_M][BLOCK_SIZE_N] = {0};
```
在使用`-Xptxas=-v`进行编译时，可以得到如下log：
```
ptxas info    : 282 bytes gmem, 56 bytes cmem[4]
ptxas info    : Compiling entry function '_Z14matrixMulOnGPUIaiEvPT_S1_PT0_iii' for 'sm_89'
ptxas info    : Function properties for _Z14matrixMulOnGPUIaiEvPT_S1_PT0_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 33 registers, 388 bytes cmem[0]
ptxas info    : Compiling entry function '_Z8gemm_mmaILi64ELi64ELi128EEvPaS0_Piiii' for 'sm_89'
ptxas info    : Function properties for _Z8gemm_mmaILi64ELi64ELi128EEvPaS0_Piiii
    16384 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, 16384 bytes cumulative stack size, 16384 bytes smem, 388 bytes cmem[0]
```
可以看到，kernel `gemm_mma<64, 64, 128> 有16384 bytes 放到了stack上。
* Stack frame 是每个线程私有的空间，用于保存局部数组、结构体、过大的局部变量；
* 位于 local memory（全局内存中分配），访问很慢；


后来的kernel编译的时候，log如下：
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z8gemm_mmaILi64ELi64ELi128ELi2ELi4EEvPaS0_Piiii' for 'sm_89'
ptxas info    : Function properties for _Z8gemm_mmaILi64ELi64ELi128ELi2ELi4EEvPaS0_Piiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 95 registers, 16384 bytes smem, 388 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14matrixMulOnGPUIaiEvPT_S1_PT0_iii' for 'sm_89'
ptxas info    : Function properties for _Z14matrixMulOnGPUIaiEvPT_S1_PT0_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 33 registers, 388 bytes cmem[0]
```
可以看到每一个thread占用了95个register。

一个SM有65536个register，一个warp包含32个thread。
如果blockDim（32，32）中1024 个线程全部开满的话，每个线程平均分到64个register。
如果每个线程的register数目超过了64，那么设置blockDim为（32，32）的时候，就会直接跳过kernel的执行（也不会报错），表现出来就是高的离谱的 FLOPS和 结果矩阵全部为0（压根没有计算）。

当设置blockDim不是（32，32）时，满足寄存器数目不超过SM总数的时候，就能够正常计算完成。

### 总结：
上边两个问题：kernel太慢和结果不对，分别通过上边两个log以及分析得到了明确的答案。
* 结果太慢：使用了位于global memory上的stack
* 结果不对：寄存器数目超出限制，kernel没有执行

### tile 参数与TFLOPS的关系

表格中的各项内容：
矩阵A 和矩阵 B 在kernel中 分配的 shared memory 为： A[block_m][block_k]，B[block_k][block_n].

kernel thread 分配 blockDim(block_n / tile_col, block_m / tile_row)。

shm 为shared memory占用大小，cmem为kernel中const 值占用大小。

shm, cmem, register, stack frame, spill stores, spill loads的内容都是由编译时加`-Xptxas=-v`来获取的。

TFLOPS 是由计时函数计算得来。

datatype_A = int8_t, datatype_B = int8_t, datatype_C = int32_t。

矩阵维度为m=2048, n= 2048, k=2048. 当m n k 都为4096时，结果差不太多，都为1024时，TFLOPS相对较低。


但是实际测试的时候，m n k 都为4096时， <256, 128, 128, 16 8> 的TFLOPS为28.284， <128, 256, 128, 8, 16>的TFLOPS为30.618。

| block_m | block_n | block_k | tile_row| tile_col |TFLOPS|blockDim|gridDim|shm|cmem|register|stack frame| spill stores| spill loads|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|256|128|128|16|8|33.337| (16, 16) | (16, 8)|49152B|388B|254,| 0| 0| 0|
|128|256|128|8|16|31.025| (16, 16) | (8, 16)|49152B|388B|255, 96B cumulative stack size| 96B| 284B| 196B|
|128|128|128|16|8|28.078| (16, 8) | (16, 16)|32768B|388B|255| 0| 0| 0|
|128|128|128|8|16|25.791| (8, 16) | (16, 16)|32768B|388B|255, 96B cumulative stack size| 96B| 284B| 196B|

具体的编译log：
```txt
nvcc -Xptxas=-v -lineinfo -O3 -g gemm_shared_memory.cu -o build/gemm_shared_memory -std=c++17 -gencode arch=compute_89,code=sm_89
ptxas info    : 963 bytes gmem, 80 bytes cmem[4]
ptxas info    : Compiling entry function '_Z8gemm_mmaILi128ELi128ELi128ELi8ELi16EEvPaS0_Piiii' for 'sm_89'
ptxas info    : Function properties for _Z8gemm_mmaILi128ELi128ELi128ELi8ELi16EEvPaS0_Piiii
    96 bytes stack frame, 284 bytes spill stores, 196 bytes spill loads
ptxas info    : Used 255 registers, 96 bytes cumulative stack size, 32768 bytes smem, 388 bytes cmem[0]
ptxas info    : Compiling entry function '_Z8gemm_mmaILi128ELi128ELi128ELi16ELi8EEvPaS0_Piiii' for 'sm_89'
ptxas info    : Function properties for _Z8gemm_mmaILi128ELi128ELi128ELi16ELi8EEvPaS0_Piiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 255 registers, 32768 bytes smem, 388 bytes cmem[0]
ptxas info    : Compiling entry function '_Z8gemm_mmaILi128ELi256ELi128ELi8ELi16EEvPaS0_Piiii' for 'sm_89'
ptxas info    : Function properties for _Z8gemm_mmaILi128ELi256ELi128ELi8ELi16EEvPaS0_Piiii
    96 bytes stack frame, 284 bytes spill stores, 196 bytes spill loads
ptxas info    : Used 255 registers, 96 bytes cumulative stack size, 49152 bytes smem, 388 bytes cmem[0]
ptxas info    : Compiling entry function '_Z8gemm_mmaILi256ELi128ELi128ELi16ELi8EEvPaS0_Piiii' for 'sm_89'
ptxas info    : Function properties for _Z8gemm_mmaILi256ELi128ELi128ELi16ELi8EEvPaS0_Piiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 254 registers, 49152 bytes smem, 388 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14matrixMulOnGPUIaiEvPT_S1_PT0_iii' for 'sm_89'
ptxas info    : Function properties for _Z14matrixMulOnGPUIaiEvPT_S1_PT0_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 33 registers, 388 bytes cmem[0]
```

### 编译时间：

不加 -Xptxas=-v, 编译

* mm_mma_int8<128, 128, 128, 16, 8>(warmup, repeat, m, n, k); 编译时间 30.3s。
* mm_mma_int8<256, 128, 128, 16, 8>(warmup, repeat, m, n, k); 编译时间29.8s。

加上-Xptxas=-v：
* mm_mma_int8<256, 128, 128, 16, 8>(warmup, repeat, m, n, k); 编译时间28.1s。

再加上-lineinfo:
* mm_mma_int8<256, 128, 128, 16, 8>(warmup, repeat, m, n, k); 编译时间30.5s。


加上-Xptxas=-v -lineinfo 之后，编译下边三个kernel：
```chda
  mm_mma_int8<64, 32, 128, 4, 2>(warmup, repeat, m, n, k);
  mm_mma_int8<32, 32, 128, 2, 2>(warmup, repeat, m, n, k);
  mm_mma_int8<16, 16, 128, 1, 1>(warmup, repeat, m, n, k);
```
总共用时6.3s。

### 为什么大型 kernel 编译慢？(来自 chatGPT)
#### 原因 1：模板参数放大了代码生成规模
像 `mm_mma_int8<256,128,128,16,8>` 这种参数会生成一个非常大的 kernel 函数体：

* 包含嵌套 3~4 层的循环；
* #pragma unroll 展开后是成百上千行的 PTX；
* 很多乘法 / 累加 / 条件判断会内联展开；
* 每个 thread 的寄存器分配压力非常高 → 编译器必须做寄存器分配 + spill 分析 + warp 分配模拟

#### 原因 2：大 tile 会产生复杂的共享内存访问模式
```cpp
__shared__ int8_t smem_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
```

当 BLOCK_SIZE_M=256, BLOCK_SIZE_K=128，就意味着你静态声明了 32KB 的共享内存 tile。

* 编译器要检查是否有 warp bank conflict；
* 如果 #pragma unroll 全展开后访问维度混乱，还得做地址别名分析；
* 所有这些分析，都是编译时间开销的元凶。


#### 原因 3：中间值展开生成成千上万 PTX 指令
举个例子：
```cpp
for (int i = 0; i < TILES_COMPUTE_ROW; ++i)
  for (int j = 0; j < TILES_COMPUTE_COL; ++j)
    acc[i][j] += ...
```
如果 i=16, j=8，这是 128 组累加器寄存器，每次迭代生成 128 条指令，loop unroll 后乘上 BLOCK_SIZE_K，你就有成千上万条 PTX → 巨大的 SASS 生成开销。

### 可以考虑ccache进行编译加速。
