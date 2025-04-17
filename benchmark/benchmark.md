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

---
第二天对于结果不正确的问题进一步明确：
* 结果不正确的具体表现为：TFLOPS极高，kernel直接跳过；输出结果矩阵都为0.
* 因此说明kennel没有执行。

后续的分析如下：


### ***结果已经明确：***
TL DL:
一个kernel中分配的shared memory或者每个线程使用的寄存器过多，导致kernel不启动或者将数据放在栈上（太慢）。

在编译时加上`-Xptxas=v`参数，可以获取kernel的shared memory，寄存器以及offloading的具体情况。

例如：
之前的分配变量方式：
```cuda
  int32_t c_accum[BLOCK_SIZE_M][BLOCK_SIZE_N] = {0};
```
在使用`-Xptxas=v`进行编译时，可以得到如下log：
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

