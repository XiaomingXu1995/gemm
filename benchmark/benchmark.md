# develop benchmark log



---
gemm_mma 中，写的shared_memory 方式，计算gemm，有一个问题：
```cuda
mm_mma_int8<64, 64, 128, 4, 4>(warmup, repeat, m, n, k);
mm_mma_float<96, 32, 64, 6, 4>(warmup, repeat, m, n, k);
```
当设置的blockDim 中的线程数目变化的时候，结果会不正确。（划分方式不同）这是什么问题导致的呢？

之前的实现，有些问题，开的accum_c的数据过多，导致性能变慢。但是现在的操作，有时候，结果不稳定，这个需要明确一下。

