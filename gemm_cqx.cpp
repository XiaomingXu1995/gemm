#include <iostream>
#include <random>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <pthread.h>
#include <sys/time.h>

//7.39G
//void sgemm_naive(float *A, float *B, float *C, int m, int k)
//{
//    for(int i = 0; i < m; ++i)
//    {
//        for(int j = 0; j < 24; ++j)
//        {
//            for(int kk = 0; kk < k; ++kk)
//                C[i * 24 + j] += A[i * k + kk] * B[kk * 24 + j];
//        }
//    }
//}

//32.01G
//void sgemm_naive(float *A, float *B, float *C, int m, int k)
//{
//    for(int i = 0; i < m; ++i)
//    {
//        for(int kk = 0; kk < k; ++kk)
//        {
//            for(int j = 0; j < 24; ++j)
//                C[i * 24 + j] += A[i * k + kk] * B[kk * 24 + j];
//        }
//    }
//}

//36.30G
//void sgemm_naive(float *A, float *B, float *C, int m, int k)
//{
//    float *AA = A;
//    float *CC = C;
//
//    for(int i = 0; i < m; ++i)
//    {
//        float *BB = B;
//
//        __m256 vc0 = _mm256_loadu_ps(CC + 0);
//        __m256 vc1 = _mm256_loadu_ps(CC + 8);
//        __m256 vc2 = _mm256_loadu_ps(CC + 16);
//
//
//        for(int kk = 0; kk < k; ++kk)
//        {
//            __m256 va0 = _mm256_set1_ps(*AA);
//
//            __m256 vb0 = _mm256_loadu_ps(BB + 0);
//            __m256 vb1 = _mm256_loadu_ps(BB + 8);
//            __m256 vb2 = _mm256_loadu_ps(BB + 16);
//
//            vc0 = _mm256_fmadd_ps(va0, vb0, vc0);
//            vc1 = _mm256_fmadd_ps(va0, vb1, vc1);
//            vc2 = _mm256_fmadd_ps(va0, vb2, vc2);
//
//            AA += 1;
//            BB += 24;
//        }
//
//        _mm256_storeu_ps(CC + 0, vc0);
//        _mm256_storeu_ps(CC + 8, vc1);
//        _mm256_storeu_ps(CC + 16, vc2);
//
//        CC += 24;
//    }
//}

//36.33G
//void sgemm_naive(float *A, float *B, float *C, int m, int k)
//{
//    float *AA = A;
//    float *CC = C;
//
//    for(int i = 0; i < m; ++i)
//    {
//        float *BB = B;
//
//        __m256 vc0 = _mm256_loadu_ps(CC + 0);
//        __m256 vc1 = _mm256_loadu_ps(CC + 8);
//        __m256 vc2 = _mm256_loadu_ps(CC + 16);
//
//
//        for(int kk = 0; kk < k; kk += 2)
//        {
//            __m256 va0 = _mm256_set1_ps(AA[0]);
//            __m256 va1 = _mm256_set1_ps(AA[1]);
//
//            __m256 vb0 = _mm256_loadu_ps(BB + 0);
//            __m256 vb1 = _mm256_loadu_ps(BB + 8);
//            __m256 vb2 = _mm256_loadu_ps(BB + 16);
//
//            __m256 vb3 = _mm256_loadu_ps(BB + 24);
//            __m256 vb4 = _mm256_loadu_ps(BB + 32);
//            __m256 vb5 = _mm256_loadu_ps(BB + 40);
//
//            vc0 = _mm256_fmadd_ps(va0, vb0, vc0);
//            vc1 = _mm256_fmadd_ps(va0, vb1, vc1);
//            vc2 = _mm256_fmadd_ps(va0, vb2, vc2);
//
//            vc0 = _mm256_fmadd_ps(va1, vb3, vc0);
//            vc1 = _mm256_fmadd_ps(va1, vb4, vc1);
//            vc2 = _mm256_fmadd_ps(va1, vb5, vc2);
//
//            AA += 2;
//            BB += 48;
//        }
//
//        _mm256_storeu_ps(CC + 0, vc0);
//        _mm256_storeu_ps(CC + 8, vc1);
//        _mm256_storeu_ps(CC + 16, vc2);
//
//        CC += 24;
//    }
//}

//65.07G
//void sgemm_naive(float *A, float *B, float *C, int m, int k)
//{
//    float *AA = A;
//    float *CC = C;
//
//    for(int i = 0; i < m; i += 2)
//    {
//        float *BB = B;
//
//        __m256 vc0 = _mm256_loadu_ps(CC + 0);
//        __m256 vc1 = _mm256_loadu_ps(CC + 8);
//        __m256 vc2 = _mm256_loadu_ps(CC + 16);
//
//        __m256 vc3 = _mm256_loadu_ps(CC + 24);
//        __m256 vc4 = _mm256_loadu_ps(CC + 32);
//        __m256 vc5 = _mm256_loadu_ps(CC + 40);
//
//
//        for(int kk = 0; kk < k; kk += 1)
//        {
//            __m256 va0 = _mm256_set1_ps(AA[0]);
//            __m256 va1 = _mm256_set1_ps(AA[k]);
//
//            __m256 vb0 = _mm256_loadu_ps(BB + 0);
//            __m256 vb1 = _mm256_loadu_ps(BB + 8);
//            __m256 vb2 = _mm256_loadu_ps(BB + 16);
//
//            vc0 = _mm256_fmadd_ps(va0, vb0, vc0);
//            vc1 = _mm256_fmadd_ps(va0, vb1, vc1);
//            vc2 = _mm256_fmadd_ps(va0, vb2, vc2);
//
//            vc3 = _mm256_fmadd_ps(va1, vb0, vc3);
//            vc4 = _mm256_fmadd_ps(va1, vb1, vc4);
//            vc5 = _mm256_fmadd_ps(va1, vb2, vc5);
//
//            AA += 1;
//            BB += 24;
//        }
//
//        AA += k;
//
//        _mm256_storeu_ps(CC + 0, vc0);
//        _mm256_storeu_ps(CC + 8, vc1);
//        _mm256_storeu_ps(CC + 16, vc2);
//
//        _mm256_storeu_ps(CC + 24, vc3);
//        _mm256_storeu_ps(CC + 32, vc4);
//        _mm256_storeu_ps(CC + 40, vc5);
//
//        CC += 48;
//    }
//}

//64.94G
//void sgemm_naive(float *A, float *B, float *C, int m, int k)
//{
//    float *AA = A;
//    float *CC = C;
//
//    for(int i = 0; i < m; i += 2)
//    {
//        float *BB = B;
//
//        __m256 vc0 = _mm256_loadu_ps(CC + 0);
//        __m256 vc1 = _mm256_loadu_ps(CC + 8);
//        __m256 vc2 = _mm256_loadu_ps(CC + 16);
//
//        __m256 vc3 = _mm256_loadu_ps(CC + 24);
//        __m256 vc4 = _mm256_loadu_ps(CC + 32);
//        __m256 vc5 = _mm256_loadu_ps(CC + 40);
//
//
//        for(int kk = 0; kk < k; kk += 2)
//        {
//            __m256 va00 = _mm256_set1_ps(AA[0]);
//            __m256 va01 = _mm256_set1_ps(AA[1]);
//            __m256 va10 = _mm256_set1_ps(AA[k]);
//            __m256 va11 = _mm256_set1_ps(AA[k + 1]);
//
//            __m256 vb0 = _mm256_loadu_ps(BB + 0);
//            __m256 vb1 = _mm256_loadu_ps(BB + 8);
//            __m256 vb2 = _mm256_loadu_ps(BB + 16);
//
//            __m256 vb3 = _mm256_loadu_ps(BB + 24);
//            __m256 vb4 = _mm256_loadu_ps(BB + 32);
//            __m256 vb5 = _mm256_loadu_ps(BB + 40);
//
//            vc0 = _mm256_fmadd_ps(va00, vb0, vc0);
//            vc1 = _mm256_fmadd_ps(va00, vb1, vc1);
//            vc2 = _mm256_fmadd_ps(va00, vb2, vc2);
//
//            vc3 = _mm256_fmadd_ps(va10, vb0, vc3);
//            vc4 = _mm256_fmadd_ps(va10, vb1, vc4);
//            vc5 = _mm256_fmadd_ps(va10, vb2, vc5);
//
//            vc0 = _mm256_fmadd_ps(va01, vb3, vc0);
//            vc1 = _mm256_fmadd_ps(va01, vb4, vc1);
//            vc2 = _mm256_fmadd_ps(va01, vb5, vc2);
//
//            vc3 = _mm256_fmadd_ps(va11, vb3, vc3);
//            vc4 = _mm256_fmadd_ps(va11, vb4, vc4);
//            vc5 = _mm256_fmadd_ps(va11, vb5, vc5);
//
//            AA += 2;
//            BB += 48;
//        }
//
//        AA += k;
//
//        _mm256_storeu_ps(CC + 0, vc0);
//        _mm256_storeu_ps(CC + 8, vc1);
//        _mm256_storeu_ps(CC + 16, vc2);
//
//        _mm256_storeu_ps(CC + 24, vc3);
//        _mm256_storeu_ps(CC + 32, vc4);
//        _mm256_storeu_ps(CC + 40, vc5);
//
//        CC += 48;
//    }
//}

//103G flops
//void sgemm_naive(float *A, float *B, float *C, int m, int k)
//{
//    float *AA = A;
//    float *CC = C;
//
//    for(int i = 0; i < m; i += 4)
//    {
//        float *BB = B;
//
//        __m256 vc0 = _mm256_loadu_ps(CC + 0);
//        __m256 vc1 = _mm256_loadu_ps(CC + 8);
//        __m256 vc2 = _mm256_loadu_ps(CC + 16);
//
//        __m256 vc3 = _mm256_loadu_ps(CC + 24);
//        __m256 vc4 = _mm256_loadu_ps(CC + 32);
//        __m256 vc5 = _mm256_loadu_ps(CC + 40);
//
//        __m256 vc6 = _mm256_loadu_ps(CC + 48);
//        __m256 vc7 = _mm256_loadu_ps(CC + 56);
//        __m256 vc8 = _mm256_loadu_ps(CC + 64);
//
//        __m256 vc9 = _mm256_loadu_ps(CC + 72);
//        __m256 vc10 = _mm256_loadu_ps(CC + 80);
//        __m256 vc11 = _mm256_loadu_ps(CC + 88);
//
//
//        for(int kk = 0; kk < k; kk += 1)
//        {
//            __m256 va0 = _mm256_set1_ps(AA[0]);
//            //__m256 va1 = _mm256_set1_ps(AA[k]);
//
//            //__m256 va2 = _mm256_set1_ps(AA[2 * k]);
//            //__m256 va3 = _mm256_set1_ps(AA[3 * k]);
//
//            //__m256 vb0 = _mm256_loadu_ps(B + kk * 24 + 0);
//            //__m256 vb1 = _mm256_loadu_ps(B + kk * 24 + 8);
//            //__m256 vb2 = _mm256_loadu_ps(B + kk * 24 + 16);
//            __m256 vb0 = _mm256_loadu_ps(BB + 0);
//            __m256 vb1 = _mm256_loadu_ps(BB + 8);
//            __m256 vb2 = _mm256_loadu_ps(BB + 16);
//
//            vc0 = _mm256_fmadd_ps(va0, vb0, vc0);
//            vc1 = _mm256_fmadd_ps(va0, vb1, vc1);
//            vc2 = _mm256_fmadd_ps(va0, vb2, vc2);
//
//            va0 = _mm256_set1_ps(AA[k]);
//
//            vc3 = _mm256_fmadd_ps(va0, vb0, vc3);
//            vc4 = _mm256_fmadd_ps(va0, vb1, vc4);
//            vc5 = _mm256_fmadd_ps(va0, vb2, vc5);
//
//            //__asm__ __volatile__("lfence":::);
//
//            va0 = _mm256_set1_ps(AA[2 * k]);
//
//            vc6 = _mm256_fmadd_ps(va0, vb0, vc6);
//            vc7 = _mm256_fmadd_ps(va0, vb1, vc7);
//            vc8 = _mm256_fmadd_ps(va0, vb2, vc8);
//
//            va0 = _mm256_set1_ps(AA[3 * k]);
//
//            vc9 = _mm256_fmadd_ps(va0, vb0, vc9);
//            vc10 = _mm256_fmadd_ps(va0, vb1, vc10);
//            vc11 = _mm256_fmadd_ps(va0, vb2, vc11);
//
//            AA += 1;
//            BB += 24;
//        }
//
//        AA += 3 * k;
//
//        _mm256_storeu_ps(CC + 0, vc0);
//        _mm256_storeu_ps(CC + 8, vc1);
//        _mm256_storeu_ps(CC + 16, vc2);
//
//        _mm256_storeu_ps(CC + 24, vc3);
//        _mm256_storeu_ps(CC + 32, vc4);
//        _mm256_storeu_ps(CC + 40, vc5);
//
//        _mm256_storeu_ps(CC + 48, vc6);
//        _mm256_storeu_ps(CC + 56, vc7);
//        _mm256_storeu_ps(CC + 64, vc8);
//
//        _mm256_storeu_ps(CC + 72, vc9);
//        _mm256_storeu_ps(CC + 80, vc10);
//        _mm256_storeu_ps(CC + 88, vc11);
//
//        CC += 96;
//    }
//}

static void thread_bind(int cpu)
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu, &cpu_set);

    if(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) != 0)
    {
        std::cout << "Error: cpu[" << cpu << "] bind failed" << std::endl;
        exit(0);
    }
}

int main(int argc, char **argv)
{
    thread_bind(0);

    if(argc != 3)
    {
        std::cout << "Usage: ./main m k" << std::endl;
        return 1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);

    float *A = new float[m * k];
    float *B = new float[k * 24];
    float *C = new float[m * 24];

    std::default_random_engine dre(13);
    std::uniform_real_distribution<float> urd(-10.0, 10.0);

    for(int i = 0; i < m * k; ++i)
        A[i] = urd(dre);
    for(int i = 0; i < k * 24; ++i)
        B[i] = urd(dre);

    memset(C, 0, sizeof(int) * m * 24);

    constexpr int loop_number = 102400;

    //------------------------------------------------------
    //warm up
    for(int i = 0; i < loop_number; ++i)
        sgemm_naive(A, B, C, m, k);

    timeval s, e;
    gettimeofday(&s, nullptr);
    for(int i = 0; i < loop_number; ++i)
        sgemm_naive(A, B, C, m, k);
    gettimeofday(&e, nullptr);

    double time = (e.tv_sec - s.tv_sec + (0.0 + e.tv_usec - s.tv_usec) / 1000000) / loop_number;
    std::cout << "time is " << time << std::endl;

    long comp = 2l * m * k * 24;
    double f = static_cast<double>(comp) / time * 1e-9; 
    std::cout << "perf = " << f << " GFLOPS" << std::endl;
    //------------------------------------------------------
    memset(C, 0, sizeof(int) * m * 24);
    sgemm_naive(A, B, C, m, k);

    double res = 0.0;
    for(int i = 0; i < m * 24; ++i)
        res += C[i];
    std::cout << "the sum of matrix C is " << res << std::endl;
    
    return 0;
}
