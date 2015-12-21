/********************************************//** 
 * AVX version of Newton method 
 * To calculate the root
 *
 * Written by:
 * Dongyang Yao (dongyang.yao@rutgers.edu)
 ***********************************************/

#include <immintrin.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;

/* Determine if all numbers in a vector are convergent */
bool is_convergent(__m256 k_v, __m256 tmp_v) {
  float error1 = 0.0001;
  float error2 = -0.0001;
  __m256 cmp1 = _mm256_cmp_ps(_mm256_sub_ps(k_v, tmp_v), _mm256_broadcast_ss(&error1), _CMP_GT_OQ);
  __m256 cmp2 = _mm256_cmp_ps(_mm256_sub_ps(k_v, tmp_v), _mm256_broadcast_ss(&error2), _CMP_LT_OQ);
  return !((_mm256_movemask_ps(cmp1)&255) || (_mm256_movemask_ps(cmp2)&255));
}

/* Calculate root in vectors */
void iterative_sort_avx(float *randoms_orgin, float *results_orgin, int N) {
  
  double time = 0.0;
  const int M = 2000000;
  
  for (int j = 0; j < N / M; j++) {
    float randoms[M];
    for (int i = 0; i < M; i++) randoms[i] = randoms_orgin[i + j * M];

    for (int i = 0; i < M; i += 8) {
      __m256 numbers = _mm256_load_ps(&randoms[i]);
      float k = 1.0;
      __m256 k_v = _mm256_broadcast_ss(&k);
      float c = 2.0;
      float tmp = 0.0;
      __m256 tmp_v = _mm256_broadcast_ss(&tmp);

      high_resolution_clock::time_point start = high_resolution_clock::now();

      while (!is_convergent(k_v, tmp_v)) {
	tmp_v = k_v;
	k_v = _mm256_div_ps(_mm256_add_ps(k_v, _mm256_div_ps(numbers, k_v)), _mm256_broadcast_ss(&c));
      }

      high_resolution_clock::time_point end = high_resolution_clock::now();
      duration<double> time_span = duration_cast<duration<double> >(end - start);
      time += time_span.count() * 1000;
      
      float results[8];
      _mm256_store_ps(results, k_v);
      for (int k = 0; k < 8; k++) results_orgin[k + i + j * M] = results[k];
    }

  }

  std::cout << "avx(code-implemented): " <<  time << " ms" << std::endl;

}

