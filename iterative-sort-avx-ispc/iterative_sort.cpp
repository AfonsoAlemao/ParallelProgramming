/********************************************//**
 * Calculate root by newton method
 * Compare sequential version and
 * Parallel version of ISPC & AVX
 *
 * Written by:
 * Dongyang Yao (dongyang.yao@rutgers.edu)
 ***********************************************/

#include <iostream>
#include <chrono>
#include <cmath>
#include <random>  
#include <thread>
#include <pthread.h>
#include "iterative_sort_avx.h"
#include "iterative_sort_ispc.h"

using namespace std::chrono;
using namespace ispc;

/* Random sequence structure */
typedef struct {
  int start;
  int end;
  float* randoms;
  float* results;
} t_args;

/* Get root in random sequence by ISPC */
void* t_iterative_sort_ispc(void* void_args) {
  t_args* args = (t_args*)void_args;
  iterative_sort_ispc(args->randoms, args->results, args->start, args->end);
  return NULL;
}

/* Sequential Newton method */
void get_root_by_newton(float n, float* r) {
  float k = 1.0;
  float tmp = 0.0;

  // Convergence requirements
  while (std::abs(k * k - n) > 0.0001 || std::abs(k - tmp) > 0.0001) {
    tmp = k;
    k = (k + n / k) / 2;
  }
  *r = k;
}

int main() {

  // Length of the sequence
  const int N = 20000000;
  
  float* randoms = new float[N];
  float* results_sequence = new float[N];
  float* results_avx = new float[N];
  float* results_ispc = new float[N];

  // Prepare random numbers and the truth
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 3.0);
  
  for (int i = 0; i < N; i++) {
    randoms[i] = distribution(generator);
  }

  std::cout << "-----" << "elapsed time" << "-----" << std::endl;

  high_resolution_clock::time_point start = high_resolution_clock::now();

  // Run sequential version 
  for (int i = 0; i < N; i++) {
    get_root_by_newton(randoms[i], &results_sequence[i]);
  }

  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double> >(end - start);
  std::cout << "sequence: " <<  time_span.count() * 1000 << " ms" << std::endl;
  
  // Run AVX version in a single core
  iterative_sort_avx(randoms, results_avx, N);

  start = high_resolution_clock::now();

  // Run ISPC version in a single core 
  iterative_sort_ispc(randoms, results_ispc, 0, N);

  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(end - start);
  std::cout << "ispc(avx-x8, single-core): " << time_span.count() * 1000 << " ms" << std::endl;
  
  int nthreads = std::thread::hardware_concurrency();
  //std::cout << "max number of threads: " << nthreads << std::endl;
  
  nthreads = 4;

  start = high_resolution_clock::now();

  pthread_t threads[nthreads];
  t_args args[nthreads];

  // Run SIMD with multi-core 
  for (int t = 1; t < nthreads; t++) {
    args[t].randoms = randoms;
    args[t].results = results_ispc;
    args[t].start = t * N / nthreads;
    args[t].end = args[t].start + N / nthreads;

    // Create a thread for part of sequence
    int rc = pthread_create(&threads[t], NULL, t_iterative_sort_ispc, (void*)&args[t]);
    if (rc) std::cout << "thread " << t << "create error" << std::endl;
  }

  // Work in main thread
  iterative_sort_ispc(randoms, results_ispc, 0, N / nthreads);

  // Join the threads
  for (int t = 1; t < nthreads; t++) {
    pthread_join(threads[t], NULL);
  }

  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(end - start);
  std::cout << "ispc(avx-x8, multi-core): " << time_span.count() * 1000 << " ms" << std::endl;  

  std::cout << "-----" << "validation samples" << "-----" << std::endl;

  for (int i = 0; i < 8; i++) {
    std::cout << "random: " << randoms[i] << " truth: " << sqrt(randoms[i]) << " sequence: " << results_sequence[i] << " ispc: " << results_ispc[i] << " avx: " << results_avx[i] << std::endl;
  }

  return 0;
}
