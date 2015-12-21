/********************************************//** 
 * Calculate exclusive scan, find repeats, 
 * Output remainings in parallel with GPU
 * Using CUDA language
 *
 * Written by:
 * Dongyang Yao (dongyang.yao@rutgers.edu)
 ***********************************************/

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

using namespace std::chrono;

int N = 1000000;

#define THREADS_PER_BLK 128

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

/* Get random inputs */
void generate_randoms(int* randoms, int length, int max) {
  std::srand(std::time(0));
  for (int i = 0; i < length; i++) {
    randoms[i] = std::rand() % max;
    //std::cout << randoms[i] << std::endl;
  }
}

/* Print out the numbers */
void show_samples(int* numbers, int count) {
  for (int i = 0; i < count; i++) std::cout << numbers[i] << std::endl;
}

/* Get exclusive scan in sequential */
void generate_exclusive_scan_truth(int* check, int* randoms, int length) {
  check[0] = 0;
  for (int i = 1; i < length; i++)
    check[i] = check[i - 1] + randoms[i - 1];
}

/* Get repeats in sequential */
int generate_find_repeats_truth(std::vector<int>* check, int* randoms, int length) {
  int count = 0;
  for (int i = 0; i < length - 1; i++) {
    if (randoms[i] == randoms[i + 1]) {
      count++;
      check->push_back(i);
    }
  }
  return count;
}

/* Get remainings in sequential */
int generate_remove_repeats_truth(std::vector<int>* check, int* randoms, int length) {
  int count = 0;
  for (int i = 0; i < length - 1; i++) {
    if (randoms[i] != randoms[i + 1]) {
      count++;
      check->push_back(randoms[i]);
    }
  }
  count++;
  check->push_back(randoms[length - 1]);
  return count;
}

/* Compare result with truth */
void check_results(int* results, int* check, int length) {
  bool result = true;
  for (int i = 0; i < length; i++) {
    //std::cout << check[i] << " " << results[i] << std::endl;
    if (!(check[i] == results[i])) {
      std::cout << "mis-match at " << i << std::endl;
      result = false;
    }
  }

  if (result) std::cout << "pass successfully" << std::endl;
  else std::cout << "you have error shown above" << std::endl;
}

/* Compare result with truth */
void check_results(int* results, std::vector<int>* check) {
  bool result = true;

  int i = 0;
  for (std::vector<int>::iterator iter = check->begin(); iter != check->end(); iter++) {
    if (!(*iter == results[i])) {
      std::cout << *iter << " " << results[i] << std::endl;
      std::cout << "mis-match at " << i << std::endl;
      result = false;
    }
    i++;
  }

  if (result) std::cout << "pass successfully" << std::endl;
  else std::cout << "you have error shown above" << std::endl;
}

/* Get CUDA info on this computer */
void get_cuda_info() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  std::cout << "number of gpu: " << device_count << std::endl;

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, i);
    std::cout << "name: " << device_props.name << std::endl;
  }
}

/* Get next POW of 2 */
int get_next_pow_2(int n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

/* Get exclusive scan on GPU */
__global__ void exclusive_scan_gpu(int* input, int* output, int n) {
  __shared__ int temp[4 * THREADS_PER_BLK];
  int thid_global = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  int thid = threadIdx.x;

  {  
    int offset = 1;
    //temp[2 * thid] = input[2 * thid_global];
    //temp[2 * thid + 1] = input[2 * thid_global + 1];
    
    int aind = thid;
    int bind = thid + n / 2;
    int bankOffsetA = CONFLICT_FREE_OFFSET(aind);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bind);
    temp[aind + bankOffsetA] = input[thid_global];
    temp[bind + bankOffsetB] = input[thid_global + n / 2];  
     

    for (int d = n >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (thid == 0) { 
      //temp[n - 1] = 0;
      temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);
        int t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }

    __syncthreads();
    //output[2 * thid_global] = temp[2 * thid];
    //output[2 * thid_global + 1] = temp[2 * thid + 1];
    //printf("%d:%d %d:%d\n", 2 * thid_global, output[2 * thid_global], 2 * thid_global + 1, output[2 * thid_global + 1]);
    output[thid_global] = temp[aind + bankOffsetA];
    output[thid_global + n / 2] = temp[bind + bankOffsetB];
  }

}

/* Add partial results with base to get full result on GPU */
__global__ void add_base_gpu(int* device_input, int* device_output, int block_index) {
  int block_last_element = block_index * THREADS_PER_BLK * 2 - 1;
  
  int base = device_input[block_last_element] + device_output[block_last_element];
  
  int thid = block_index * blockDim.x + threadIdx.x;

  device_output[2 * thid] += base;
  device_output[2 * thid + 1] += base;
}

/* Mark repeat on GPU */
__global__ void mark_flags_gpu(int* input, int* flags, int length, bool mark_repeat) {
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  if (thid < length - 1) {
     if (input[thid] == input[thid + 1]) {
       flags[thid] = mark_repeat ? 1 : 0;
     } else {
       flags[thid] = mark_repeat ? 0 : 1;
     }
     //printf("id:%d %d\n", thid, flags[thid]);
  }
}

/* Get repeats on GPU */
__global__ void get_repeat_results(int* input, int* flags_scaned, int length, int* output, bool mark_repeat) {
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  if ((thid < length - 1) && (flags_scaned[thid] < flags_scaned[thid + 1])) {
    //printf("id:%d %d\n", thid, flags_scaned[thid]);
    output[flags_scaned[thid]] = mark_repeat ? thid : input[thid]; 
  }
  if ((thid == length - 1) && (!mark_repeat)) {
    //printf("id:%d %d\n", thid, flags_scaned[thid]);
    output[flags_scaned[thid]] = input[length - 1];
    
  }
}

/* Get exclusive scan on CPU */
void exclusive_scan_sequential(int* randoms, int length, int* output) {
    memmove(output, randoms, length * sizeof(int));

    // Upsweep phase
    for (int twod = 1; twod < length; twod*=2)
    {
        int twod1 = twod*2;
        // Parallel
        for (int i = 0; i < length; i += twod1)
        {
            output[i+twod1-1] += output[i+twod-1];
        }
    }

    output[length-1] = 0;

    // Downsweep phase
    for (int twod = length/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod*2;
        // Parallel
        for (int i = 0; i < length; i += twod1)
        {
            int tmp = output[i+twod-1];
            output[i+twod-1] = output[i+twod1-1];
            output[i+twod1-1] += tmp;
        }
    }
}

/* Get repeats on CPU */
int find_repeats_sequential(int* results, int* randoms, int length) {
  int count = 0;

  int* flags = new int[length];

  for (int i = 0; i < length - 1; i++) {
    if (randoms[i] == randoms[i + 1]) {
      count++;
      flags[i] = 1;
    } else {
      flags[i] = 0;
    }
  }

  //for (int i = 0; i < length; i++) std::cout << flags[i] << std::endl;
  
  int length_rounded = get_next_pow_2(N);
  int* flags_scaned = new int[length_rounded];  

  exclusive_scan_sequential(flags, length_rounded, flags_scaned);

  //for (int i = 0; i < length; i++) std::cout << flags_scaned[i] << std::endl;

  for (int i = 0; i < length - 1; i++) {
    if (flags_scaned[i] < flags_scaned[i + 1]) {
      results[flags_scaned[i]] = i;
    }
  }

  delete[] flags;
  delete[] flags_scaned;

  return count;
}

int main(int argc, char** argv) {

  int* randoms;
  bool use_external = false;
  
  if (argc == 2) {
     int in = atoi(argv[1]);
     if (in != 0) N = in;
     else {
       std::string line;
       std::ifstream file (argv[1]);
       if (file.is_open()) {
         use_external = true;

         std::cout << "loading external data..." << std::endl;
	 getline(file, line);
	 N = std::stoi(line);
	 randoms = new int[N];

	 int i = 0;
	 while (getline(file, line)) {
	   randoms[i++] = std::stoi(line);   
	 }
	 
	 file.close();
       } else {
         std::cout << "cannot find the file!" << std::endl;
       }      
     }
  }

  std::cout << "**********" << std::endl;
  std::cout << "DEBUG INFO" << std::endl;
  std::cout << "**********" << std::endl;    

  std::cout << "number of threads per block: " << THREADS_PER_BLK << std::endl;
  
  const int MAX = 100;
  //const int NUM_SAMPLE = 10;

  if (!use_external) {  
    randoms = new int[N];
  
    std::cout << "generating random numbers..." << std::endl;
    std::cout << "max: " << MAX << std::endl;
  
    generate_randoms(randoms, N, MAX);
  }

  std::cout << "count: " << N << std::endl;
  
  //for (int i = 0; i < N; i++) std::cout << i << ":" << randoms[i] << std::endl;

  /*
  std::cout << "showing random numbers..." << std::endl;
  std::cout << "count: " << NUM_SAMPLE << std::endl;

  show_samples(randoms, NUM_SAMPLE);
  */

  int* exclusive_scan_check = new int[N];

  std::cout << "generating exclusive scan ground truth..." << std::endl;

  generate_exclusive_scan_truth(exclusive_scan_check, randoms, N);

  std::cout << "computing exclusive scan in cpu..." << std::endl;

  int length = get_next_pow_2(N);
  int* output_sequential = new int[length];
  
  high_resolution_clock::time_point start = high_resolution_clock::now();  
  
  exclusive_scan_sequential(randoms, length, output_sequential);
  
  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double> >(end - start);
  std::cout << "elapsed time: " <<  time_span.count() * 1000 << " ms" << std::endl;  
  
  check_results(output_sequential, exclusive_scan_check, N);

  /*
  std::cout << "showing ground truth..." << std::endl;
  std::cout << "count: " << NUM_SAMPLE << std::endl;
  
  show_samples(exclusive_scan_check, NUM_SAMPLE);
  */

  std::cout << "checking gpu availability..." << std::endl;
  get_cuda_info();
  
  std::cout << "rounding up to the next highest power of 2..." << std::endl;
  std:: cout << "rounded length: " << length << std::endl;
  
  int* exclusive_scan_gpu_results = new int[N];
  int* device_input;
  int* device_output;

  int* find_repeat_gpu_results = new int[N];
  int* flags;
  int* flags_scaned;
  int* find_repeat_output;

  int* remove_repeat_gpu_results = new int[N];
  int* flags_remain;
  int* flags_remain_scaned;
  int* remove_repeat_output;
  
  std::cout << "allocateing memory on gpu for input and output..." << std::endl;
  cudaMalloc((void **) &device_input, sizeof(int) * length);
  cudaMalloc((void **) &device_output, sizeof(int) * length);

  cudaMalloc((void **) &flags, sizeof(int) * length);
  cudaMalloc((void **) &flags_scaned, sizeof(int) * length);
  cudaMalloc((void **) &find_repeat_output, sizeof(int) * length);	

  cudaMalloc((void **) &flags_remain, sizeof(int) * length);
  cudaMalloc((void **) &flags_remain_scaned, sizeof(int) * length);
  cudaMalloc((void **) &remove_repeat_output, sizeof(int) * length);

  std::cout << "copying the random numbers from cpu to gpu..." << std::endl;
  cudaMemcpy(device_input, randoms, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(device_output, randoms, sizeof(int) * N, cudaMemcpyHostToDevice);

  std::cout << "computing exclusive scan on gpu..." << std::endl;
  
  int num_block = length / (THREADS_PER_BLK * 2);
  if (num_block == 0) num_block = 1;
  
  std::cout << "number of block: " << num_block << std::endl;

  start = high_resolution_clock::now();  

  exclusive_scan_gpu<<<num_block, THREADS_PER_BLK>>>(device_input, device_output, length / num_block);
  cudaThreadSynchronize();

  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(end - start);
  std::cout << "elapsed time: " << time_span.count() * 1000 << " ms" << std::endl;

  //cudaMemcpy(exclusive_scan_gpu_results, device_output, sizeof(int) * N, cudaMemcpyDeviceToHost);

  //for (int i = 0; i < N; i++) std::cout << i << ":" << exclusive_scan_gpu_results[i] << ":" << exclusive_scan_check[i] << std::endl;
  
  //std::cout << "multi-block" << std::endl;

  for (int i = 1; i < num_block; i++)
    add_base_gpu<<<1, THREADS_PER_BLK>>>(device_input, device_output, i);

  cudaMemcpy(exclusive_scan_gpu_results, device_output, sizeof(int) * N, cudaMemcpyDeviceToHost);
  
  //for (int i = 0; i < N; i++) std::cout << i << ":" << exclusive_scan_gpu_results[i] << ":" << exclusive_scan_check[i] << std::endl;

  check_results(exclusive_scan_gpu_results, exclusive_scan_check, N);

  std::cout << "computing exclusive scan using THRUST library..." << std::endl;

  int* scan_thrust_results = new int[N];

  thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
  thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
  
  cudaMemcpy(d_input.get(), randoms, N * sizeof(int), cudaMemcpyHostToDevice);

  start = high_resolution_clock::now();

  thrust::exclusive_scan(d_input, d_input + length, d_output);

  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(end - start);
  std::cout << "elapsed time: " << time_span.count() * 1000 << " ms" << std::endl;

  cudaMemcpy(scan_thrust_results, d_output.get(), N * sizeof(int), cudaMemcpyDeviceToHost);

  check_results(scan_thrust_results, exclusive_scan_check, N);

  std::cout << "generating find repeats ground truth..." << std::endl;
  
  std::vector<int>* find_repeats_check = new std::vector<int>(); 
  
  int repeats_count = generate_find_repeats_truth(find_repeats_check, randoms, N);

  std::cout << "computing find repeats on cpu..." << std::endl;

  int* find_repeats_sequential_results = new int[N];

  start = high_resolution_clock::now();

  find_repeats_sequential(find_repeats_sequential_results, randoms, N);

  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(end - start);
  std::cout << "elapsed time: " << time_span.count() * 1000 << " ms" << std::endl;

  check_results(find_repeats_sequential_results, find_repeats_check);

  std::cout << "number of repeats: " << repeats_count << std::endl;
  
  std::cout << "computing find repeats on gpu..." << std::endl;

  int num_block_repeat = length / THREADS_PER_BLK;
  if (num_block_repeat == 0) num_block_repeat = 1;

  start = high_resolution_clock::now();

  mark_flags_gpu<<<num_block_repeat, THREADS_PER_BLK>>>(device_input, flags, length, true);

  exclusive_scan_gpu<<<num_block, THREADS_PER_BLK>>>(flags, flags_scaned, length / num_block);
  cudaThreadSynchronize();
  
  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(end - start);
  std::cout << "elapsed time: " << time_span.count() * 1000 << " ms" << std::endl;

  for (int i = 1; i < num_block; i++)
    add_base_gpu<<<1, THREADS_PER_BLK>>>(flags, flags_scaned, i);

  get_repeat_results<<<num_block_repeat, THREADS_PER_BLK>>>(device_input, flags_scaned, N, find_repeat_output, true);
  cudaMemcpy(find_repeat_gpu_results, find_repeat_output, sizeof(int) * N, cudaMemcpyDeviceToHost);

  check_results(find_repeat_gpu_results, find_repeats_check);

  std::cout << "generating remove repeats ground truth..." << std::endl;
  
  std::vector<int>* remove_repeats_check = new std::vector<int>();

  start = high_resolution_clock::now();

  int remain_count = generate_remove_repeats_truth(remove_repeats_check, randoms, N);

  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(end - start);
  std::cout << "elapsed time: " << time_span.count() * 1000 << " ms" << std::endl;

  std::cout << "number of remains: " << remain_count << std::endl;

  std::cout << "computing remove repeats on gpu..." << std::endl;

  start = high_resolution_clock::now();

  mark_flags_gpu<<<num_block_repeat, THREADS_PER_BLK>>>(device_input, flags_remain, length, false);

  exclusive_scan_gpu<<<num_block, THREADS_PER_BLK>>>(flags_remain, flags_remain_scaned, length / num_block);
  cudaThreadSynchronize();

  end = high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(end - start);
  std::cout << "elapsed time: " << time_span.count() * 1000 << " ms" << std::endl;

  for (int i = 1; i < num_block; i++)
    add_base_gpu<<<1, THREADS_PER_BLK>>>(flags_remain, flags_remain_scaned, i);

  get_repeat_results<<<num_block_repeat, THREADS_PER_BLK>>>(device_input, flags_remain_scaned, N, remove_repeat_output, false);

  cudaMemcpy(remove_repeat_gpu_results, remove_repeat_output, sizeof(int) * N, cudaMemcpyDeviceToHost);

  check_results(remove_repeat_gpu_results, remove_repeats_check);

  std::cout << "************" << std::endl;
  std::cout << "REQUIREMENTS" << std::endl;
  std::cout << "************" << std::endl;

  std::cout << "array A (exclusive scan)" << std::endl;
  std::cout << "size: " << N << std::endl;
  std::cout << "last element: " << exclusive_scan_gpu_results[N - 1] << std::endl;
  
  std::cout << "array B (repeating indices)" << std::endl;
  std::cout << "size: " << repeats_count << std::endl;
  if (repeats_count != 0)
    std::cout << "last element:" << find_repeat_gpu_results[repeats_count - 1] << std::endl;

  std::cout << "array C (remaining entries)" << std::endl;
  std::cout << "size: " << remain_count << std::endl;
  if (remain_count != 0)
    std::cout << "last_element: " << remove_repeat_gpu_results[remain_count - 1] << std::endl;

  std::cout << "output exclusive scan gpu results file..." << std::endl;

  std::ofstream myfile1 ("A_exclusive_scan.txt");
  if (myfile1.is_open())
  {
    myfile1 << "size: " << N << "\n";
    for (int i = 0; i < N; i++) {
      myfile1 << exclusive_scan_gpu_results[i] << "\n";
    }
    myfile1.close();
  }

  std::cout << "output repeat indices gpu results file..." << std::endl;

  std::ofstream myfile2 ("B_repeat_indices.txt");
  if (myfile2.is_open())
  {
    myfile2 << "size: " << repeats_count << "\n";
    for (int i = 0; i < repeats_count; i++) {
      myfile2 << find_repeat_gpu_results[i] << "\n";
    }
    myfile2.close();
  }

  std::cout << "output remaining entries gpu results file..." << std::endl;

  std::ofstream myfile3 ("C_remaining_entries.txt");
  if (myfile3.is_open())
  {
    myfile3 << "size: " << remain_count << "\n";
    for (int i = 0; i < remain_count; i++) {
      myfile3 << remove_repeat_gpu_results[i] << "\n";
    }
    myfile3.close();
  }
  
  delete[] randoms;
  delete[] exclusive_scan_check;
  delete[] output_sequential;
  delete[] exclusive_scan_gpu_results;
  delete[] find_repeat_gpu_results;
  delete[] remove_repeat_gpu_results;
  delete[] scan_thrust_results;
  
  return 0;
  
}
