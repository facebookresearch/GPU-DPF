// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


/*
  Simulates the performance of 2 server PIR 
  - Does not measure communication latency
  - Performs DPF on the CPU _only_
  - Measures runtime for a single server to expand DPF vector + CUDA Matvecmul
    (Note: since the other server does the same thing, runtime will be same)
    
  Current workflow
  - Initialize embedding tables (w/ random numbers) assumed to be a secret
    share of the true table. Table size is parameterized. Put on GPU
  - Initialize DPF key using google's DPF CPU library
  - Start timer
  - Expand DPF on vector
  - Transfer vector GPU memory, perform call to Cublas GEMV
  - Stop timer
 */

#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cassert>
#include "dpf_helpers.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <ctime>
#include <sys/time.h>
#include <chrono>
#include <omp.h>

void *AllocEmbeddingTable(int n, int l) {
  // n - number of table entries
  // l - number of elements per table entry (vector dimension)
  // Assum each entry is uint32_t

  // Allocate on CPU, populate with some values, cpy to GPU
  uint32_t *cpu_table = (uint32_t *)malloc(n*l*sizeof(uint32_t));
  if (cpu_table == NULL) assert(0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < l; j++) {
      cpu_table[i*l+j] = i;
    }
  }

  void *gpu_table;
  cudaMalloc((void**)&gpu_table, sizeof(uint32_t)*n*l);
  cudaMemcpy(gpu_table, cpu_table, sizeof(uint32_t)*n*l, cudaMemcpyHostToDevice);

  free(cpu_table);

  return gpu_table;
}


int main(int argc, char *argv[]) {
  int N_EMBEDDING_ENTRIES = atoi(argv[1]);
  int EMBEDDING_LENGTH = atoi(argv[2]);
  int USE_DPF = atoi(argv[3]);
  int BATCH_SIZE = atoi(argv[4]);
  int REPS = atoi(argv[5]);
  int USE_GEMM = atoi(argv[6]);
  int DPF_THREADS = atoi(argv[7]);

  printf("Params: n_embedding_entries=%d, embedding_length=%d, use_dpf=%d, batch=%d reps=%d\n", N_EMBEDDING_ENTRIES, EMBEDDING_LENGTH, USE_DPF, BATCH_SIZE, REPS);
  
  std::cout << "Init CUDA" << std::endl;
  cudaStream_t cudaStream;
  cublasHandle_t handle;
  cublasCreate(&handle);
  if (CUBLAS_STATUS_SUCCESS != cublasSetStream(handle, cudaStream))
    {
      printf("Cublas set stream failed\n");
      exit(-1);
    }
  
  std::cout << "Malloc embedding table onto GPU" << std::endl;
  void *embedding_table = AllocEmbeddingTable(N_EMBEDDING_ENTRIES,
					      EMBEDDING_LENGTH);

  std::cout << "Initializing DPF" << std::endl;
  void *dpf = DPFInitialize(32, 32);
  void *k1, *k2;
  DPFGetKey(dpf, 42, 21, &k1, &k2);

  std::cout << "Mallocing indicator vector" << std::endl;
  uint32_t *indicator_vector;
  cudaMallocManaged(&indicator_vector, sizeof(uint32_t)*N_EMBEDDING_ENTRIES*BATCH_SIZE);

  std::cout << "Mallocing output vector" << std::endl;
  uint32_t *o;
  cudaMallocManaged(&o, sizeof(uint32_t)*EMBEDDING_LENGTH);
  cudaMemset(o, 0, sizeof(uint32_t)*EMBEDDING_LENGTH);

  ////////////////////////////////////////////////////////////////////////
  // Past this point the benchmark times
  std::cout << "Benchmarking..." << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float gemm_ms_time_cumulative = 0;
  float dpf_ms_time_cumulative = 0;

  omp_set_num_threads(DPF_THREADS);

  for (int i = 0; i < REPS; i++) {
    std::cout << i << std::endl;

    // DPF initialize batch of secret shared indicator vectors
    if (USE_DPF) {
      auto t0 = std::chrono::high_resolution_clock::now();

      #pragma omp parallel for
      for (int j = 0; j < BATCH_SIZE; j++) {
	DPFExpand(dpf, k1, N_EMBEDDING_ENTRIES, indicator_vector + N_EMBEDDING_ENTRIES*j);
      }
      
      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration< double > fs = t1 - t0;
      std::chrono::milliseconds d = std::chrono::duration_cast< std::chrono::milliseconds >(fs);
      
      dpf_ms_time_cumulative += d.count();
    }

    if (USE_GEMM) {
      // Batch matmul
      int alpha = 1;
      int beta = 0;
      cudaEventRecord(start);
      auto status = cublasGemmEx(handle,
				 CUBLAS_OP_T, // Embed Table is row major
				 CUBLAS_OP_N, // indicator vec is col major
			     
				 EMBEDDING_LENGTH,    //m, where mxkxn, n=1
				 BATCH_SIZE,          //n (batch)
				 N_EMBEDDING_ENTRIES, //k
			     
				 &alpha,              //alpha
				 embedding_table,     //A
				 CUDA_R_32F,          //dtype of A
				 N_EMBEDDING_ENTRIES, //lda
				 indicator_vector,    //B
				 CUDA_R_32F,          //dtype of B
				 N_EMBEDDING_ENTRIES, //ldb
				 &beta,               //beta
				 o,                   //C
				 CUDA_R_32F,          //dtype of C. idea is values wrap around and hardware overflow implicitly does mod math
				 EMBEDDING_LENGTH,    //ldc
				 CUDA_R_32F,
				 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      cudaEventRecord(stop);
      if (status != CUBLAS_STATUS_SUCCESS) {
	std::cout << "GemmEx Failed " << status << " " << CUBLAS_STATUS_NOT_SUPPORTED << std::endl;
      }
    
      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      gemm_ms_time_cumulative += milliseconds;
    }
  }

  float total_time_ms = gemm_ms_time_cumulative + dpf_ms_time_cumulative;
  float total_throughput = (REPS*BATCH_SIZE)/total_time_ms;
  float dpf_throughput = (REPS*BATCH_SIZE)/dpf_ms_time_cumulative;
  float gemm_throughput = (REPS*BATCH_SIZE)/gemm_ms_time_cumulative;

  printf("{'total_time_ms' : %f, 'total_throughput': %f,"
	 "'dpf_throughput': %f, 'gemm_throughput': %f,"
	 "'gemm_ms_time_cumulative':%f,"
	 "'dpf_ms_time_cumulative': %f,"
	 "'N_EMBEDDING_ENTRIES': %d,"
	 "'EMBEDDING_LENGTH': %d,"
	 "'USE_DPF': %d,"
	 "'BATCH_SIZE': %d,"
	 "'REPS': %d,"
	 "'USE_GEMM': %d,"
	 "'DPF_THREADS': %d}",	 
	 total_time_ms, total_throughput, dpf_throughput,
	 gemm_throughput, gemm_ms_time_cumulative, dpf_ms_time_cumulative,
	 N_EMBEDDING_ENTRIES, EMBEDDING_LENGTH, USE_DPF, BATCH_SIZE, REPS, USE_GEMM, DPF_THREADS);
}
