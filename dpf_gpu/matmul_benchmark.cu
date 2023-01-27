// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

// Benchmark and test 128-bit matmul for DPF

#include "utils.h"
#include "matmul/matmul.cu"

#ifndef REPS
#define REPS 10
#endif

void print_params() {
  printf("------------------------------------------------------\n");  
  printf("matmul_benchmark.cu:\n");
  printf("------------------------------------------------------\n");  
  printf("- Entries in table (K): %d\n", KK);
  printf("- Batch size (N): %d\n", NN);
  printf("- Entry size (M): %d\n", MM);
  printf("------------------------------------------------------\n");  
}

void alloc_test_matrix(uint128_t_gpu **A_gpu,
		       uint128_t_gpu **A_cpu,
		       int M, int N) {
  *A_cpu = new uint128_t_gpu[M*N];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      (*A_cpu)[i*N+j] = uint128_gpu_from((uint128_t)i*N+j);
    }
  }

  cudaMalloc(A_gpu, sizeof(uint128_t_gpu)*M*N);
  cudaMemcpy(*A_gpu, *A_cpu, sizeof(uint128_t_gpu)*M*N, cudaMemcpyHostToDevice);
}

void check_correct(uint128_t_gpu *A,
		   uint128_t_gpu *B,
		   uint128_t_gpu *C,
		   int M, int K, int N) {
  uint128_t_gpu *C_ref = new uint128_t_gpu[M*N];
  memset(C_ref, 0, sizeof(uint128_t_gpu)*M*N);

  // Compute ref solution
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      for (int k = 0; k < N; k++) {
	uint128_t c = uint128_from_gpu(C_ref[i*N+k]);
	uint128_t a = uint128_from_gpu(A[i*K+j]);
	uint128_t b = uint128_from_gpu(B[j+k*K]);
	uint128_t accum = c+a*b;
	C_ref[i*N+k] = uint128_gpu_from(accum);
      }
    }
  }

  // Assert same
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      uint128_t_gpu got = C[i*N+j];
      uint128_t_gpu expected = C_ref[i*N+j];

      assert(got.x == expected.x &&
	got.y == expected.y &&
	got.z == expected.z &&
	got.w == expected.w);      
    }
  }
  
  printf("PASS CHECKS\n");
}

int main(void) {
  print_params();

  // Alloc & Init buffers
  uint128_t_gpu *A_gpu, *B_gpu, *C_gpu;
  uint128_t_gpu *A_cpu, *B_cpu, *C_cpu;  
  
  alloc_test_matrix(&A_gpu, &A_cpu, MM, KK);
  alloc_test_matrix(&B_gpu, &B_cpu, KK, NN);
  alloc_test_matrix(&C_gpu, &C_cpu, MM, NN);

  cudaMemset(C_gpu, 0, sizeof(uint128_t_gpu)*MM*NN);

  // Init
  initialize_matmul(MM, KK, NN);

  // Kernel benchmark
  cudaStream_t s1;
  cudaStreamCreate(&s1);  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Run throughput benchmark
  cudaEventRecord(start);
  for (int i = 0; i < REPS; i++) {
    GEMM128(A_gpu, C_gpu, B_gpu, MM, KK, NN, s1);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Run latency benchmark
  cudaEvent_t start_latency, stop_latency;
  cudaEventCreate(&start_latency);
  cudaEventCreate(&stop_latency);
  cudaEventRecord(start_latency);  

  GEMM128(A_gpu, C_gpu, B_gpu, MM, KK, NN, s1);

  cudaEventRecord(stop_latency);
  cudaEventSynchronize(stop_latency);  
  CUDA_CHECK(cudaGetLastError());

  // Correctness checks
  cudaMemcpy(C_cpu, C_gpu, sizeof(uint128_t_gpu)*MM*NN, cudaMemcpyDeviceToHost);
  //  check_correct(A_cpu, B_cpu, C_cpu, MM, KK, NN);
  
  // Stats
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float throughput_per_query = NN*REPS/ms;

  float ms_latency = 0;
  cudaEventElapsedTime(&ms_latency, start_latency, stop_latency);

  // Final logging output
  printf("{'entries (K)': %d, 'entry_size_ints (M)': %d, 'batch_size (N)': %d,"
	 "'latency_ms' : %f, 'throughput_queries_per_ms' : %f'}\n",
	 KK, MM, NN,
	 ms_latency, throughput_per_query);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);

  delete A_cpu;
  delete B_cpu;  
  delete C_cpu;
}
