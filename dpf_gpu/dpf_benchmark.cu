// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#include "prf/prf.cu"
#include "matmul/matmul.cu"
#include "utils.h"

// By default expand dpf in full
#ifndef FUSES_MATMUL
#define FUSES_MATMUL 0
#endif

/****************************
 *     DPF Strategies       *
 ****************************/

#define DPF_NAIVE 0
#define DPF_BREADTH_FIRST 1
#define DPF_HYBRID 2
#define DPF_COOP 3

#ifndef DPF_STRATEGY 
#define DPF_STRATEGY DPF_HYBRID
#endif

// Include different DPF methods depending on set strategy.
// We do this as some DPF methods may pre-initialize or use
// global memory.
#if(DPF_STRATEGY == DPF_NAIVE)
#include "dpf/dpf_naive.cu"

#undef FUSES_MATMUL
#define FUSES_MATMUL 0

#endif
#if(DPF_STRATEGY == DPF_BREADTH_FIRST)
#include "dpf/dpf_breadth_first.cu"

#undef FUSES_MATMUL
#define FUSES_MATMUL 0

#endif
#if(DPF_STRATEGY == DPF_HYBRID)
#include "dpf/dpf_hybrid.cu"
#endif
#if(DPF_STRATEGY == DPF_COOP)
#include "dpf/dpf_coop.cu"
#endif


// Flag if we perform matmul at end
#ifndef PERFORM_MATMUL
#define PERFORM_MATMUL 0
#endif

#if(FUSES_MATMUL == 1)
#undef PERFORM_MATMUL
#define PERFORM_MATMUL 1
#endif

/*********************************
 * Compile time table constants  *
 *********************************/
// MM - entry size (note this does not affect DPF expansion)
// KK - number of entries
// NN - batch size
#ifndef MM
#define MM 1
#endif

#ifndef KK
#define KK 1048576
#endif

#ifndef NN
#define NN 64
#endif

// Benchmarking constants
#ifndef REPS
#define REPS 10
#endif

// Table on GPU
uint128_t_gpu *TABLE;

void initialize_table(uint128_t_gpu *table, int num_entries, int entry_size) {

  // Make sure num entries is pow of 2
  assert((num_entries & (num_entries-1)) == 0);

  // First, re-order the table according to DPF scattered output and cvt to uint128_t_gpu
  uint128_t_gpu *table_reordered_cvted = new uint128_t_gpu[num_entries*entry_size];
  for (int j = 0; j < entry_size; j++) {
    for (int i = 0; i < num_entries; i++) {
      int reordered_indx = brev_cpu(i) >> 32 - (int)log2(num_entries);
      table_reordered_cvted[i+j*num_entries] = table[reordered_indx+j*num_entries];
    }
  }

  // Alloc and cpy to uint128_t_gpu array
  cudaMalloc(&TABLE, sizeof(uint128_t_gpu)*num_entries*entry_size);
  cudaMemcpy(TABLE, table_reordered_cvted, sizeof(uint128_t_gpu)*num_entries*entry_size, cudaMemcpyHostToDevice);
  
  delete table_reordered_cvted;
}

std::string get_DPF_strategy() {
  if (DPF_STRATEGY == DPF_NAIVE) return "Naive";
  if (DPF_STRATEGY == DPF_BREADTH_FIRST) return "Breadth-first";
  if (DPF_STRATEGY == DPF_HYBRID) return "Memory-efficient";
  if (DPF_STRATEGY == DPF_COOP) return "Cooperative threads";
}

void print_params() {
  printf("------------------------------------------------------\n");  
  printf("dpf_benchmark.cu:\n");
  printf("------------------------------------------------------\n");  
  printf("- Entries in table: %d\n", KK);
  printf("- Batch size: %d\n", NN);
  printf("- Entry size: %d\n", MM);
  printf("- PRF method: %s\n", get_PRF_method().c_str());
  printf("- DPF Strategy: %s\n", get_DPF_strategy().c_str());
  printf("- Matmul fusion: %d\n", FUSES_MATMUL);
  printf("- Perform final matmul: %d\n", PERFORM_MATMUL);  
  printf("------------------------------------------------------\n");  
}

int main() {

  print_params();
  
  // Allocate codewords
  SeedsCodewordsFlatGPU *cw_gpu;
  auto cw_cpu = GenCodewords(KK, NN, &cw_gpu);


#if(PERFORM_MATMUL)  
  // Allocate & init CPU table  
  uint128_t_gpu *table = new uint128_t_gpu[KK*MM];
  for (int j = 0; j < MM; j++) {
    for (int i = 0; i < KK; i++) table[i+KK*j] = uint128_gpu_from((uint128_t)i);
  }

  // Initialize gpu table
  initialize_table(table, KK, MM);  

  // If perform final matmul, create output buffer for it
  uint128_t_gpu *final_output_gpu;
  cudaMalloc((void **)&final_output_gpu, sizeof(uint128_t_gpu)*NN*MM);
  cudaMemset(final_output_gpu, 0, sizeof(uint128_t_gpu)*NN*MM);
#endif

  // Allocate the B vector which holds either
  // - the DPF expanded one-hot secret share (C=A*B)
  // - batch size uint128_t_gpu for output
  uint128_t_gpu *B_gpu;
#if(FUSES_MATMUL)
  B_gpu = final_output_gpu;
#else
  cudaMalloc((void **)&B_gpu, sizeof(uint128_t_gpu)*KK*NN);
  cudaMemset(B_gpu, 0, sizeof(uint128_t_gpu)*KK*NN);  
#endif

  // Timer event trackers
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Do any initialization if needed
#if(DPF_STRATEGY == DPF_BREADTH_FIRST)
  dpf_breadth_first_initialize(NN, KK);
#endif
#if(DPF_STRATEGY == DPF_HYBRID)
  dpf_hybrid_initialize(NN, KK);
#endif
#if(DPF_STRATEGY == DPF_COOP)
  dpf_coop_initialize(NN, KK, MM);
#endif

#if(PERFORM_MATMUL && !FUSES_MATMUL)
  initialize_matmul(MM, KK, NN);
#endif
  

  // 
  // Throughput benchmarks
  //

  // Use 2 streams.
  // This interleaves matmul and dpf expansion across multiple runs
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  
  for (int i = 0; i < REPS; i++) {
#if(DPF_STRATEGY == DPF_NAIVE)
    dpf_naive(cw_gpu, B_gpu, NN, KK, s1);
#endif
#if(DPF_STRATEGY == DPF_BREADTH_FIRST)
    dpf_breadth_first(cw_gpu, B_gpu, NN, KK, s1);
#endif
#if(DPF_STRATEGY == DPF_HYBRID)
    dpf_hybrid(cw_gpu, B_gpu, TABLE, NN, KK, s1);
#endif
#if(DPF_STRATEGY == DPF_COOP)
    dpf_coop(cw_gpu, B_gpu, TABLE, NN, KK, s1);
#endif
    

#if(!FUSES_MATMUL && PERFORM_MATMUL)
    // Final matmul for obtaining results
    GEMM128(TABLE, final_output_gpu, B_gpu, MM, KK, NN, s1);
#endif

#if(DPF_STRATEGY == DPF_NAIVE)
    dpf_naive(cw_gpu, B_gpu, NN, KK, s2);
#endif
#if(DPF_STRATEGY == DPF_BREADTH_FIRST)
    dpf_breadth_first(cw_gpu, B_gpu, NN, KK, s2);
#endif
#if(DPF_STRATEGY == DPF_HYBRID)
    dpf_hybrid(cw_gpu, B_gpu, TABLE, NN, KK, s2);
#endif

#if(!FUSES_MATMUL && PERFORM_MATMUL)
    // Final matmul for obtaining results
    GEMM128(TABLE, final_output_gpu, B_gpu, MM, KK, NN, s2);
#endif    
  }
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  CUDA_CHECK(cudaGetLastError());

  //
  // End throughput benchmark
  //

  //
  // Latency benchmark
  //
  cudaEvent_t start_latency, stop_latency;
  cudaEventCreate(&start_latency);
  cudaEventCreate(&stop_latency);
  cudaEventRecord(start_latency);  

#if(DPF_STRATEGY == DPF_NAIVE)
  dpf_naive(cw_gpu, B_gpu, NN, KK, s1);
#endif
#if(DPF_STRATEGY == DPF_BREADTH_FIRST)
  dpf_breadth_first(cw_gpu, B_gpu, NN, KK, s1);
#endif
#if(DPF_STRATEGY == DPF_HYBRID)
  dpf_hybrid(cw_gpu, B_gpu, TABLE, NN, KK, s1);
#endif

#if(!FUSES_MATMUL && PERFORM_MATMUL)
  // Final matmul for obtaining results
  GEMM128(TABLE, final_output_gpu, B_gpu, MM, KK, NN, s1);
#endif

#if(DPF_STRATEGY == DPF_COOP)
    dpf_coop(cw_gpu, B_gpu, TABLE, NN, KK, s1);
#endif  
    
  
  cudaEventRecord(stop_latency);
  cudaEventSynchronize(stop_latency);  
  CUDA_CHECK(cudaGetLastError());
  
  //
  // End latency benchmark
  //  

  // 
  // Check correctness if PRF is dummy method
  //
  if (PRF_METHOD == DUMMY) {

#if(PERFORM_MATMUL)
    // If fuses matmul, check correctness of the dot product
    auto B_cpu = std::vector<uint128_t_gpu>(MM * NN);
    cudaMemcpy(B_cpu.data(), final_output_gpu, sizeof(uint128_t_gpu)*MM*NN, cudaMemcpyDeviceToHost);
    check_correct_fused(cw_cpu.data(), B_cpu.data(), table, MM, NN, KK);
#else
    // If expanding the full DPF, check the correctness of the expanded DPF only    
    auto B_cpu = std::vector<uint128_t_gpu>(KK * NN);
    cudaMemcpy(B_cpu.data(), B_gpu, sizeof(uint128_t_gpu)*KK*NN, cudaMemcpyDeviceToHost);
    check_correct(cw_cpu.data(), B_cpu.data(), NN, KK, DPF_STRATEGY==DPF_BREADTH_FIRST || DPF_STRATEGY==DPF_HYBRID || DPF_STRATEGY==DPF_COOP);
#endif
  }

  //
  // Log benchmark output to dict
  //
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float throughput_per_query = NN*REPS*2/ms;

  float ms_latency = 0;
  cudaEventElapsedTime(&ms_latency, start_latency, stop_latency);

  // Final logging output
  printf("{'entries': %d, 'entry_size_ints': %d, 'batch_size': %d,"
	 "'prf_method': '%s', 'dpf_strategy': '%s', "
	 "'latency_ms' : %f, 'throughput_queries_per_ms' : %f,"
	 "'fuses_matmul' : %d, 'performs_matmul' : %d}\n",
	 KK, MM, NN,
	 get_PRF_method().c_str(), get_DPF_strategy().c_str(),
	 ms_latency, throughput_per_query, FUSES_MATMUL,
	 PERFORM_MATMUL);  
  
  cudaFree(B_gpu);
  cudaFree(cw_gpu);

#if(PERFORM_MATMUL)
  cudaFree(final_output_gpu);
  delete table;  
#endif
}
