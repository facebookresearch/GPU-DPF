// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#include <torch/extension.h>
#include <iostream>
#include <random>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Number of uint128_ts per entry
#define MM 16

// Batch size
#define BATCH_SIZE 512

#include "dpf_base/dpf.h"
#include "dpf_gpu/dpf/dpf_hybrid.cu"

at::Tensor key_from_codewords(SeedsCodewordsFlat *k, int n) {
  at::Tensor key = torch::zeros({524}, at::kInt);
  uint128_t *key_ptr = (uint128_t *)key.data_ptr();
  key_ptr[0] = k->depth;
  memcpy(&key_ptr[1], k->cw_1, sizeof(uint128_t)*64);
  memcpy(&key_ptr[65], k->cw_2, sizeof(uint128_t)*64);
  key_ptr[129] = k->last_keys[0];
  key_ptr[130] = n;
  return key;
}

SeedsCodewordsFlat *codewords_from_key(at::Tensor &key, int *n) {
  SeedsCodewordsFlat *k = new SeedsCodewordsFlat;
  uint128_t *key_ptr = (uint128_t *)key.data_ptr();
  k->depth = key_ptr[0];
  memcpy(k->cw_1, &key_ptr[1], sizeof(uint128_t)*64);
  memcpy(k->cw_2, &key_ptr[65], sizeof(uint128_t)*64);
  k->last_keys[0] = key_ptr[129];
  *n = key_ptr[130];
  return k;
}

// Note: s1, and s2 are seeds that concatenate to form a 128-bit seed
std::vector<at::Tensor> gen(int k, int n, char *seed, int prf_method) {
  
  // Generate DPF codewords and flatten them
  std::mt19937 g_gen(*(uint128_t *)seed);
  SeedsCodewords *s = GenerateSeedsAndCodewordsLog(k, 1, n, g_gen, prf_method);
  SeedsCodewordsFlat *k_1 = new SeedsCodewordsFlat;
  SeedsCodewordsFlat *k_2 = new SeedsCodewordsFlat;    
  FlattenCodewords(s, 0, k_1);
  FlattenCodewords(s, 1, k_2);

  // Copy over to tensor
  at::Tensor key_1 = key_from_codewords(k_1, n);
  at::Tensor key_2 = key_from_codewords(k_2, n);  
 
  FreeSeedsCodewords(s);
  free(k_1);
  free(k_2);

  return {key_1, key_2};
}

at::Tensor eval_dpf_cpu(at::Tensor key, int prf_method) {

  int n;
  SeedsCodewordsFlat *k = codewords_from_key(key, &n);

  // Expand codewords
  at::Tensor result = torch::ones({n}, at::kInt);

  // CPU expansion
  for (int i = 0; i < n; i++) {
    result[i] = (int)EvaluateFlat(k, i, prf_method);
  }
  
  return result;
}

void eval_free(std::vector<void *> buffers) {
  cudaFree(buffers[0]);
  cudaFree(buffers[1]);
  cudaFree(buffers[2]);
  dpf_hybrid_deinitialize();
}

std::vector<void *> eval_init(at::Tensor table) {

  
  int num_entries = table.size(0);
  int entry_size = table.size(1);

  assert((num_entries & (num_entries-1)) == 0);
  assert(entry_size == MM);

  // Initialize the table on GPU memory
  uint128_t_gpu *table_reordered_cvted = new uint128_t_gpu[num_entries*entry_size];
  for (int j = 0; j < entry_size; j++) {
    for (int i = 0; i < num_entries; i++) {
      int reordered_indx = brev_cpu(i) >> 32 - (int)log2(num_entries);
      table_reordered_cvted[i+j*num_entries] = uint128_gpu_from((uint128_t)table[reordered_indx][j].item<int>());
    }
  }

  uint128_t_gpu *TABLE;  

  // Alloc and cpy to uint128_t_gpu array
  gpuErrchk(cudaMalloc(&TABLE, sizeof(uint128_t_gpu)*num_entries*entry_size));
  cudaMemcpy(TABLE, table_reordered_cvted, sizeof(uint128_t_gpu)*num_entries*entry_size, cudaMemcpyHostToDevice);
  
  delete table_reordered_cvted;

  // Allocate gpu buffer for the input keys
  SeedsCodewordsFlatGPU *CW_GPU;
  gpuErrchk(cudaMalloc((void **)&CW_GPU, sizeof(SeedsCodewordsFlatGPU)*BATCH_SIZE));

  // Allocate gpu buffer for the output
  uint128_t_gpu *OUT;
  gpuErrchk(cudaMalloc((void **)&OUT, sizeof(uint128_t_gpu)*BATCH_SIZE*MM));
  cudaMemset(OUT, sizeof(uint128_t_gpu)*BATCH_SIZE*MM, 0);

  // Initialize hybrid strat
  dpf_hybrid_initialize(BATCH_SIZE, num_entries);
  
  return {TABLE, CW_GPU, OUT};
}		     

at::Tensor eval_gpu(std::vector<at::Tensor> keys, std::vector<void *> buffers, int n, int prf_method) {
  assert(keys.size() == BATCH_SIZE);

  SeedsCodewordsFlatGPU *cw_intermediate = (SeedsCodewordsFlatGPU *)malloc(sizeof(SeedsCodewordsFlatGPU)*BATCH_SIZE);  
  
  // Convert seeds/codewords to CW_GPU
  for (int i = 0; i < keys.size(); i++) {
    int num_entries = 0;      
    SeedsCodewordsFlat *k = codewords_from_key(keys[i], &num_entries);
    assert(num_entries == n);
    cw_intermediate[i] = SeedsCodewordsFlatGPUFromCPU(*k);    
    free(k);
  }

  // Copy to codewords to GPU buffer
  SeedsCodewordsFlatGPU *CW_GPU = (SeedsCodewordsFlatGPU *)buffers[1];
  cudaMemcpy(CW_GPU, cw_intermediate, sizeof(SeedsCodewordsFlatGPU)*(keys.size()), cudaMemcpyHostToDevice);  
  uint128_t_gpu *TABLE = (uint128_t_gpu *)buffers[0];
  uint128_t_gpu *OUT = (uint128_t_gpu *)buffers[2];  

  // Perform batched dpf lookup
  cudaStream_t s;
  cudaStreamCreate(&s);
  if (prf_method == DUMMY) {
    dpf_hybrid<DUMMY>(CW_GPU, OUT, TABLE, BATCH_SIZE, n, s);
  }
  else if (prf_method == SALSA20) {
    dpf_hybrid<SALSA20>(CW_GPU, OUT, TABLE, BATCH_SIZE, n, s);    
  }
  else if (prf_method == CHACHA20) {
    dpf_hybrid<CHACHA20>(CW_GPU, OUT, TABLE, BATCH_SIZE, n, s);    
  }
  else if (prf_method == AES128) {
    dpf_hybrid<AES128>(CW_GPU, OUT, TABLE, BATCH_SIZE, n, s);    
  }
  
  else {
    assert(0);
  }

  // Cvt GPU output to CPU output
  uint128_t_gpu out_cpu[BATCH_SIZE*MM];
  cudaMemcpy(out_cpu, OUT, sizeof(uint128_t_gpu)*BATCH_SIZE*MM, cudaMemcpyDeviceToHost);

  at::Tensor result = torch::zeros({BATCH_SIZE, MM}, at::kInt);  
  uint32_t *r = (uint32_t *)result.data_ptr();
  for (int i = 0; i < BATCH_SIZE; i++) {
    for (int j = 0; j < MM; j++) {
      r[i*MM+j] = (uint32_t)uint128_from_gpu(out_cpu[i+j*BATCH_SIZE]);
    }
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Funcs
  m.def("gen", &gen, "dpf gen");
  m.def("eval_cpu", &eval_dpf_cpu, "dpf eval cpu");
  m.def("eval_gpu", &eval_gpu, "dpf eval gpu");
  m.def("eval_init", &eval_init, "dpf eval init");
  m.def("eval_free", &eval_free, "dpf eval free");

  // Consts
  m.attr("ENTRY_SIZE") = py::int_(MM);
  m.attr("BATCH_SIZE") = py::int_(BATCH_SIZE);

  m.attr("PRF_DUMMY") = py::int_(DUMMY);
  m.attr("PRF_SALSA20") = py::int_(SALSA20);
  m.attr("PRF_CHACHA20") = py::int_(CHACHA20);
  m.attr("PRF_AES128") = py::int_(AES128);        
}
