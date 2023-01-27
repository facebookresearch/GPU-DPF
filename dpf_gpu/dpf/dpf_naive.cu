// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#include "../utils.h"

#define DPF_NAIVE_BLOCK_W (8)
#define DPF_NAIVE_BLOCK_H (128)

__device__ uint128_t_gpu expand_dpf_naive_kernel(const SeedsCodewordsFlatGPU *s, int indx) {
  
  int indx_remaining = indx;  
  uint128_t_gpu key = s->last_keys[0];
  uint128_t_gpu value;
  
  for (int i = s->depth-1; i >= 0; i--) {
    int indx_into_codewords = indx_remaining % 2;
    value = PRF(key, indx_into_codewords);
    const uint128_t_gpu *cw = (key.x & 1) == 0 ? s->cw_1 : s->cw_2;
    cw = &cw[i*2];
    key = add_uint128(value, cw[indx_into_codewords]);    
    indx_remaining >>= 1;
  }

  return key;  
}

__global__ void dpf_naive_kernel(SeedsCodewordsFlatGPU *cw,
				 uint128_t_gpu *out,
				 int batch_size) {
  
  int x_indx = blockIdx.x*DPF_NAIVE_BLOCK_W + threadIdx.x;
  int y_indx = blockIdx.y*DPF_NAIVE_BLOCK_H + threadIdx.y;
  int out_indx = y_indx*batch_size + x_indx;

  out[out_indx] = expand_dpf_naive_kernel(&cw[x_indx], y_indx);
}

void dpf_naive(SeedsCodewordsFlatGPU *cw,
	       uint128_t_gpu *out,
	       int batch_size, int num_entries,
	       cudaStream_t s) {
  dim3 threads_per_block_naive(DPF_NAIVE_BLOCK_W, DPF_NAIVE_BLOCK_H);
  dim3 n_blocks_naive(batch_size/DPF_NAIVE_BLOCK_W, num_entries/DPF_NAIVE_BLOCK_H);

  //printf("%d %d\n", batch_size/DPF_NAIVE_BLOCK_W, num_entries/DPF_NAIVE_BLOCK_H);
  //return;

  dpf_naive_kernel<<<n_blocks_naive, threads_per_block_naive, 0, s>>>(cw, out, batch_size);
}
