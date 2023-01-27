// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#ifndef UTILS
#define UTILS

#include "../dpf_base/dpf.h"
#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// 128-bit functionalities                                                    //
// from: https://stackoverflow.com/questions/6162140/128-bit-integer-on-cuda  //
////////////////////////////////////////////////////////////////////////////////
typedef uint4 uint128_t_gpu;

uint128_t_gpu uint128_gpu_from(uint128_t val) {
  uint128_t_gpu res;
  res.w = (val >> 96) & 0xFFFFFFFF;
  res.z = (val >> 64) & 0xFFFFFFFF;
  res.y = (val >> 32) & 0xFFFFFFFF;
  res.x = (val >> 0) & 0xFFFFFFFF;
  return res;
}

uint128_t uint128_from_gpu(uint128_t_gpu val) {
  uint128_t res = 0;
  return val.x +
    ((uint128_t)val.y << 32) +
    ((uint128_t)val.z << 64) +
    ((uint128_t)val.w << 96);
}

__device__ uint128_t_gpu uint128_from(uint64_t hi,
				      uint64_t lo) {
  uint128_t_gpu res;
  res.w = (hi >> 32);
  res.z = hi & 0x00000000FFFFFFFF;
  res.y = (lo >> 32);
  res.x = lo & 0x00000000FFFFFFFF;
  return res;
}

__device__ uint128_t_gpu add_uint128(uint128_t_gpu addend, uint128_t_gpu augend)
{
    uint128_t_gpu res;
    asm ("add.cc.u32      %0, %4, %8;\n\t"
         "addc.cc.u32     %1, %5, %9;\n\t"
         "addc.cc.u32     %2, %6, %10;\n\t"
         "addc.u32        %3, %7, %11;\n\t"
         : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
         : "r"(addend.x), "r"(addend.y), "r"(addend.z), "r"(addend.w),
           "r"(augend.x), "r"(augend.y), "r"(augend.z), "r"(augend.w));
    return res;
}

__device__ uint128_t_gpu mul_uint128(uint128_t_gpu a, uint128_t_gpu b)
{
    uint128_t_gpu res;
    asm ("{\n\t"
         "mul.lo.u32      %0, %4, %8;    \n\t"
         "mul.hi.u32      %1, %4, %8;    \n\t"
         "mad.lo.cc.u32   %1, %4, %9, %1;\n\t"
         "madc.hi.u32     %2, %4, %9,  0;\n\t"
         "mad.lo.cc.u32   %1, %5, %8, %1;\n\t"
         "madc.hi.cc.u32  %2, %5, %8, %2;\n\t"
         "madc.hi.u32     %3, %4,%10,  0;\n\t"
         "mad.lo.cc.u32   %2, %4,%10, %2;\n\t"
         "madc.hi.u32     %3, %5, %9, %3;\n\t"
         "mad.lo.cc.u32   %2, %5, %9, %2;\n\t"
         "madc.hi.u32     %3, %6, %8, %3;\n\t"
         "mad.lo.cc.u32   %2, %6, %8, %2;\n\t"
         "madc.lo.u32     %3, %4,%11, %3;\n\t"
         "mad.lo.u32      %3, %5,%10, %3;\n\t"
         "mad.lo.u32      %3, %6, %9, %3;\n\t"
         "mad.lo.u32      %3, %7, %8, %3;\n\t"
         "}"
         : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
         : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
           "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
    return res;
}

// Error check functionality
inline void error_check(cudaError_t err, const char* file, int line) {
    if(err != cudaSuccess) {
        ::fprintf(stderr, "CUDA ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
        abort();
    }
}
#define CUDA_CHECK(err) do { error_check(err, __FILE__, __LINE__); } while(0)

// SeedsCodewordsFlat for GPU (replaces uint128_t with vector version)
struct SeedsCodewordsFlatGPU {
  int depth;
  uint128_t_gpu cw_1[64], cw_2[64];
  uint128_t_gpu last_keys[1];
};

SeedsCodewordsFlatGPU SeedsCodewordsFlatGPUFromCPU(SeedsCodewordsFlat &f) {
  SeedsCodewordsFlatGPU g;
  g.depth = f.depth;
  for (int i = 0; i < 64; i++) {
    g.cw_1[i] = uint128_gpu_from(f.cw_1[i]);
    g.cw_2[i] = uint128_gpu_from(f.cw_2[i]);    
  }
  g.last_keys[0] = uint128_gpu_from(f.last_keys[0]);
  return g;
}

/*// Generates dummy codewords for testint
std::vector<SeedsCodewordsFlat> GenCodewords(int k, int n,
					     SeedsCodewordsFlatGPU **cw_gpu) {  

  //auto cw_cpu = std::vector<SeedsCodewordsFlat>(n);
  for (int i = 0; i < n; i++) {
    
    std::mt19937 g_gen(i);
    int alpha = (100+i) % k;
    int beta = 4242+i;
    
    SeedsCodewords *s = GenerateSeedsAndCodewordsLog(alpha, beta, k, g_gen);    
    FlattenCodewords(s, 0, &cw_cpu[i]);
    FreeSeedsCodewords(s);
  }

  // Convert codewords to gpu rep
  SeedsCodewordsFlatGPU *cw_intermediate = (SeedsCodewordsFlatGPU *)malloc(sizeof(SeedsCodewordsFlatGPU)*n);
  for (int i = 0; i < n; i++) {
    cw_intermediate[i] = SeedsCodewordsFlatGPUFromCPU(cw_cpu[i]);
  }

  cudaMalloc((void **)cw_gpu, sizeof(SeedsCodewordsFlatGPU)*n);
  cudaMemcpy(*cw_gpu, cw_intermediate, sizeof(SeedsCodewordsFlatGPU)*(n), cudaMemcpyHostToDevice);
  free(cw_intermediate);

  return cw_cpu;
  }*/

// https://stackoverflow.com/questions/9144800/c-reverse-bits-in-unsigned-integer
uint32_t brev_cpu(uint32_t x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}

// Correctness checks the output of GPU kernel code
void check_correct(SeedsCodewordsFlat *cw, uint128_t_gpu *target,
		   int batch_size, int num_entries,
		   int permutated_ordering) {
  int zz = 0;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_entries; j++) {
      
      uint128_t truth = EvaluateFlat(&cw[i], j, 0);      
      uint128_t_gpu truth_128_t_gpu = uint128_gpu_from(truth);

      // This is the "standard" ordering
      uint128_t_gpu got;
      if (!permutated_ordering) {
	got = target[j*batch_size+i];
      }
      else {
	// This is the "permutated" ordering
	//int tgt_indx = brev_cpu(j) >> 32 - cw[0].depth;
	int tgt_indx = brev_cpu(j) >> 32 - (int)log2(num_entries);
	got = target[tgt_indx + i*num_entries];
      }

      // For debugging
      //printf("Got   : %d %d %d %d\n", got.x, got.y, got.z, got.w);
      //printf("Expect: %d %d %d %d\n", truth_128_t_gpu.x, truth_128_t_gpu.y, truth_128_t_gpu.z, truth_128_t_gpu.w);
      //zz += 1;
      //if (zz >= 100) return;
      
      assert(got.x == truth_128_t_gpu.x &&
	got.y == truth_128_t_gpu.y &&
	got.z == truth_128_t_gpu.z &&
	got.w == truth_128_t_gpu.w);
    }
  }
  printf("PASS\n");
}

void check_correct_fused(SeedsCodewordsFlat *cw, uint128_t_gpu *target, uint128_t_gpu *table,
			 int entry_size, int batch_size, int num_entries) {
  for (int i = 0; i < batch_size; i++) {
    for (int k = 0; k < entry_size; k++) {
      uint128_t accum = 0;
      for (int j = 0; j < num_entries; j++) {
	uint128_t truth = EvaluateFlat(&cw[i], j, 0);            
	accum += truth * uint128_from_gpu(table[j+k*num_entries]);
      }

      uint128_t_gpu cmp = uint128_gpu_from(accum);
      uint128_t_gpu got = target[i+k*batch_size];

      assert(got.x == cmp.x &&
	     got.y == cmp.y &&
	     got.z == cmp.z &&
	     got.w == cmp.w);
    }
  }
  printf("PASS MATMUL CHECK\n");
}

#endif
