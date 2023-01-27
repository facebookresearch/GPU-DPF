// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#include <cooperative_groups.h>

using namespace cooperative_groups;

#ifndef FUSES_MATMUL
#define FUSES_MATMUL 1
#endif

//#define DPF_COOP_N_BLOCKS 64
#define DPF_COOP_THREADS_PER_BLOCK 128

int DPF_COOP_N_BLOCKS = -1;

uint128_t_gpu *DPF_COOP_KEYS_1, *DPF_COOP_KEYS_2;
uint128_t_gpu *TABLE_REDUCTION;

__global__ void dpf_coop_kernel(SeedsCodewordsFlatGPU *cw,
				uint128_t_gpu *TABLE,
				uint128_t_gpu *TABLE_REDUCTION,				
				uint128_t_gpu *out,
				uint128_t_gpu *DPF_COOP_KEYS_1,
				uint128_t_gpu *DPF_COOP_KEYS_2,
				int batch_size, int num_entries,
				int DPF_COOP_N_BLOCKS) {
  
  // Computes DPF expansion in a breadth parallel way
  int thread_idx = threadIdx.x;
  int block_idx = blockIdx.x;

  // Load cw to shared memory. Recall only 1 cw as batchsize=1
  __shared__ SeedsCodewordsFlatGPU cw_shared[1];
  if (thread_idx == 0) {
    cw_shared[thread_idx] = cw[0];
  }

  // Use cooperative groups to sync blocks
  this_grid().sync();
  __syncthreads();

  // Algorithm same as breadth parallel, see breadth parallel method for high level DPF strat
  uint128_t_gpu *key_write = DPF_COOP_KEYS_1;
  uint128_t_gpu *key_read = DPF_COOP_KEYS_2;
  uint128_t_gpu *tmp;
  
  // Init the first seed
  key_write[0] = cw_shared[0].last_keys[0];

  // Outer loop loops from top level of tree down to bottom
  for (int i = cw_shared[0].depth-1; i >= 0; i--) {

    // Swap read and write buffers
    tmp = key_read;
    key_read = key_write;
    key_write = tmp;

    // Can parallelize _within_ a level of the tree, but not _across_ levels of the tree
    this_grid().sync();
    __syncthreads();
    
    // Inner loop scans the current level of the tree (in parallel batches)
    int start = 0, end = 1<<(cw_shared[0].depth-i);

    // Scan through the work. All threads of each block eval a single PRF
    for (int j = start; j < end; j += DPF_COOP_N_BLOCKS*DPF_COOP_THREADS_PER_BLOCK) {
      int expansion_idx = j + (block_idx*DPF_COOP_THREADS_PER_BLOCK + thread_idx);
      
      if (expansion_idx < end) {
	int idx_into_codewords = expansion_idx % 2;
	uint128_t_gpu key = key_read[(expansion_idx/2)];
	uint128_t_gpu value = PRF(key, idx_into_codewords);
	uint128_t_gpu *cw = (key.x & 1) == 0 ? cw_shared[0].cw_1 : cw_shared[0].cw_2;
	cw = &cw[i*2];
	key_write[expansion_idx] = add_uint128(value, cw[idx_into_codewords]);
      }
    }
  }

#if(!FUSES_MATMUL)
  // Postamble, write to output
  for (int i = 0; i < num_entries; i += DPF_COOP_N_BLOCKS*DPF_COOP_THREADS_PER_BLOCK) {
    int expansion_idx = i + (block_idx*DPF_COOP_THREADS_PER_BLOCK + thread_idx);

    // Do note: the best way to write memory is with _coalescing_.
    // Without it, huge performance slowdowns (2.5x slowdown!)
    // However, this writes to the output buffer in a permutated order.
    if (expansion_idx < num_entries) {
      out[expansion_idx] = key_write[expansion_idx];
    }
  }
#else

  // Fused matmul. Recall MM is num_elements_per_entry
  uint128_t_gpu per_thread_accumulate[MM] = {0};
  for (int i = 0; i < num_entries; i += DPF_COOP_N_BLOCKS*DPF_COOP_THREADS_PER_BLOCK) {
    int expansion_idx = i + (block_idx*DPF_COOP_THREADS_PER_BLOCK + thread_idx);
    if (expansion_idx < num_entries) {
      for (int z = 0; z < MM; z++) {
	per_thread_accumulate[z] = add_uint128(mul_uint128(key_write[expansion_idx], TABLE[expansion_idx]),
					       per_thread_accumulate[z]);
      }
    }
  }

  // Tree sum reduction on accumulates
  int total_threads = DPF_COOP_N_BLOCKS*DPF_COOP_THREADS_PER_BLOCK;
  int glob_thread_idx = block_idx*DPF_COOP_THREADS_PER_BLOCK+thread_idx;

  // Write local accumulates to table
  for (int i = 0; i < MM; i++) {
    TABLE_REDUCTION[i*total_threads+glob_thread_idx] = per_thread_accumulate[i];
  }

  this_grid().sync();
  __syncthreads();

  for (int neighbor = 1; neighbor < total_threads; neighbor*=2) {
      if (glob_thread_idx % (neighbor*2) == 0 && glob_thread_idx+neighbor < total_threads) {
	for (int z = 0; z < MM; z++) {
	  TABLE_REDUCTION[z*total_threads+glob_thread_idx] =
	    add_uint128(TABLE_REDUCTION[z*total_threads+glob_thread_idx],
			TABLE_REDUCTION[z*total_threads+glob_thread_idx+neighbor]);
	}
      }
      this_grid().sync();
      __syncthreads();        
  }

  if (glob_thread_idx == 0) {
    for (int z = 0; z < MM; z++) {
      out[z] = TABLE_REDUCTION[z*total_threads+0];
    }
  }
  
#endif
}

int getMaxInterpreterGrid(int numThreads) {
  int maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, dpf_coop_kernel, numThreads, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int numSM = deviceProp.multiProcessorCount;

  return maxBlocksPerSM * numSM;
}

void dpf_coop_initialize(int batch_size, int num_entries, int entry_size) {
  // Same as breadth parallel strategy, except always uses
  // batch 1.
  if (batch_size != 1) {
    printf("Cooperative threads DPF strategy requires batch_size=1\n");
  }  
  assert(batch_size == 1);

  DPF_COOP_N_BLOCKS = getMaxInterpreterGrid(DPF_COOP_THREADS_PER_BLOCK);
  printf("Coooperative threads DPF strategy with grid size %d\n", DPF_COOP_N_BLOCKS);
  
  cudaMalloc(&DPF_COOP_KEYS_1, sizeof(uint128_t_gpu)*batch_size*num_entries);
  cudaMalloc(&DPF_COOP_KEYS_2, sizeof(uint128_t_gpu)*batch_size*num_entries);

  // Given batch size 1, we also initialize a table of size num_entries*entry_size
  // for the purpose of reducing the final accumulates
  cudaMalloc(&TABLE_REDUCTION, sizeof(uint128_t_gpu)*entry_size*DPF_COOP_N_BLOCKS*DPF_COOP_THREADS_PER_BLOCK);
}

void dpf_coop(SeedsCodewordsFlatGPU * cw,
	      uint128_t_gpu *out,
	      uint128_t_gpu *TABLE,
	      int batch_size, int num_entries,
	      cudaStream_t s) {
  dim3 n_blocks(DPF_COOP_N_BLOCKS);
  dim3 n_threads(DPF_COOP_THREADS_PER_BLOCK);

  void *kernel_args[] =
    {
     &cw, &TABLE, &TABLE_REDUCTION, &out,
     &DPF_COOP_KEYS_1,
     &DPF_COOP_KEYS_2,
     &batch_size,
     &num_entries,
     &DPF_COOP_N_BLOCKS,
    };
  cudaLaunchCooperativeKernel((void *)dpf_coop_kernel,
			      n_blocks, n_threads, kernel_args);
}

