// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

// This is like batch size: a block expands _multiple_ dpfs
#define DPF_BREADTH_PARALLEL_THREADS_PER_BLOCK 256
#define DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK 1

uint128_t_gpu *DPF_BREADTH_PARALLEL_KEYS_1, *DPF_BREADTH_PARALLEL_KEYS_2;

void dpf_breadth_first_initialize(int batch_size, int num_entries) {
  cudaMalloc(&DPF_BREADTH_PARALLEL_KEYS_1, sizeof(uint128_t_gpu)*batch_size*num_entries);
  cudaMalloc(&DPF_BREADTH_PARALLEL_KEYS_2, sizeof(uint128_t_gpu)*batch_size*num_entries);
}

__global__ void dpf_breadth_first_kernel(SeedsCodewordsFlatGPU *cw, uint128_t_gpu *out,
					 uint128_t_gpu *DPF_BREADTH_PARALLEL_KEYS_1,
					 uint128_t_gpu *DPF_BREADTH_PARALLEL_KEYS_2,
					 int batch_size, int num_entries) {
  
  // Computes DPF expansion in a breadth parallel way
  int thread_idx = threadIdx.x;
  int block_idx = blockIdx.x;
  
  // This block handles DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK DPF expansions
  int cw_start = blockIdx.x*DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK;
  int cw_end = cw_start+DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK;  

  // Load cw to shared memory
  __shared__ SeedsCodewordsFlatGPU cw_shared[DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK];
  if (thread_idx < DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK) {
    cw_shared[thread_idx] = cw[cw_start+thread_idx];
  }

  __syncthreads();

  // Simple recurrence relation for expanding binary tree-based DPF.
  // Nodes numbered with following format:
  //      0
  //    /   \
  //   0     1
  //  / \   / \
  // 0   1 2   3
  //
  // Relation:
  // k_1 = seed
  // k_i = PRF(k_{i//2}, i % 2) + CW_{k_{i//2} & 1}(i % 2)
  //
  // Output k_{2^{depth-1}} to k_{2^{depth-1}} + N
  //
  // Do note, we are expanding multiple binary tree DPFs.
  // In this algo, each threadblock expands DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK dpfs.
  // Following checks ensure blocking params are correct:
  // assert(DPF_BREADTH_PARALLEL_THREADS_PER_BLOCK/DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK >= 1)
  // assert(DPF_BREADTH_PARALLEL_THREADS_PER_BLOCK%DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK == 0)
  uint128_t_gpu *key_write = DPF_BREADTH_PARALLEL_KEYS_1;
  uint128_t_gpu *key_read = DPF_BREADTH_PARALLEL_KEYS_2;
  uint128_t_gpu *tmp;
  
  constexpr int parallel_work_per_threadblock_per_dpf = (DPF_BREADTH_PARALLEL_THREADS_PER_BLOCK/DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK);

  // Init the first seed
  int batch_idx = thread_idx % DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK;
  key_write[0 + (block_idx+batch_idx)*num_entries] = cw_shared[batch_idx].last_keys[0];
  
  // Outer loop loops from top level of tree down to bottom
  for (int i = cw_shared[0].depth-1; i >= 0; i--) {

    // Swap read and write buffers
    tmp = key_read;
    key_read = key_write;
    key_write = tmp;

    // Can parallelize _within_ a level of the tree, but not _across_ levels of the tree
    __syncthreads();    
    
    // Inner loop scans the current level of the tree (in parallel batches)
    int start = 0, end = 1<<(cw_shared[0].depth-i);
    for (int j = start; j < end; j += parallel_work_per_threadblock_per_dpf) {
      int expansion_idx = j + (thread_idx % parallel_work_per_threadblock_per_dpf);
      int batch_idx = thread_idx / parallel_work_per_threadblock_per_dpf;
      
      if (expansion_idx < end) {
	int idx_into_codewords = expansion_idx % 2;
	uint128_t_gpu key = key_read[(expansion_idx/2) + (block_idx+batch_idx)*num_entries];
	uint128_t_gpu value = PRF(key, idx_into_codewords);
	uint128_t_gpu *cw = (key.x & 1) == 0 ? cw_shared[batch_idx].cw_1 : cw_shared[batch_idx].cw_2;
	cw = &cw[i*2];
	key_write[expansion_idx + (block_idx+batch_idx)*num_entries] = add_uint128(value, cw[idx_into_codewords]);
      }
    }
  }
  
  // Postamble, write to output
  for (int i = 0; i < num_entries; i+= parallel_work_per_threadblock_per_dpf) {
    int expansion_idx = i + (thread_idx % parallel_work_per_threadblock_per_dpf);
    int batch_idx = thread_idx / parallel_work_per_threadblock_per_dpf;
    int dst_idx = __brev(expansion_idx) >> (32-cw_shared[0].depth);

    // Do note: the best way to write memory is with _coalescing_.
    // Without it, huge performance slowdowns (2.5x slowdown!)
    // However, this writes to the output buffer in a permutated order.
    out[expansion_idx + batch_idx*num_entries + block_idx*DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK*num_entries] =
      key_write[expansion_idx + (block_idx+batch_idx)*num_entries];
  }  
}

void dpf_breadth_first(SeedsCodewordsFlatGPU *cw,
		       uint128_t_gpu *out,
		       int batch_size, int num_entries,
		       cudaStream_t s) {
  dim3 n_blocks_breadth_parallel(batch_size / DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK);
  dim3 n_threads_breadth_parallel(DPF_BREADTH_PARALLEL_THREADS_PER_BLOCK);
  
  dpf_breadth_first_kernel<<<n_blocks_breadth_parallel, n_threads_breadth_parallel, 0, s>>>(cw, out,
											 DPF_BREADTH_PARALLEL_KEYS_1,
											 DPF_BREADTH_PARALLEL_KEYS_2,
											 batch_size,
											 num_entries);
}
