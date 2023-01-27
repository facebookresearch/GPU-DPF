// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


// Hybrid DPF GPU strategy for memory efficient DPF expansion

// Z parameter defines how many DPF nodes we process in parallel
// at a time. Must be power of 2.
//
// Memory complexity = O(BZLog(N)), where B is batch,
// N is number of table entries
#define Z (128)

uint128_t_gpu *DPF_HYBRID_STACK_1, *DPF_HYBRID_STACK_2;

#define DPF_HYBRID_THREADS_PER_BLOCK 128
#define DPF_HYBRID_DPFS_PER_BLOCK 1

#ifndef FUSES_MATMUL
#define FUSES_MATMUL 1
#endif

void dpf_hybrid_initialize(int batch_size, int num_entries) {  
  size_t n_bytes = sizeof(uint128_t_gpu)*batch_size*((int)ceil(log2((float)num_entries)))*Z*2;
  size_t megabytes = n_bytes / 1024 / 1024;
  printf("dpf_hybrid_initialize: stack size = %zu MB\n", megabytes);
  cudaMalloc(&DPF_HYBRID_STACK_1, n_bytes);
  cudaMalloc(&DPF_HYBRID_STACK_2, n_bytes);
}

__global__ void dpf_hybrid_kernel(SeedsCodewordsFlatGPU *cw, uint128_t_gpu *out,
				  uint128_t_gpu *DPF_HYBRID_STACK_1,
				  uint128_t_gpu *DPF_HYBRID_STACK_2,
				  uint128_t_gpu *TABLE,
				  int batch_size, int num_entries) {
  
  /////////////////////////////////////////////////////////////////////////
  // Step 1) Perform breadth-parallel expansion up until K*2 nodes. See  //
  //         the breadth parallel file for details.                      //
  /////////////////////////////////////////////////////////////////////////

  int thread_idx = threadIdx.x;
  int block_idx = blockIdx.x;

  // This block handles DPF_BREADTH_PARALLEL_DPFS_PER_BLOCK DPF expansions
  int cw_start = blockIdx.x*DPF_HYBRID_DPFS_PER_BLOCK;
  int cw_end = cw_start+DPF_HYBRID_DPFS_PER_BLOCK;  

  // Load cw to shared memory
  __shared__ SeedsCodewordsFlatGPU cw_shared[DPF_HYBRID_DPFS_PER_BLOCK];
  if (thread_idx < DPF_HYBRID_DPFS_PER_BLOCK) {
    cw_shared[thread_idx] = cw[cw_start+thread_idx];
  }

  __syncthreads();
  
  uint128_t_gpu *key_write = DPF_HYBRID_STACK_1;
  uint128_t_gpu *key_read = DPF_HYBRID_STACK_2;
  uint128_t_gpu *tmp;
  
  constexpr int parallel_work_per_threadblock_per_dpf = (DPF_HYBRID_THREADS_PER_BLOCK/DPF_HYBRID_DPFS_PER_BLOCK);

  // Init the first seed
  int LOGZ = log2((float)Z);
  int max_stack_size_per_dpf = ceil(log2((float)num_entries))*Z*2;
  int batch_idx = thread_idx % DPF_HYBRID_DPFS_PER_BLOCK;
  key_write[0 + (block_idx+batch_idx)*max_stack_size_per_dpf] = cw_shared[batch_idx].last_keys[0];
  int prev_start = 0, prev_end = 0;
  
  // Outer loop loops from the Z+1'th level of the tree down.
  // Afterwards, there should be Z nodes on the stack.
  for (int i = cw_shared[0].depth-1; i >= cw_shared[0].depth-LOGZ; i--) {

    // Swap read and write buffers
    tmp = key_read;
    key_read = key_write;
    key_write = tmp;

    // Can parallelize _within_ a level of the tree, but not _across_ levels of the tree
    __syncthreads();    
    
    // Inner loop scans the current level of the tree (in parallel batches)
    int start = 0, end = 1<<(cw_shared[0].depth-i);
    prev_start = start;
    prev_end = end;
    for (int j = start; j < end; j += parallel_work_per_threadblock_per_dpf) {
      int expansion_idx = j + (thread_idx % parallel_work_per_threadblock_per_dpf);
      int batch_idx = thread_idx / parallel_work_per_threadblock_per_dpf;
      
      if (expansion_idx < end) {
	int idx_into_codewords = expansion_idx % 2;
	uint128_t_gpu key = key_read[(expansion_idx/2) + (block_idx+batch_idx)*max_stack_size_per_dpf];
	uint128_t_gpu value = PRF(key, idx_into_codewords);
	uint128_t_gpu *cw = (key.x & 1) == 0 ? cw_shared[batch_idx].cw_1 : cw_shared[batch_idx].cw_2;
	cw = &cw[(i)*2];
	key_write[expansion_idx + (block_idx+batch_idx)*max_stack_size_per_dpf] = add_uint128(value, cw[idx_into_codewords]);
      }
    }
  }

    // Sync read and write buffers
  __syncthreads();
  
  for (int j = prev_start; j < prev_end; j += parallel_work_per_threadblock_per_dpf) {
      int expansion_idx = j + (thread_idx % parallel_work_per_threadblock_per_dpf);
      int batch_idx = thread_idx / parallel_work_per_threadblock_per_dpf;
      
      if (expansion_idx < prev_end) {
	key_read[expansion_idx + (block_idx+batch_idx)*max_stack_size_per_dpf] =
	  key_write[expansion_idx + (block_idx+batch_idx)*max_stack_size_per_dpf];
      }
  }

  ////////////////////////////////////////////////
  // Step 2+: Perform DFS Z nodes at a time     //
  ////////////////////////////////////////////////
  
  // Track the depth that we are currently expanding.
  // Assume table size less than 2^32 entries.
  int depth_stack[32];
  int stack_indx = 0;
  int write_out = num_entries-Z;  
  depth_stack[stack_indx] = LOGZ;
  __shared__ uint128_t_gpu cached_write[Z*2];

  // Data for matmul fusion. One per entry. Recall
  // MM (inherited from dpf_benchmark.cu) specifies
  // number of elements per entry.
  uint128_t_gpu per_thread_accumulate[MM] = {0};

  // DFS K nodes at a time
  while (stack_indx >= 0) {

    // Current state of stack
    int cur_depth = depth_stack[stack_indx];
    int cw_read_write_indx_strt = stack_indx*Z;    

    // Sync
    __syncthreads();
    
    if (cur_depth == cw_shared[0].depth) {
      
      // Reached leaf node, pop off the stack      
      stack_indx -= 1;

      // Write to output
      for (int i = 0; i < Z; i += parallel_work_per_threadblock_per_dpf) {
	int expansion_idx = i + (thread_idx % parallel_work_per_threadblock_per_dpf);
	int batch_idx = thread_idx / parallel_work_per_threadblock_per_dpf;

	if (expansion_idx < Z) {

#if(FUSES_MATMUL == 0)
	  // This writes the full expanded DPF output (non-fused) in a permuted ordering
	  out[write_out + expansion_idx + batch_idx*num_entries + block_idx*DPF_HYBRID_DPFS_PER_BLOCK*num_entries] =
	  key_read[cw_read_write_indx_strt + expansion_idx + (block_idx+batch_idx)*max_stack_size_per_dpf];
#else	 
	  // This accumulates, per-thread, partial dot product of table against DPF output
	  uint128_t_gpu dpf_output = key_read[cw_read_write_indx_strt + expansion_idx + (block_idx+batch_idx)*max_stack_size_per_dpf];
	  for (int z = 0; z < MM; z++) {
	    per_thread_accumulate[z] = add_uint128(mul_uint128(dpf_output, TABLE[write_out + expansion_idx]), per_thread_accumulate[z]);
	  }
#endif
	}
      }

      write_out -= Z;
    }
    else {

      //
      // Parallel DPF Expansion procedure
      //
      int start = 0, end = 1 << (LOGZ+1);
      prev_start = start;
      prev_end = end;

      //
      // Main work: Expand K*2 DPFs at a time
      //
      for (int j = start; j < end; j += parallel_work_per_threadblock_per_dpf) {
	int expansion_idx = j + (thread_idx % parallel_work_per_threadblock_per_dpf);
	int batch_idx = thread_idx / parallel_work_per_threadblock_per_dpf;
	
	if (expansion_idx < end) {
	  int idx_into_codewords = expansion_idx % 2;

	  // We could try to load these into shared memory to reduce reads by 1/2.
	  // Tried this and it didn't help. Likely because primarily compute bound.
	  uint128_t_gpu key = key_read[cw_read_write_indx_strt + (expansion_idx/2) + (block_idx+batch_idx)*max_stack_size_per_dpf];
	  uint128_t_gpu value = PRF(key, idx_into_codewords);
	  uint128_t_gpu *cw = (key.x & 1) == 0 ? cw_shared[batch_idx].cw_1 : cw_shared[batch_idx].cw_2;
	  cw = &cw[(cw_shared[0].depth - cur_depth - 1)*2];
	  
	  cached_write[expansion_idx] = add_uint128(value, cw[idx_into_codewords]);
	}
      }

      //
      // Postamble: Sync read and write buffers
      //
      __syncthreads();
      for (int j = prev_start; j < prev_end; j += parallel_work_per_threadblock_per_dpf) {
	int expansion_idx = j + (thread_idx % parallel_work_per_threadblock_per_dpf);
	int batch_idx = thread_idx / parallel_work_per_threadblock_per_dpf;
	
	if (expansion_idx < prev_end) {
	  key_read[cw_read_write_indx_strt + expansion_idx + (block_idx+batch_idx)*max_stack_size_per_dpf] =
	    cached_write[expansion_idx];
	}
      }
      
      
      // Interior node, extend the stack by 2 nodes 
      // (each node representing a group of K DPF outputs)
      depth_stack[stack_indx] = cur_depth+1;
      depth_stack[stack_indx+1] = cur_depth+1;      
      
      stack_indx += 1;
    }
  }

#if(FUSES_MATMUL == 1)
  // Accumulate dot product
  __shared__ uint128_t_gpu shared_accumulates[DPF_HYBRID_THREADS_PER_BLOCK];
  for (int element = 0; element < MM; element++) {

    // Write to shared memory
    shared_accumulates[thread_idx] = per_thread_accumulate[element];
    __syncthreads();

    // Tree sum
    for (int neighbor = 1; neighbor < DPF_HYBRID_THREADS_PER_BLOCK; neighbor*=2) {
      if (thread_idx % (neighbor*2) == 0 && thread_idx+neighbor < DPF_HYBRID_THREADS_PER_BLOCK) {
	shared_accumulates[thread_idx] = add_uint128(shared_accumulates[thread_idx],
						     shared_accumulates[thread_idx+neighbor]);
      }
      __syncthreads();      
    }

    if (thread_idx == 0) {
      out[block_idx + batch_size*element] = shared_accumulates[0];
    }
  }
#endif
}

void dpf_hybrid(SeedsCodewordsFlatGPU * cw,
		uint128_t_gpu *out,
		uint128_t_gpu *TABLE,
		int batch_size, int num_entries,
		cudaStream_t s) {
  dim3 n_blocks(batch_size / DPF_HYBRID_DPFS_PER_BLOCK);
  dim3 n_threads(DPF_HYBRID_THREADS_PER_BLOCK);
  dpf_hybrid_kernel<<<n_blocks, n_threads, 0, s>>>(cw, out,
						DPF_HYBRID_STACK_1,
						DPF_HYBRID_STACK_2,
						TABLE,
						batch_size,
						num_entries);
}
