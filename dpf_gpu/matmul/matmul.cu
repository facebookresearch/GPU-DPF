// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#include "../utils.h"

// Define stride K to iterate over
#define BLOCK_TILE_K 16

// Define sizes of blocks of output C to operate over in parallel
#define BLOCK_H 4
#define BLOCK_W 4

// If K is really large, might exceed launch config size restrictions
// Going to hack this to set the right size (TODO: fix)
#define MAX(a,b)	       \
  ({ __typeof__ (a) _a = (a);	\
    __typeof__ (b) _b = (b);	\
    _a > _b ? _a : _b; })
#define BLOCK_K (MAX(128, K/32768))

// Tile inner loop, by outer products with dimension
// K reduction dimension is iterated 1 by 1
#define THREAD_BLOCK_H 1
#define THREAD_BLOCK_W 1

//// Reduction params
#define REDUCTION_THREADS_PER_BLOCK 128

uint128_t_gpu *MATMUL_TABLE_REDUCTION;

void initialize_matmul(int M, int K, int N) {
  // We further initialize global memory for reducing across the K dimension
  assert((K&(K-1)) == 0);
  cudaMalloc(&MATMUL_TABLE_REDUCTION, sizeof(uint128_t_gpu)*M*N*K/BLOCK_K);
  cudaMemset(MATMUL_TABLE_REDUCTION, 0, sizeof(uint128_t_gpu)*M*N*K/BLOCK_K);
}

// Matmul of shape: MxK * KxN -> MxN
__global__ void GEMM128_kernel(uint128_t_gpu *A,
			       uint128_t_gpu *C,
			       uint128_t_gpu *B,
			       uint128_t_gpu *MATMUL_TABLE_REDUCTION,
			       int M, int K, int N) {

  int block_indx_x = blockIdx.x;
  int block_indx_y = blockIdx.y;
  int block_indx_k = blockIdx.z;

  int thread_indx_x = threadIdx.x;
  int thread_indx_y = threadIdx.y;

  int thread_id_within_block = thread_indx_y*BLOCK_W + thread_indx_x;

  // Threads in a block handle block starting from
  int block_C_indx_start = block_indx_y*N*BLOCK_H + block_indx_x*BLOCK_W;

  int threads_per_block = (BLOCK_H/THREAD_BLOCK_H)*(BLOCK_W/THREAD_BLOCK_W);
  int thread_id = thread_indx_y*(BLOCK_W/THREAD_BLOCK_W)+thread_indx_x;    

  __shared__ uint128_t_gpu A_block_local[BLOCK_H][BLOCK_TILE_K+1];
  __shared__ uint128_t_gpu B_block_local[BLOCK_TILE_K][BLOCK_W+1];
  uint128_t_gpu C_frag_local[THREAD_BLOCK_H][THREAD_BLOCK_W] = {0};  
  
  // This is the same as the nvidia post, loop over entire K dimension
  for (int k = block_indx_k*BLOCK_K; k < block_indx_k*BLOCK_K + BLOCK_K; k += BLOCK_TILE_K) {

    // Load blocks of A,B into shared memory in parallel
    int block_A_indx_start = block_indx_y*K*BLOCK_H;
    int block_B_indx_start = block_indx_x*BLOCK_W;

    for (int i = 0; i < BLOCK_H*BLOCK_TILE_K; i+= threads_per_block) {
      int ii = (i+thread_id) / BLOCK_TILE_K;
      int jj = (i+thread_id) % BLOCK_TILE_K;
      A_block_local[ii][jj] = A[k+block_A_indx_start + ii*K + jj];
    }

    for (int i = 0; i < BLOCK_TILE_K*BLOCK_W; i+= threads_per_block) {
      int ii = (i+thread_id) / BLOCK_W;
      int jj = (i+thread_id) % BLOCK_W;
      //B_block_local[ii][jj] = B[block_B_indx_start + k*N + ii*N + jj];
      B_block_local[ii][jj] = B[(block_B_indx_start+jj)*K + (k+ii)];
    }
    
    __syncthreads();

    // Compute over thread block tiles
    for (int i = 0; i < BLOCK_TILE_K; i++) {

      // More efficient method should be outer product
      // Load fragments into registers
      uint128_t_gpu A_frag_local[THREAD_BLOCK_H];
      uint128_t_gpu B_frag_local[THREAD_BLOCK_W];
      
      for (int j = 0; j < THREAD_BLOCK_H; j++) {
	A_frag_local[j] = A_block_local[j+thread_indx_y*THREAD_BLOCK_H][i];
      }
      for (int j = 0; j < THREAD_BLOCK_W; j++) {
	B_frag_local[j] = B_block_local[i][j+thread_indx_x*THREAD_BLOCK_W];
      }

      // Outer product into per-thread mem
      for (int jj = 0; jj < THREAD_BLOCK_H; jj++) {      
	for (int kk = 0; kk < THREAD_BLOCK_W; kk++) {
	  C_frag_local[jj][kk] = add_uint128(C_frag_local[jj][kk],
					     mul_uint128(A_frag_local[jj], B_frag_local[kk]));
	}
      }
    }
  }

  //////////////////////////////////////////////////
  // Reduction across threads in the K dimension //
  /////////////////////////////////////////////////
  
  // Write C frag locals to intermediate output
  int k_stride = M*N;
  for (int j = 0; j < THREAD_BLOCK_W; j++) {    
    for (int i = 0; i < THREAD_BLOCK_H; i++) {  
      MATMUL_TABLE_REDUCTION[block_indx_k*k_stride + block_C_indx_start + thread_indx_y*THREAD_BLOCK_H*N + thread_indx_x*THREAD_BLOCK_W + i*N + j] = C_frag_local[i][j];
    }
  }
}

__global__ void GEMM128_reduction_kernel(uint128_t_gpu *MATMUL_TABLE_REDUCTION,
					 uint128_t_gpu *out,
					 int M, int K, int N) {
  int block_indx = blockIdx.x;
  int thread_idx = threadIdx.x;
  int work_per_block = REDUCTION_THREADS_PER_BLOCK;  
  int work_indx = block_indx*work_per_block + thread_idx;

  if (work_indx >= M*N) return;
  
  int k_stride = M*N;
  uint128_t_gpu accum[1] = {0};
  for (int k = 0; k < K/BLOCK_K; k++) {
    uint128_t_gpu op2 = MATMUL_TABLE_REDUCTION[k*k_stride + work_indx];
    accum[0] = add_uint128(accum[0], op2);
  }

  out[work_indx] = accum[0];
}

void GEMM128(uint128_t_gpu *A,
	     uint128_t_gpu *C,
	     uint128_t_gpu *B,
	     int M, int K, int N,
	     cudaStream_t s) {
  
  assert(BLOCK_W%THREAD_BLOCK_W == 0);
  assert(BLOCK_H%THREAD_BLOCK_H == 0);
  assert(N%BLOCK_W == 0);
  assert(M%BLOCK_H == 0);
  
  dim3 threads_per_block(BLOCK_W/THREAD_BLOCK_W, BLOCK_H/THREAD_BLOCK_H);
  dim3 n_blocks(N/BLOCK_W, M/BLOCK_H, K/BLOCK_K);

  GEMM128_kernel<<<n_blocks, threads_per_block, 0, s>>>(A, C, B, MATMUL_TABLE_REDUCTION, M, K, N);

  dim3 threads_per_block_reduce(REDUCTION_THREADS_PER_BLOCK);
  dim3 n_blocks_reduce((M*N)/REDUCTION_THREADS_PER_BLOCK+1);
  GEMM128_reduction_kernel<<<n_blocks_reduce, threads_per_block_reduce, 0, s>>>(MATMUL_TABLE_REDUCTION, C, M, K, N);
}
