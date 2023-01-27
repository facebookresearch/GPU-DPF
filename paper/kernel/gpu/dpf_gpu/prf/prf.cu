// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#include "../utils.h"

#define DUMMY 0

// These hash methods are from: https://github.com/mochimodev/cuda-hashing-algos
//
// One thing to note is that these operate over "batches" of inputs.
// But as it is, we're calling it one function at a time on DPF codewords (non-batch)
// which could be why it's showing extremely poor performance (besides MD5)
//
// These are don't work well.
#define MD5_HASH_SLOW 1
#define BLAKE2B_HASH_SLOW 2
#define KECCAK_HASH_SLOW 3
#define MD2_HASH_SLOW 4
#define SHA256_HASH_SLOW 5
///////////////////////

// Hash functions
#define MD5_FASTER 6
#define SHA256_FASTER 11

// Pure PRFs
#define SIPHASH 12
#define HIGHWAYHASH 13

// Ciphers (stream + block)
#define SALSA20_CIPHER 7
#define SALSA20_12_CIPHER 8
#define SALSA20_8_CIPHER 9
#define AES_128_CTR_CIPHER 10

#ifndef PRF_METHOD
#define PRF_METHOD DUMMY
//#define PRF_METHOD HIGHWAYHASH
//#define PRF_METHOD SIPHASH
//#define PRF_METHOD SHA256_FASTER
//#define PRF_METHOD AES_128_CTR_CIPHER
//#define PRF_METHOD SALSA20_CIPHER
//#define PRF_METHOD SALSA20_8_CIPHER
//#define PRF_METHOD SALSA20_12_CIPHER
//#define PRF_METHOD MD5_FASTER
//#define PRF_METHOD MD2_HASH_SLOW
//#define PRF_METHOD KECCAK_HASH_SLOW
//#define PRF_METHOD BLAKE2B_HASH_SLOW
//#define PRF_METHOD MD5_HASH_SLOW
//#define PRF_METHOD SHA256_HASH_SLOW
#endif

#if(PRF_METHOD == MD5_HASH_SLOW)
#include "prf_algos/md5_mochi.cu"
#endif

#if(PRF_METHOD == BLAKE2B_HASH_SLOW)
#include "prf_algos/blake2b_mochi.cu"
#endif

#if(PRF_METHOD == KECCAK_HASH_SLOW)
#include "prf_algos/keccak_mochi.cu"
#endif

#if(PRF_METHOD == MD2_HASH_SLOW)
#include "prf_algos/md2_mochi.cu"
#endif

#if(PRF_METHOD == SHA256_HASH_SLOW)
#include "prf_algos/sha256_mochi.cu"
#endif

#if(PRF_METHOD == MD5_FASTER)
#include "prf_algos/md5.cu"
#endif

#if(PRF_METHOD == SALSA20_CIPHER || PRF_METHOD == SALSA20_8_CIPHER || PRF_METHOD == SALSA20_12_CIPHER)
#include "prf_algos/salsa20.cu"
#endif

#if(PRF_METHOD == AES_128_CTR_CIPHER)
#include "prf_algos/aes_cuda.cu"
#endif

#if(PRF_METHOD == SHA256_FASTER)
#include "prf_algos/sha256.cu"
#endif

#if(PRF_METHOD == SIPHASH)
#include "prf_algos/siphash.cu"
#endif

#if(PRF_METHOD == HIGHWAYHASH)
#include "prf_algos/highwayhash.cu"
#endif

std::string get_PRF_method() {
  if (PRF_METHOD == DUMMY) return "DUMMY";
  if (PRF_METHOD == HIGHWAYHASH) return "HIGHWAYHASH";
  if (PRF_METHOD == SIPHASH) return "SIPHASH";
  if (PRF_METHOD == MD5_FASTER) return "MD5";
  if (PRF_METHOD == SHA256_FASTER) return "SHA256";
  if (PRF_METHOD == SALSA20_8_CIPHER) return "SALSA20_8";
  if (PRF_METHOD == SALSA20_12_CIPHER) return "SALSA20_12";
  if (PRF_METHOD == SALSA20_CIPHER) return "SHA25620_20";
  if (PRF_METHOD == AES_128_CTR_CIPHER) return "AES128";  
}

// Ignore warnings since there are unused variables due to
// swapping out included files
#pragma push
#pragma diag_suppress = 253-D
#pragma diag_suppress = 549-D
#pragma diag_suppress = 550-D
#pragma diag_suppress = code_is_unreachable 
#pragma diag_suppress = declared_but_not_referenced

__device__ void hash(uint64_t *out_data, uint64_t *in_data) {
  
  unsigned char *in = (unsigned char *)in_data;
  unsigned char *out = (unsigned char *)out_data;
  
#if(PRF_METHOD == MD5_HASH_SLOW)
  CUDA_MD5_CTX ctx;
  cuda_md5_init(&ctx);      
  cuda_md5_update(&ctx, in, 16);
  cuda_md5_final(&ctx, out);
#endif
  
#if(PRF_METHOD == BLAKE2B_HASH_SLOW)
  CUDA_BLAKE2B_CTX ctx = c_CTX;
  // IMPORTANT: c_CTX is not initialized, but couldn't get it to work
  // with initialization (seg fault)
  cuda_blake2b_update(&ctx, in, 16);
  cuda_blake2b_final(&ctx, out);
#endif
  
#if(PRF_METHOD == KECCAK_HASH_SLOW)
  CUDA_KECCAK_CTX ctx;
  cuda_keccak_init(&ctx, 128);
  cuda_keccak_update(&ctx, in, 16);
  cuda_keccak_final(&ctx, out);    
#endif
  
#if(PRF_METHOD == MD2_HASH_SLOW)
  CUDA_MD2_CTX ctx;
  cuda_md2_init(&ctx);
  cuda_md2_update(&ctx, in, 16);
  cuda_md2_final(&ctx, out);
#endif
  
#if(PRF_METHOD == SHA256_HASH_SLOW)
  CUDA_SHA256_CTX ctx;
  cuda_sha256_init(&ctx);
  cuda_sha256_update(&ctx, in, 16);
  cuda_sha256_final(&ctx, out);
#endif
  
#if(PRF_METHOD == MD5_FASTER)
  md5Hash(in, 16, ((uint32_t *)out) + 0, ((uint32_t *)out) + 1, ((uint32_t *)out) + 2, ((uint32_t *)out) + 3);
#endif

#if(PRF_METHOD == SHA256_FASTER)
  uint32_t out_larger[8];
  sha256(out_larger,
	 ((uint32_t *)in)[0],
	 ((uint32_t *)in)[1],
	 ((uint32_t *)in)[2],
	 ((uint32_t *)in)[3],
	 0, 0, 0, 0,
	 0, 0, 0, 0,
	 0, 0, 0, 0);
  ((uint32_t *)out)[0] = out_larger[0];
  ((uint32_t *)out)[1] = out_larger[1];
  ((uint32_t *)out)[2] = out_larger[2];
  ((uint32_t *)out)[3] = out_larger[3];  
#endif
}

__device__ uint128_t_gpu HMAC(uint128_t_gpu seed, int i) {
  // PERFORMS HMAC
  uint64_t in[2];
  uint64_t out[2];

  uint64_t seed_first = (((uint64_t)seed.x) << 32) | seed.y;
  uint64_t seed_second = (((uint64_t)seed.z) << 32) | seed.w;
  
  in[0] = (seed_first ^ 0x3636363636363636) | i;
  in[1] = (seed_second ^ 0x3636363636363636) | i;
  
  hash(out, in);

  in[0] = out[0] | (seed_first ^ 0x5c5c5c5c5c5c5c5c);
  in[1] = out[1] | (seed_second ^ 0x5c5c5c5c5c5c5c5c);

  hash(out, in);

  uint128_t_gpu r;
  r.x = out[0] >> 32;
  r.y = out[0] & 0xFFFFFFFF;
  r.z = out[1] >> 32;
  r.w = out[1] & 0xFFFFFFFF;
  return r;
}

__device__ uint128_t_gpu PRF(uint128_t_gpu seed, uint32_t i) {

  if (PRF_METHOD == DUMMY) {
    uint128_t_gpu val_4242 = uint128_from(0, 4242);
    uint128_t_gpu val_i = uint128_from(0, i);
    return add_uint128(mul_uint128(seed, add_uint128(val_4242, val_i)),
		       add_uint128(val_4242, val_i));
  }

  // Check if stream cipher (otherwise uses hash + HMAC)
  if (PRF_METHOD == SALSA20_CIPHER ||
      PRF_METHOD == SALSA20_8_CIPHER ||
      PRF_METHOD == SALSA20_12_CIPHER) {
    
    // Set initial state
    uint32_t in[16];
    uint32_t out[16];
    
    // Fixed words
    in[0] = 0x65787061;
    in[5] = 0x6e642033;
    in[10] = 0x322d6279;
    in[15] = 0x7465206b;

    // Nonce (this is fixed as well)
    in[6] = 0;
    in[7] = 0;

    // CTR
    in[8] = i;
    in[9] = 0;

    // Keys
    in[1] = seed.x;
    in[2] = seed.y;
    in[3] = seed.z;
    in[4] = seed.w;
    
    in[11] = in[12] = in[13] = in[14] = 0;

#if(PRF_METHOD == SALSA20_CIPHER)
      SALSA20(out, in);
#endif
#if(PRF_METHOD == SALSA20_8_CIPHER)      
      SALSA20_8(out, in);
#endif
#if(PRF_METHOD == SALSA20_12_CIPHER)      
      SALSA20_12(out, in);
#endif
      uint128_t_gpu r;
      r.x = out[0];
      r.y = out[1];
      r.w = out[2];
      r.z = out[3];
      return r;
  }
  else if (PRF_METHOD == AES_128_CTR_CIPHER) {
    // Note that, this method does not do key expansion
    // and assumes it is already done. Performance is
    // poor enough and key expansion makes it worse
    
    uint32_t block[4];
    for (int ii = 0; ii < 44; ii++) block[i] = i;
    uint32_t key[44];
    for (int ii = 0; ii < 44; ii++) key[i] = seed.x;

#if(PRF_METHOD == AES_128_CTR_CIPHER)
    encrypt((uint8_t *)block, (uint8_t *)key, 0);
#endif

    uint128_t_gpu r;
    r.x = block[0];
    r.y = block[1];
    r.w = block[2];
    r.z = block[3];
    return r;
  }
  else if (PRF_METHOD == SIPHASH) {
    // Note that siphash isn't really a hash function but instead
    // is directly a PRF, so we can call it directly
    // TODO: wikipedia says 64-bit output. CONVERT TO 128 bit!
    uint8_t out[16];
    uint32_t in[4] = {i, 0, 0, 0};
    uint32_t k[4] = {seed.x, seed.y, seed.z, seed.w};

#if(PRF_METHOD == SIPHASH)    
    siphash(out, (const uint8_t *)in, 16, (const uint8_t *)k);
#endif

    uint128_t_gpu r;
    r.x = out[0];
    r.y = out[1];

    // TODO: Note these two should be out[2], out[3] for 128-bit security
    r.w = out[0];
    r.y = out[1];
    return r;
  }
  else if (PRF_METHOD == HIGHWAYHASH) {
    // Note there are criticisms of highway hash as cryptographically
    // secure hash. But authors claim it is a PRF.
    uint32_t data[4] = {i, 0, 0, 0};
    size_t size = 16;
    uint64_t key[4] = {seed.x, seed.y, seed.z, seed.w};
    uint64_t hash[2];
    
#if(PRF_METHOD == HIGHWAYHASH)
    HighwayHash128((uint8_t *)data, size, key, hash);
#endif

    uint128_t_gpu r;
    r.x = hash[0] >> 32;
    r.y = hash[0] & 0xFFFFFFFF;
    r.w = hash[1] >> 32;
    r.y = hash[1] & 0xFFFFFFFF;
    
    return r;
  }
  else {
    return HMAC(seed, i);
  }
}
#pragma pop


