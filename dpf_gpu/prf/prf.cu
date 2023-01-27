// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#include "../utils.h"
#include "prf_algos/aes_core.h"

#define DUMMY 0
#define SALSA20 1
#define CHACHA20 2
#define AES128 3

#define PRF_METHOD DUMMY

std::string get_PRF_method() {
  if (PRF_METHOD == DUMMY) return "DUMMY";
  if (PRF_METHOD == SALSA20) return "SALSA20";
  if (PRF_METHOD == CHACHA20) return "CHACHA20";
  if (PRF_METHOD == AES128) return "AES128";    
  assert(0);
}

// Ignore warnings since there are unused variables due to
// swapping out included files
#pragma push
#pragma diag_suppress = 253-D
#pragma diag_suppress = 549-D
#pragma diag_suppress = 550-D
#pragma diag_suppress = code_is_unreachable 
#pragma diag_suppress = declared_but_not_referenced

__device__ uint128_t_gpu PRF_DUMMY(uint128_t_gpu seed, uint32_t i) {
  uint128_t_gpu val_4242 = uint128_from(0, 4242);
  uint128_t_gpu val_i = uint128_from(0, i);
  return add_uint128(mul_uint128(seed, add_uint128(val_4242, val_i)),
		     add_uint128(val_4242, val_i));
}


// Salsa20 Source: https://en.wikipedia.org/wiki/Salsa20
#define ROTL(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR(a, b, c, d)(		\
	b ^= ROTL(a + d, 7),	\
	c ^= ROTL(b + a, 9),	\
	d ^= ROTL(c + b,13),	\
	a ^= ROTL(d + c,18))

__device__ uint128_t_gpu salsa20_12_gpu(uint128_t_gpu seed, uint32_t pos) {
  
  // Set up initial state
  uint32_t in[16] = {0};
  uint32_t out[16] = {0};

  // Only use the upper half of 256-bit key
  in[1] = (seed.w) & 0xFFFFFFFF;
  in[2] = (seed.z) & 0xFFFFFFFF;
  in[3] = (seed.y) & 0xFFFFFFFF;
  in[4] = (seed.x) & 0xFFFFFFFF;

  // Set position in stream (pos actual value is 32-bit)
  in[8] = (pos >> 32) & 0xFFFFFFFF;
  in[9] = (pos >> 0) & 0xFFFFFFFF;

  // Rest
  in[0] = 0x65787061;
  in[5] = 0x6e642033;
  in[10] = 0x322d6279;
  in[15] = 0x7465206b;
  
  int i;
  uint32_t x[16];

  for (i = 0; i < 16; ++i)
    x[i] = in[i];
  // 10 loops × 2 rounds/loop = 20 rounds
  for (i = 0; i < 12; i += 2) {
    // Odd round
    QR(x[ 0], x[ 4], x[ 8], x[12]);	// column 1
    QR(x[ 5], x[ 9], x[13], x[ 1]);	// column 2
    QR(x[10], x[14], x[ 2], x[ 6]);	// column 3
    QR(x[15], x[ 3], x[ 7], x[11]);	// column 4
    // Even round
    QR(x[ 0], x[ 1], x[ 2], x[ 3]);	// row 1
    QR(x[ 5], x[ 6], x[ 7], x[ 4]);	// row 2
    QR(x[10], x[11], x[ 8], x[ 9]);	// row 3
    QR(x[15], x[12], x[13], x[14]);	// row 4
  }
  for (i = 0; i < 16; ++i)
    out[i] = x[i] + in[i];

  // Use upper half as result of PRF
  uint128_t_gpu result;
  result.x = out[4];
  result.y = out[3];
  result.z = out[2];
  result.w = out[1];
  return result;  
}

// ChaCha20 Source: https://en.wikipedia.org/wiki/Salsa20
#define ROTL_CHA(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR_CHA(a, b, c, d) (			\
	a += b,  d ^= a,  d = ROTL_CHA(d,16),	\
	c += d,  b ^= c,  b = ROTL_CHA(b,12),	\
	a += b,  d ^= a,  d = ROTL_CHA(d, 8),	\
	c += d,  b ^= c,  b = ROTL_CHA(b, 7))

__device__ uint128_t_gpu chacha20_12_gpu(uint128_t_gpu seed, uint32_t pos) 
{

  // Set up initial state
  uint32_t in[16] = {0};
  uint32_t out[16] = {0};

  // Only use the upper half of 256-bit key
  in[4] = (seed.w) & 0xFFFFFFFF;
  in[5] = (seed.z) & 0xFFFFFFFF;
  in[6] = (seed.y) & 0xFFFFFFFF;
  in[7] = (seed.x) & 0xFFFFFFFF;

  // Set position in stream (pos actual value is 32-bit)
  in[12] = (pos >> 32) & 0xFFFFFFFF;
  in[13] = (pos >> 0) & 0xFFFFFFFF;

  // Rest
  in[0] = 0x65787061;
  in[1] = 0x6e642033;
  in[2] = 0x322d6279;
  in[3] = 0x7465206b;  
  
  int i;
  uint32_t x[16];
  
  for (i = 0; i < 16; ++i)	
    x[i] = in[i];
  // 10 loops × 2 rounds/loop = 20 rounds
  for (i = 0; i < 12; i += 2) {
    // Odd round
    QR_CHA(x[0], x[4], x[ 8], x[12]); // column 0
    QR_CHA(x[1], x[5], x[ 9], x[13]); // column 1
    QR_CHA(x[2], x[6], x[10], x[14]); // column 2
    QR_CHA(x[3], x[7], x[11], x[15]); // column 3
    // Even round
    QR_CHA(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
    QR_CHA(x[1], x[6], x[11], x[12]); // diagonal 2
    QR_CHA(x[2], x[7], x[ 8], x[13]); // diagonal 3
    QR_CHA(x[3], x[4], x[ 9], x[14]); // diagonal 4
  }
  for (i = 0; i < 16; ++i)
    out[i] = x[i] + in[i];

  // Use upper half as result of PRF
  uint128_t_gpu result;
  result.x = out[7];
  result.y = out[6];
  result.z = out[5];
  result.w = out[4];
  return result;  
}

__device__ uint128_t_gpu aes128_gpu(uint128_t_gpu seed, uint32_t pos) {
  unsigned char in[16] = {0};
  unsigned char out[16] = {0};
  const int nr = 10;

  // Input to AES is just the counter (no nonce)
  uint128_t_gpu pos_128 = {0};
  pos_128.x = pos;
  memcpy(in, &pos_128, 16);

  // Key expansion
  AES_KEY k;
  AES_set_encrypt_key((const unsigned char *)&seed,
		      128, &k);
		      

  // AES 128
  AES_encrypt(in, out, &k);

  // Return output
  uint128_t_gpu r = {0};
  memcpy(&r, out, 16);

  return r;

}

template <int prf_method>
__device__ uint128_t_gpu PRF(uint128_t_gpu seed, uint32_t i) {
  if (prf_method == DUMMY) {
    return PRF_DUMMY(seed, i);
  }
  if (prf_method == SALSA20) {
    return salsa20_12_gpu(seed, i);
  }
  if (prf_method == CHACHA20) {
    return chacha20_12_gpu(seed, i);
  }
  if (prf_method == AES128) {
    return aes128_gpu(seed, i);
  }
  assert(0);
}

#pragma pop


