// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#ifndef DPF_BASELINE
#define DPF_BASELINE 1

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <assert.h>
#include <random>
#include <memory>
#include <vector>
#include "aes_core.h"

typedef unsigned __int128 uint128_t;

struct SeedsCodewordsFlat {
  // Bare minimal construction of SeedsCodewords for
  // log(n) construction of DPF for evaluation for one server
  // Supports up to 2^32 size

  // Keys per level of the DPF
  int depth;

  // Concatenation of all codewords for each level
  uint128_t cw_1[64], cw_2[64];
  uint128_t last_keys[1];
};

// Generated by the third party for the two servers.
// Captures the seeds and correction codewords for a single
// "level" of the DPF computation.
//
// May need multiple when recursing on the seeds
// (which themselves may be computed using DPF).
struct SeedsCodewords {

  int recurse;
  
  int alpha, beta;
  
  // Can view n_keys as the # of cols, n_codewords as # of rows
  // Note that a key is the concatenation of seeds and codewrod selectors
  int n_codewords, n_keys;

  std::vector<uint128_t> codewords_1, codewords_2;

  // k1, and k2 represent _keys_, where each key
  // represents concatenation of _seed_ and codeword selector.
  // k1, and k2 are the same except at _one_ indx, and hence can be
  // recursively recreated (this saves mem).
  // k1 and k2 are NULL at a non-leaf level, non-NULL at leaf node.
  std::vector<uint128_t> k1, k2;
  SeedsCodewords *sub;
  

  // Stores grid point (within a n_codewords x n_seeds matrix) where beta is
  // i.row indx, j.col indx, as always
  // j . index into keys
  // i . index into codewords
  int i,j;
};

/********
 * PRFs *
 ********/

// These must match exactly w/ GPU version

// Dummy encryption function
uint128_t K(uint128_t seed, uint128_t i) {
  return seed * (i+4242) + (i+4242);
}

// Salsa20 Source: https://en.wikipedia.org/wiki/Salsa20
#define ROTL(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR(a, b, c, d)(		\
	b ^= ROTL(a + d, 7),	\
	c ^= ROTL(b + a, 9),	\
	d ^= ROTL(c + b,13),	\
	a ^= ROTL(d + c,18))

uint128_t salsa20_12(uint128_t seed, uint128_t pos) 
{

  // Set up initial state
  uint32_t in[16] = {0};
  uint32_t out[16] = {0};

  // Only use the upper half of 256-bit key
  in[1] = (seed >> 96) & 0xFFFFFFFF;
  in[2] = (seed >> 64) & 0xFFFFFFFF;
  in[3] = (seed >> 32) & 0xFFFFFFFF;
  in[4] = (seed >> 0) & 0xFFFFFFFF;

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
  uint128_t result = 0;
  result = ((uint128_t)out[1] << 96) +
    ((uint128_t)out[2] << 64) +
    ((uint128_t)out[3] << 32) +
    ((uint128_t)out[4] << 0);
  return result;
}

// ChaCha20 Source: https://en.wikipedia.org/wiki/Salsa20
#define ROTL_CHA(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR_CHA(a, b, c, d) (			\
	a += b,  d ^= a,  d = ROTL_CHA(d,16),	\
	c += d,  b ^= c,  b = ROTL_CHA(b,12),	\
	a += b,  d ^= a,  d = ROTL_CHA(d, 8),	\
	c += d,  b ^= c,  b = ROTL_CHA(b, 7))

uint128_t chacha20_12(uint128_t seed, uint128_t pos) 
{

  // Set up initial state
  uint32_t in[16] = {0};
  uint32_t out[16] = {0};

  // Only use the upper half of 256-bit key
  in[4] = (seed >> 96) & 0xFFFFFFFF;
  in[5] = (seed >> 64) & 0xFFFFFFFF;
  in[6] = (seed >> 32) & 0xFFFFFFFF;
  in[7] = (seed >> 0) & 0xFFFFFFFF;

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
  uint128_t result = 0;
  result = ((uint128_t)out[4] << 96) +
    ((uint128_t)out[5] << 64) +
    ((uint128_t)out[6] << 32) +
    ((uint128_t)out[7] << 0);
  return result;  
}

uint128_t aes128(uint128_t seed, uint128_t pos) {
  unsigned char in[16] = {0};
  unsigned char out[16] = {0};
  const int nr = 10;

  // Input to AES is just the counter (no nonce)
  memcpy(in, &pos, 16);

  // Expand the seed (key) into key schedule
  u64 w[22] = {0};
  KeyExpansion((const unsigned char *)&seed,
	       w, nr, 4);
  
  // AES 128
  Cipher(in, out, w, nr);

  // Return output
  uint128_t r = 0;
  memcpy(&r, out, 16);

  return r;
}

#define DUMMY 0
#define SALSA 1
#define CHACHA 2
#define AES128 3

// Glob method for selecting PRF
uint128_t (*PRF_SELECT(int prf_method))(uint128_t, uint128_t) {
  uint128_t (*PRF)(uint128_t, uint128_t);
  if (prf_method == DUMMY) PRF = K;
  else if (prf_method == SALSA) PRF = salsa20_12;
  else if (prf_method == CHACHA) PRF = chacha20_12;
  else if (prf_method == AES128) PRF = aes128;
  else assert(0);
  return PRF;
}

/*********/

void FlattenCodewords(SeedsCodewords *s, int srv, SeedsCodewordsFlat *f) {
  int total_n_cws = 0, total_n_keys = 0;
  SeedsCodewords *cur = s;
  while (cur != NULL) {
    total_n_cws += cur->codewords_1.size();
    total_n_keys += 1;
    cur = cur->sub;
  }

  f->depth = total_n_keys;

  cur = s;
  int indx_key = 0, indx_cw = 0;
  while (cur != NULL) {
    
    for (int i = 0; i < cur->codewords_1.size(); i++) {
      f->cw_1[indx_cw+i] = cur->codewords_1[i];
      f->cw_2[indx_cw+i] = cur->codewords_2[i];      
    }

    if (cur->k1.size() != 0) {
      std::vector<uint128_t> &k = srv ? cur->k2 : cur->k1;
      for (int i = 0; i < k.size(); i++) {
	f->last_keys[i] = k[i];
      }
    }
    
    indx_key += 1;
    indx_cw += cur->codewords_1.size();
    cur = cur->sub;
  }
}

uint128_t GenerateRandomNumber(std::mt19937 &gen) {
  std::uniform_int_distribution<uint64_t> d(0, std::numeric_limits<uint64_t>::max());
  uint64_t l = d(gen);
  uint64_t r = d(gen);
  return ((uint128_t)l) << 64 | (uint128_t)r;
}

uint128_t GenerateRandomOddNumber(std::mt19937 &gen) {
  uint128_t k = 0;
  while (k % 2 == 0) k = GenerateRandomNumber(gen);
  return k;
}

// Called from the perspective of the generator (not the two servers)
// DPF with parameters
// alpha, beta - such that, v_1[alpha] - v_2[alpha] = beta
// 0 otherwise
// N - size of v of the space of DPF
SeedsCodewords *GenerateSeedsAndCodewords(int alpha, uint128_t beta, int N, 
					  int n_keys, int n_codewords,
					  std::mt19937 &g_gen, int prf_method) {
  assert(alpha < N);

  assert(n_keys * n_codewords == N);

  // Generate seeds and codewords
  SeedsCodewords *s = new SeedsCodewords;
  s->recurse = 0;
  s->sub = NULL;
  s->n_keys = n_keys;
  s->n_codewords = n_codewords;
  s->alpha = alpha;
  s->beta = beta;
  s->j = alpha % s->n_keys;
  s->i = alpha / s->n_keys;
  assert(s->i*s->n_keys + s->j == alpha);

  // Alloc codeword data
  s->codewords_1.resize(s->n_codewords);
  s->codewords_2.resize(s->n_codewords);  

  s->k1.resize(s->n_keys);
  s->k2.resize(s->n_keys);  
  for (int i = 0; i < s->n_keys; i++) {

    if (i == s->j) {
      // At the column index that should differ
      uint128_t key_for_first_server = GenerateRandomNumber(g_gen);
      uint128_t key_for_second_server = GenerateRandomNumber(g_gen);

      uint128_t mask = 0xFFFFFFFFFFFFFFFF;
      mask <<= 64;
      mask |= 0xFFFFFFFFFFFFFFFF;

      // Second server always uses codeword 2 at the index
      key_for_first_server &= (mask-1);
      key_for_second_server &= (mask-1);
      key_for_second_server |= 1;
      
      s->k1[i] = key_for_first_server;
      s->k2[i] = key_for_second_server;
    }
    else {
      // At index that should be same
      s->k1[i] = s->k2[i] = GenerateRandomNumber(g_gen);
    }
  }

  // Generate correction codewords based on prev.
  // Note that, assumes, at target index, server 1 uses CW1,
  // and server 2 uses CW2
  uint128_t seed_diffs[s->n_codewords];
  uint128_t s1 = s->k1[s->j], s2 = s->k2[s->j];
  assert(s1 != s2);
  for (int i = 0; i < s->n_codewords; i++) {
    seed_diffs[i] = (PRF_SELECT(prf_method)(s1, i) - PRF_SELECT(prf_method)(s2, i));
    if (i == s->i) {
      // Offset by beta at the target row index
      seed_diffs[i] -= beta;
    }
  }

  for (int i = 0; i < s->n_codewords; i++) {
    s->codewords_1[i] = GenerateRandomNumber(g_gen);
    s->codewords_2[i] = s->codewords_1[i] + seed_diffs[i];
  }

  return s;
}

uint128_t EvaluateFlat(const SeedsCodewordsFlat *s, int indx, int prf_method) {


  int indx_remaining = indx;  
  uint128_t key = s->last_keys[0];
  uint128_t value;
  for (int i = s->depth-1; i >= 0; i--) {
    int indx_into_codewords = indx_remaining % 2;
    value = PRF_SELECT(prf_method)(key, indx_into_codewords);
    const uint128_t *cw = (key & 1) == 0 ? s->cw_1 : s->cw_2;
    cw = &cw[i*2];
    key = value + cw[indx_into_codewords];
    indx_remaining >>= 1;
  }
  return key;
}

uint128_t Evaluate(const SeedsCodewords *s, int indx, int srv_sel, int prf_method) {
  // Evaluate DPF parameterized by s at index indx
  // srv_sel is 0 or 1 depending on if caller is server 0 or 1
  
  int indx_into_keys = indx % s->n_keys;
  int indx_into_codewords = indx / s->n_keys;

  uint128_t key, value;
  if (!s->recurse) {
    const std::vector<uint128_t> &keys = srv_sel == 0 ? s->k1 : s->k2;  
    key = keys[indx_into_keys];
  }
  else {
    key = Evaluate(s->sub, indx_into_keys, srv_sel, prf_method);
  }
  value = PRF_SELECT(prf_method)(key, indx_into_codewords); 

  const std::vector<uint128_t> &cw = (key & 1) == 0 ? s->codewords_1 : s->codewords_2;
  return value + cw[indx_into_codewords];
}

/////////////////////////////////////////

// Log(n) construction of the DPF
SeedsCodewords *GenerateSeedsAndCodewordsLog(int alpha, uint128_t beta, int N, std::mt19937 &g_gen, int prf_method) {
  assert((N & (N-1)) == 0);
  assert(alpha < N);

  if (N == 2) {
    // Base case -- standard DPF construction
    return GenerateSeedsAndCodewords(alpha, beta, N, 1, 2, g_gen, prf_method);
  }

  // Recursively generate seeds and codewords for N/2
  // alpha/2 is the target column (index into keys)
  // alpha%2 is the target row (index into codewords)
  uint128_t beta_new = GenerateRandomOddNumber(g_gen);  
  SeedsCodewords *sub = GenerateSeedsAndCodewordsLog(alpha%(N/2), beta_new, N/2, g_gen, prf_method);

  // Construct
  SeedsCodewords *s = new SeedsCodewords;
  s->sub = sub;
  s->recurse = 1;
  s->n_keys = N/2;
  s->n_codewords = 2;
  s->alpha = alpha;
  s->beta = beta;
  s->j = alpha % s->n_keys;
  s->i = alpha / s->n_keys;
  
  assert(s->i*s->n_keys + s->j == alpha);

  // Alloc codewords data
  s->codewords_1.resize(s->n_codewords);
  s->codewords_2.resize(s->n_codewords);

  // To construct codewords for this level, get the value of
  // the differing generated seed
  uint128_t s1 = Evaluate(s->sub, alpha%(N/2), 0, prf_method);
  uint128_t s2 = Evaluate(s->sub, alpha%(N/2), 1, prf_method);

  assert(s1-s2 == beta_new);
  assert(s1%2 != s2%2);

  for (int i = 0; i < s->n_codewords; i++) {
    // Value for first and second server at i'th codeword
    uint128_t first_val = PRF_SELECT(prf_method)(s1, i);
    uint128_t second_val = PRF_SELECT(prf_method)(s2, i);
    uint128_t diff = second_val-first_val;
    diff = (s1%2 == 0) ? -diff : diff;

    s->codewords_1[i] = g_gen();
    s->codewords_2[i] = s->codewords_1[i]+diff;

    if (i == alpha/(N/2)) {
      if (s1 % 2 == 0) {
	s->codewords_1[i] += beta;
      }
      else {
	s->codewords_1[i] -= beta;
      }
    }
  }

  return s;
}

void FreeSeedsCodewords(SeedsCodewords *s) {
  if (s->sub != NULL)
    free(s->sub);
  free(s);
}

// Compute size of seeds/codewords with just 1 of the keys
// assuming the other is zeroed out for communication to juse 1 server
size_t SizeOf(const SeedsCodewords *s) {
  size_t rest = 0;
  if (s->sub != NULL) rest = SizeOf(s->sub);
  size_t s1 =  sizeof(s);
  size_t s2 = (s->codewords_1.size() + s->codewords_2.size() + s->k1.size()) *
    sizeof(s->k1[0]);
  return s1 + s2 + rest;
}

void test_sqrt_n_method() {
  int alpha = 2;
  int beta = 210;

  int n_seeds = 1024;
  int n_codewords = 1024;
  int N = n_seeds*n_codewords;
  std::mt19937 g_gen(0);  
  SeedsCodewords *s = GenerateSeedsAndCodewords(alpha, beta, N, n_seeds, n_codewords, g_gen, 0);
  
  for (int i = 0; i < N; i++) {
    int expected = i == alpha ? beta : 0;
    assert(Evaluate(s, i, 0, 0) - Evaluate(s, i, 1, 0) == expected);
  }
  std::cout << "SQRT(N) METHOD:" << std::endl;
  std::cout << "Pass checks (N=" << N << ")" << std::endl;
  std::cout << "Number of entries in table: " << N << " sizeof communication " << SizeOf(s) << " bytes"  << std::endl;

  FreeSeedsCodewords(s);
}

void test_log_n_method() {
  int N = 1048576;
  int alpha = 1000;
  int beta = 210;
  std::mt19937 g_gen(0);  
  
  SeedsCodewords *s = GenerateSeedsAndCodewordsLog(alpha, beta, N, g_gen, 0);
  
  for (int i = 0; i < N; i++) {
    int expected = i == alpha ? beta : 0;
    assert(Evaluate(s, i, 0, 0) - Evaluate(s, i, 1, 0) == expected);
  }
  std::cout << "LOG(N) METHOD:" << std::endl;  
  std::cout << "Pass checks (N=" << N << ")" << std::endl;
  std::cout << "Number of entries in table: " << N << " sizeof communication " << SizeOf(s) << " bytes"  << std::endl;

  FreeSeedsCodewords(s);  
}

uint64_t get_time_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void benchmark_log_n_method_perf() {
  int N = 16384;
  int alpha = 1000;
  int beta = 210;
  std::mt19937 g_gen(0);  
  
  SeedsCodewords *s = GenerateSeedsAndCodewordsLog(alpha, beta, N, g_gen, 0);

  uint64_t t1 = get_time_ms();
  uint128_t *results = (uint128_t *)malloc(sizeof(uint128_t) * 128*N);
  for (int j = 0; j < 128; j++) {
    for (int i = 0; i < N; i++) {
      results[j*N+i] = Evaluate(s, i, 0, 0);
    }
  }
  uint64_t t2 = get_time_ms();
  uint64_t elapsed = t2-t1;

  float throughput_ms = 128 / (float)elapsed;
  std::cout << "Throughput ms: " << throughput_ms << std::endl;

  free(results);
  FreeSeedsCodewords(s);  
}

void test_flat_codewords() {
  int N = 16384;
  int alpha = 142;
  int beta = 210;
  std::mt19937 g_gen(0);  
  
  SeedsCodewords *s = GenerateSeedsAndCodewordsLog(alpha, beta, N, g_gen, 0);
  SeedsCodewordsFlat *sf_1 = new SeedsCodewordsFlat;
  SeedsCodewordsFlat *sf_2 = new SeedsCodewordsFlat;    
  FlattenCodewords(s, 0, sf_1);
  FlattenCodewords(s, 1, sf_2);

  for (int i = 0; i < N; i++) {
    uint128_t a = EvaluateFlat(sf_1, i, 0);
    uint128_t b = EvaluateFlat(sf_2, i, 0);
    uint128_t correct = i==alpha ? beta : 0;

    assert(a-b == correct);
  }
  
  std::cout << "PASS" << std::endl;
  
  FreeSeedsCodewords(s);
  free(sf_1);
  free(sf_2);
}

#endif
