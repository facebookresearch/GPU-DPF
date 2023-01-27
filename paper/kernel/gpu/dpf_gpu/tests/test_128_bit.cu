// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


#include "../utils.h"

void test_uint128_gpu_from() {
  uint128_t val = 0x1122334422334455;
  val <<= 64;
  val |= 0x3344556644556677;
  
  uint128_t_gpu g1 = uint128_gpu_from(val);
  assert(g1.w == 0x11223344);
  assert(g1.z == 0x22334455);
  assert(g1.y == 0x33445566);
  assert(g1.x == 0x44556677);
}

__global__ void test_uint128_from_kernel(uint128_t_gpu *r) {
  *r = uint128_from(0x1234567823456789,
		    0x2345678934567890);
}

void test_uint128_from() {
  uint128_t_gpu *r;
  cudaMalloc((void **)&r, sizeof(uint128_t_gpu));
  test_uint128_from_kernel<<<1, 1>>>(r);
  uint128_t_gpu r_cpu;
  cudaMemcpy(&r_cpu, r, sizeof(uint128_t_gpu), cudaMemcpyDeviceToHost);
  
  assert(r_cpu.w == 0x12345678);
  assert(r_cpu.z == 0x23456789);
  assert(r_cpu.y == 0x23456789);
  assert(r_cpu.x == 0x34567890);      
  
  cudaFree(r);
}

__global__ void test_add_uint128_kernel(uint128_t_gpu *a,
					uint128_t_gpu *b,
					uint128_t_gpu *r) {
  *r = add_uint128(*a, *b);
}

void test_add_uint128() {

  // Init v1 and v2 for mult
  uint128_t v1 = 0x12345678;
  v1 <<= 64;
  v1 |= 0x23456789;

  uint128_t v2 = 0x34567890;
  v2 <<= 64;
  v2 |= 0x45678901;

  uint128_t_gpu a = uint128_gpu_from(v1);
  uint128_t_gpu b = uint128_gpu_from(v2);      

  // Alloc gpu mem
  uint128_t_gpu *r;
  cudaMalloc((void **)&r, sizeof(uint128_t_gpu));

  uint128_t_gpu *a_gpu, *b_gpu;
  cudaMalloc((void **)&a_gpu, sizeof(uint128_t_gpu));
  cudaMalloc((void **)&b_gpu, sizeof(uint128_t_gpu));
  cudaMemcpy(a_gpu, &a, sizeof(uint128_t_gpu), cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, &b, sizeof(uint128_t_gpu), cudaMemcpyHostToDevice);  
  
  test_add_uint128_kernel<<<1, 1>>>(a_gpu, b_gpu, r);
  uint128_t_gpu r_cpu;
  cudaMemcpy(&r_cpu, r, sizeof(uint128_t_gpu), cudaMemcpyDeviceToHost);
  
  uint128_t truth = v1+v2;
  assert(r_cpu.x == (truth & 0xFFFFFFFF));
  assert(r_cpu.y == ((truth & 0xFFFFFFFF00000000) >> 32));
  assert(r_cpu.w == truth >> 96);
  assert(r_cpu.z == ((truth >> 64) & 0xFFFFFFFF));  
  
  cudaFree(r);
  cudaFree(a_gpu);
  cudaFree(b_gpu);    
}

__global__ void test_mul_uint128_kernel(uint128_t_gpu *a,
					uint128_t_gpu *b,
					uint128_t_gpu *r) {
  *r = mul_uint128(*a, *b);
}

void test_mul_uint128() {
  
  // Init v1 and v2 for mult
  uint128_t v1 = 0x12345678;
  v1 <<= 64;
  v1 |= 0x23456789;

  uint128_t v2 = 0x34567890;
  v2 <<= 64;
  v2 |= 0x45678901;

  uint128_t_gpu a = uint128_gpu_from(v1);
  uint128_t_gpu b = uint128_gpu_from(v2);      

  // Alloc gpu mem
  uint128_t_gpu *r;
  cudaMalloc((void **)&r, sizeof(uint128_t_gpu));

  uint128_t_gpu *a_gpu, *b_gpu;
  cudaMalloc((void **)&a_gpu, sizeof(uint128_t_gpu));
  cudaMalloc((void **)&b_gpu, sizeof(uint128_t_gpu));
  cudaMemcpy(a_gpu, &a, sizeof(uint128_t_gpu), cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, &b, sizeof(uint128_t_gpu), cudaMemcpyHostToDevice);  
  
  test_mul_uint128_kernel<<<1, 1>>>(a_gpu, b_gpu, r);
  uint128_t_gpu r_cpu;
  cudaMemcpy(&r_cpu, r, sizeof(uint128_t_gpu), cudaMemcpyDeviceToHost);
  
  uint128_t truth = v1*v2;
  
  assert(r_cpu.x == (truth & 0xFFFFFFFF));
  assert(r_cpu.y == ((truth & 0xFFFFFFFF00000000) >> 32));
  assert(r_cpu.w == truth >> 96);
  assert(r_cpu.z == ((truth >> 64) & 0xFFFFFFFF));  
  
  cudaFree(r);
  cudaFree(a_gpu);
  cudaFree(b_gpu); 
}

__global__ void test_mul_uint128_kernel_twice(uint128_t_gpu *a,
					      uint128_t_gpu *b,
					      uint128_t_gpu *c,
					      uint128_t_gpu *r) {
  *r = mul_uint128(mul_uint128(*a, *b), *c);
}

void test_mul_uint128_twice() {
  
  // Init v1 and v2 for mult
  uint128_t v1 = 0x12345678;
  v1 <<= 64;
  v1 |= 0x23456789;

  uint128_t v2 = 0x34567890;
  v2 <<= 64;
  v2 |= 0x45678901;

  uint128_t v3 = 0x123;
  v3 <<= 64;
  v3 |= 0x456;

  uint128_t_gpu a = uint128_gpu_from(v1);
  uint128_t_gpu b = uint128_gpu_from(v2);
  uint128_t_gpu c = uint128_gpu_from(v3);

  // Alloc gpu mem
  uint128_t_gpu *r;
  cudaMalloc((void **)&r, sizeof(uint128_t_gpu));

  uint128_t_gpu *a_gpu, *b_gpu, *c_gpu;
  cudaMalloc((void **)&a_gpu, sizeof(uint128_t_gpu));
  cudaMalloc((void **)&b_gpu, sizeof(uint128_t_gpu));
  cudaMalloc((void **)&c_gpu, sizeof(uint128_t_gpu));  
  cudaMemcpy(a_gpu, &a, sizeof(uint128_t_gpu), cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, &b, sizeof(uint128_t_gpu), cudaMemcpyHostToDevice);    cudaMemcpy(c_gpu, &c, sizeof(uint128_t_gpu), cudaMemcpyHostToDevice);  
  
  test_mul_uint128_kernel_twice<<<1, 1>>>(a_gpu, b_gpu, c_gpu, r);
  uint128_t_gpu r_cpu;
  cudaMemcpy(&r_cpu, r, sizeof(uint128_t_gpu), cudaMemcpyDeviceToHost);
  
  uint128_t truth = (v1*v2)*v3;
  
  assert(r_cpu.x == (truth & 0xFFFFFFFF));
  assert(r_cpu.y == ((truth & 0xFFFFFFFF00000000) >> 32));
  assert(r_cpu.w == truth >> 96);
  assert(r_cpu.z == ((truth >> 64) & 0xFFFFFFFF));  
  
  cudaFree(r);
  cudaFree(a_gpu);
  cudaFree(b_gpu); 
}

void test_uint128_gpu_conversion() {
  for (int i = 0; i < 1000; i++) {
    uint128_t k = i * 0x12345;

    uint128_t_gpu v = uint128_gpu_from(k);
    uint128_t v_back = uint128_from_gpu(v);

    assert(v_back == k);
  }
}

int main(void) {
  test_uint128_gpu_from();
  test_uint128_from();
  test_add_uint128();
  test_mul_uint128();
  test_mul_uint128_twice();
  test_uint128_gpu_conversion();  
  printf("PASS\n");
}
