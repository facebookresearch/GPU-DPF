# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

DPF_STRATEGY="DPF_BREADTH_FIRST"
PRF_METHOD="SALSA20_12_CIPHER"
#DPF_STRATEGY="DPF_HYBRID"
#PRF_METHOD="DUMMY"
NUM_ENTRIES=16384
BATCH_SIZE=32
ENTRY_SIZE=16
FUSES_MATMUL=0
PERFORM_MATMUL=0

PRF_CONFIG_FLAGS=-DPRF_METHOD=$(PRF_METHOD) -DDPF_STRATEGY=$(DPF_STRATEGY) -DMM=$(ENTRY_SIZE) -DKK=$(NUM_ENTRIES) -DNN=$(BATCH_SIZE) -DFUSES_MATMUL=$(FUSES_MATMUL) -DPERFORM_MATMUL=$(PERFORM_MATMUL)

CC=g++
NVCC=nvcc
NVCC_FLAGS=-lineinfo --no-host-device-initializer-list -Xcudafe --display_error_number

benchmark:
	$(NVCC) $(PRF_CONFIG_FLAGS) dpf_gpu/dpf_benchmark.cu $(NVCC_FLAGS) -o dpf_benchmark

profile:
	$(NVCC) $(PRF_CONFIG_FLAGS) -DREPS=1 dpf_gpu/dpf_benchmark.cu $(NVCC_FLAGS) -o dpf_benchmark
	ncu  -f --import-source yes --set full --replay-mode application -o $(DPF_STRATEGY),num_entries=$(NUM_ENTRIES),batch_size=$(BATCH_SIZE),entry_size=$(ENTRY_SIZE),fuses_matmul=$(FUSES_MATMUL),perform_matmul=$(PERFORM_MATMUL),prf_method=$(PRF_METHOD) --target-processes all ./dpf_benchmark

benchmark_matmul:
	$(NVCC) $(PRF_CONFIG_FLAGS) dpf_gpu/matmul_benchmark.cu $(NVCC_FLAGS) -o matmul_benchmark

profile_matmul:
	$(NVCC) $(PRF_CONFIG_FLAGS) -DREPS=1 dpf_gpu/matmul_benchmark.cu $(NVCC_FLAGS) -o matmul_benchmark
	ncu -f  --import-source yes --set full -o matmul,num_entries=$(NUM_ENTRIES),batch_size=$(BATCH_SIZE),entry_size=$(ENTRY_SIZE) --target-processes all ./matmul_benchmark

test_128_bit_functionality:
	$(NVCC) dpf_gpu/tests/test_128_bit.cu $(NVCC_FLAGS) -o test_128_bit
	./test_128_bit
	rm -f *.o
	rm test_128_bit

test_dpf_base:
	$(CC) -Ofast dpf_base/dpf.cc -o dpf_cpu_base
	./dpf_cpu_base
	rm -f *.o
	rm dpf_cpu_base

