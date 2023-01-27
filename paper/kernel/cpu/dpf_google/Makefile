# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# NOTE: REMOVED GOOGLE DPF CODEBASE, DOWNLOAD IT AGAIN HERE: https://github.com/google/distributed_point_functions
# 
# Really long compile command for linking with Google's DPF library
# To get this working I
# - Add build target to DPF bazel build file and build using bazel
# - Find all the header dependencies that bazel uses + external ones (e.g: boringssl)
# - Run bazel test //... and copy over generated protobuf files for distributed_point_function.proto
#
# Make sure when running to add path to dpf.so
# - e.g: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH://private/home/maxlam/pir/dpf/distributed_point_functions/bazel-bin/dpf

CC=g++
CFLAGS=-std=c++17 -O3 -fopenmp
NVCC=nvcc -Xcompiler -fopenmp
CUDA_FLAGS=-lcuda -lcudart -lcublas
DPF_INCS=-L ./distributed_point_functions/bazel-bin/dpf/ -I /private/home/maxlam/pir/dpf/distributed_point_functions/dpf -I/private/home/maxlam/.conda/envs/pir/share/bazel/dacb9117b7bba7784d38f8b74d338ba9/external/boringssl/src/include/  -I /private/home/maxlam/.conda/envs/pir/share/bazel/dacb9117b7bba7784d38f8b74d338ba9/external/com_google_absl -I /private/home/maxlam/pir/dpf/distributed_point_functions -I /private/home/maxlam/.conda/envs/pir/share/bazel/dacb9117b7bba7784d38f8b74d338ba9/external/com_google_protobuf_protoc_linux_x86_64/include/ -I /private/home/maxlam/.conda/envs/pir/share/bazel/dacb9117b7bba7784d38f8b74d338ba9/external/com_github_protocolbuffers_protobuf/src/ -I /private/home/maxlam/.conda/envs/pir/share/bazel/dacb9117b7bba7784d38f8b74d338ba9/external/com_github_google_highway/ -ldpf 

sanity:
	$(CC) $(CFLAGS) test_dpf_so.cc $(DPF_INCS) -o sanity

benchmark:
	$(NVCC) -c benchmark.cu -o benchmark.o $(CUDA_FLAGS)
	$(CC) $(CFLAGS) -c dpf_helpers.cc $(DPF_INCS) -o dpf_helpers.o
	$(NVCC) -o benchmark benchmark.o dpf_helpers.o $(DPF_INCS) $(CUDA_FLAGS)

	# Make sure when running to add path to dpf.so
	# - e.g: export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH://private/home/maxlam/pir/dpf/distributed_point_functions/bazel-bin/dpf

	# benchmark [n_embedding_entries] [embedding_length] [use_dpf] [batch] [reps] [use_gemm] [dpf_threads]
	#./benchmark 10000000 256 0 32 100 1 1
clean:
	rm -f benchmark
	rm -f sanity
	rm -f *.o
