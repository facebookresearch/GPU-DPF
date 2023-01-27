# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

batch_sizes=( 1 32 128 512 )
lengths=( 16 128 1024 16384 1048576 )
emb_sizs=( 32 64 128 256 512 )
threads=( 100 56 28 16 8 4 2 1 )

for b in "${batch_sizes[@]}"; do
    for length in "${lengths[@]}"; do
	for emb_siz in "${emb_sizs[@]}"; do
	    for thread in "${threads[@]}"; do
		./benchmark $length $emb_siz 1 $b 100 0 $thread > multithread_dpf_benchmark_n=${length}_d=${emb_siz}_b=${b}_threads=${thread}
	    done
	done
    done
done
