# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

NUM_ENTRIES=( 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 )
BATCH_SIZE=( 8 16 32 64 128 256 512 1024 2048 4096 )

mkdir -p sweep/sweep_entry_size=1

for num_entries in "${NUM_ENTRIES[@]}"; do
    for batch_size in "${BATCH_SIZE[@]}"; do
	echo $num_entries $batch_size	
	make ENTRY_SIZE=1 NUM_ENTRIES=$num_entries FUSES_MATMUL=1 DPF_STRATEGY="DPF_HYBRID" BATCH_SIZE=$batch_size  benchmark
	./dpf_benchmark > sweep/sweep_entry_size=1/entries=${num_entries},batch_size=${batch_size} 2>&1 
    done
done
