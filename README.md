# GPU Distributed Point Functions

This codebase implements a high-performance GPU implementation of a distributed point function, and exposes a simple and easy to use python interface. 

This repository is the source code for the paper [GPU-based Private Information Retrieval for On-Device ML Inference](https://arxiv.org/abs/2301.10904).


## Background

A **Distributed Point Function (DPF)** is a cryptographic primitive that allows a client to **efficiently** and **privately** access an entry of a table replicated across two non-colluding servers.

<p align="center">
<img src="https://github.com/facebookresearch/GPU-DPF/blob/main/imgs/dpf.png" width="300">
</p>

The workflow for private table accesses using distributed point functions is:
- Client **generates** two compact keys k_a, k_b that represents the secret index they wish to retrieve
- Client **sends** k_a, k_b across the network to two non-colluding servers Server 1, Server 2 respectively
- Server 1 and Server 2 **evaluate** the keys, and return the result
- Client **sums** the returned shares to obtain the table entry

By using a DPF
1) **No information about the client's secret index is revealed**
    * Assuming no collusion between servers.
2) **Network communication costs are minimized** 
    * Key sizes are *compact* on the order of 2KB for tables with up to 2^32 entries.
    
## How do Distributed Point Functions Work?

We describe how distributed point functions work [here](https://github.com/facebookresearch/GPU-DPF/blob/main/DPF.md).

## Accelerating Distributed Point Functions with GPUs

Evaluating DPFs is computationally intensive, making GPU acceleration _key_ to obtaining high performance. 

This codebase implements a high-performance GPU implementation of a distributed point function, and exposes a simple and easy to use python interface. By leveraging the GPU we are able to speed up DPF evaluation by over an order of magnitude over a multi-core CPU. 

We accelerate the DPF construction described [here](https://www.iacr.org/archive/eurocrypt2014/84410245/84410245.pdf). This DPF construction generates keys that are **log(n)** the size of the number of entries of the table, and require **O(n)** computation to evaluate the keys.

## Requirements

- python, pytorch, numpy
- CUDA GPU (tested on cuda > 11.4)

## Installation

```
bash install.sh
```

To check success, run `python dpf.py`. All checks should pass. 

## Example

Example usage (from `sample.py`). See `dpf.py` for more.

```python
import sys
import dpf
import torch

# Table parameters
table_size = 16384
entry_size = 1

# The actual table (replicated on 2 non-colluding servers)
table = torch.randint(2**31, (table_size, entry_size)).int()
table[42,:] = 42

def server(k):

    # Server initializes DPF w/ table
    dpf_ = dpf.DPF()
    dpf_.eval_init(table)

    # Server evaluates DPF on table    
    return dpf_.eval_gpu([k])

def client():    
    secret_indx = 42

    # Generate two keys that represents the secret indx
    dpf_ = dpf.DPF()
    k1, k2 = dpf_.gen(secret_indx, table_size)

    # Send one key to each server to evaluate.
    # 
    # Assuming that these two servers do not collude,
    # the servers learn _nothing_ about secret_indx.
    a = server(k1).item()
    b = server(k2).item()

    rec = a-b
    
    print(a, b, rec)
    assert(rec == 42)

if __name__=="__main__":
    client()
```

## Benchmark
Benchmark with `python benchmark.py`. Sample output on a P100 GPU.
```
DPF(entries=16384, entry_size=16, prf_method=AES128) Key Size: 2096 bytes, Perf: 23954 dpfs/sec
DPF(entries=16384, entry_size=16, prf_method=SALSA20) Key Size: 2096 bytes, Perf: 76073 dpfs/sec
DPF(entries=16384, entry_size=16, prf_method=CHACHA20) Key Size: 2096 bytes, Perf: 75679 dpfs/sec
DPF(entries=65536, entry_size=16, prf_method=AES128) Key Size: 2096 bytes, Perf: 6131 dpfs/sec
DPF(entries=65536, entry_size=16, prf_method=SALSA20) Key Size: 2096 bytes, Perf: 23141 dpfs/sec
DPF(entries=65536, entry_size=16, prf_method=CHACHA20) Key Size: 2096 bytes, Perf: 22433 dpfs/sec
DPF(entries=262144, entry_size=16, prf_method=AES128) Key Size: 2096 bytes, Perf: 1443 dpfs/sec
DPF(entries=262144, entry_size=16, prf_method=SALSA20) Key Size: 2096 bytes, Perf: 5849 dpfs/sec
DPF(entries=262144, entry_size=16, prf_method=CHACHA20) Key Size: 2096 bytes, Perf: 5830 dpfs/sec
DPF(entries=1048576, entry_size=16, prf_method=AES128) Key Size: 2096 bytes, Perf: 379 dpfs/sec
DPF(entries=1048576, entry_size=16, prf_method=SALSA20) Key Size: 2096 bytes, Perf: 1447 dpfs/sec
DPF(entries=1048576, entry_size=16, prf_method=CHACHA20) Key Size: 2096 bytes, Perf: 1424 dpfs/sec
```

Our current implementation supports tables of sizes up to 2^32, at a fixed key size of ~2KB, assuming 16 integers per entry. These can be configured by editing the C++ wrapper code. 

**Note:** We also provide a CPU implementation of a DPF, however it is less optimized, and not the one we compare against in our paper. Please see [google_dpf](https://github.com/google/distributed_point_functions) for a more optimized CPU implementation. 

## Results

We compare performance between GPU DPF on a V100 GPU vs [CPU](https://github.com/google/distributed_point_functions), using AES-128 for the PRF, with 16 32-bit values per table entry.

| # Table Entries | PRF | CPU 1-thread DPFs/sec | CPU 32-thread DPFs/sec |  V100 GPU DPFs/sec | Speedup vs 1-thread CPU | Speedup vs 32-thread CPU |
| :-: | :-: | :-: | :-: | :-: |  :-: |  :-: | 
| 16384 | AES128 | 220 | 2,810 | **52,536** | 238x | 18.7x |
| 65536 | AES128 | 50 | 688 | **15,392** | 308x | 22.3x | 
| 262144 | AES128 | 13 | 212 | **3,967** | 305x | 18.7x | 
| 1048576 | AES128 | 3 | 55 | **923** | 307x | 16.8x | 

Overall, our GPU DPF implementation attains over **200x** speedup over an optimized single-threaded CPU DPF implementation, and over **15x** speedup over an optimized multi-threaded CPU DPF implementation.

Further performance numbers for GPU DPF on a V100 GPU for Salsa20/Chacha20 PRFs.
| # Table Entries | PRF | V100 GPU DPFs/sec | 
| :-: | :-: | :-: | 
| 16384 | SALSA20 | 145,646 |
| 65536 | SALSA20 | 54,892 |
| 262144 | SALSA20 | 16,650 |
| 1048576 | SALSA20 | 3,894 |
| 16384 | CHACHA20 | 139,590 |
| 65536 | CHACHA20 | 56,120 |
| 262144 | CHACHA20 | 16,086 |
| 1048576 | CHACHA20 | 4,054 |


## License

GPU-DPF is released under the [Apache 2.0 license](https://github.com/facebookresearch/GPU-DPF/blob/main/LICENSE).

## Citation

```
@article{lam2023gpudpf,
  title={GPU-based Private Information Retrieval for On-Device Machine Learning Inference}, 
  author={Maximilian Lam and Jeff Johnson and Wenjie Xiong and Kiwan Maeng and Udit Gupta and Minsoo Rhu and Hsien-Hsin S. Lee and Vijay Janapa Reddi and Gu-Yeon Wei and David Brooks and Edward Suh},
  year={2023},
  eprint={2301.10904},
  archivePrefix={arXiv},
  primaryClass={cs.CR}
}
```

## Disclaimer

**This open source project is for proof of concept purposes only and should not be used in production environments. The code has not been officially vetted for security vulnerabilities and provides no guarantees of correctness or security. Users should carefully review the code and conduct their own security assessments before using the software.**
