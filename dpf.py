# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
import sys
import os
import copy
import time
import torch
import dpf_cpp
import random
import uuid
import numpy as np
import torch.nn.functional as F

#
# A list of outstanding todos for future contributor PRs
#
# - Enable more than 16 32-bit entries for tables. This can be done in 2 ways:
#   1) Editing the C++ code to remove 16-bit entry constraints and modifying CUDA kernel accordingly
#   2) Modifying the python wrapper to call eval_init/eval multiple times per 16 32-bit entry chunks
#   2 is a much easier path towards general entry-lengthed tables, but 1 may be more efficient.
#
# - Pack table entries into 128-bit chunks before running through CPU DPF code.
#   This means we can serve tables w/ 4x the entries as current, accelerating perf.
#
# - Allow non-power of 2 DPFs by modifying the wrapper accordingly
#
# - Enable flags to choose other available DPF strategy (i.e: cooperative groups)
#
# - Implement more PRFs, particularly PRFs that are GPU friendly, for better perf
#
# - Possibly enable returning the shared one-hot for the eval_gpu method
#
# - Optimize AES-128 PRF by combining key expansion w/ encryption. This reduces
#   local memory usage, and increases GPU occupancy.

class DPF(object):

    PRF_CHACHA20 = dpf_cpp.PRF_CHACHA20
    PRF_DUMMY = dpf_cpp.PRF_DUMMY
    PRF_SALSA20 = dpf_cpp.PRF_SALSA20
    PRF_AES128 = dpf_cpp.PRF_AES128

    ENTRY_SIZE = dpf_cpp.ENTRY_SIZE
    BATCH_SIZE = dpf_cpp.BATCH_SIZE

    DEFAULT_PRF = dpf_cpp.PRF_AES128

    def __init__(self, prf=None):        
        
        self.buffers = None
        self.table_num_entries = None
        self.table_effective_entry_size = None

        self.table = None

        self.prf_method = prf if prf is not None else self.DEFAULT_PRF
        self.prf_method_string = {
            self.PRF_CHACHA20 : "CHACHA20",
            self.PRF_DUMMY : "DUMMY",
            self.PRF_SALSA20 : "SALSA20",
            self.PRF_AES128 : "AES128",            
        }[self.prf_method]

    def gen(self, k, n):

        # TODO: Please replace with secure 128-bit RNG
        seed = os.urandom(128)

        if n & (n-1) != 0:
            raise Exception("Table num entries (%d) must be a power of two" % (n))
        if k >= n:
            raise Exception("k (%d), the selected element, must be less than n (%d), the number of entries in the table"
                            % (k, n))
        
        return dpf_cpp.gen(k, n, seed, self.prf_method)

    def eval_cpu(self, keys, one_hot_only=False):

        # One-hot only mode returns the one-host secret shared vector
        if one_hot_only:
            return torch.stack([dpf_cpp.eval_cpu(k, self.prf_method) for k in keys])

        if self.table is None:
            raise Exception("Must call `eval_init` before `eval_cpu` with one_hot_only=False")

        one_hots = torch.stack([dpf_cpp.eval_cpu(k, self.prf_method) for k in keys])
        return torch.matmul(one_hots, self.table)

    def eval_init(self, table):

        self.table = table

        # Free buffers if previously initialized
        if self.buffers is not None:
            dpf_cpp.eval_free(self.buffers)

        # Err check
        self.table_num_entries = table.shape[0]
        self.table_effective_entry_size = table.shape[1]

        if self.table_num_entries < 128:
            raise Exception("Table (%d) must have at least 128 elements" % self.table_num_entries)
        if self.table_num_entries & (self.table_num_entries-1) != 0:
            raise Exception("Table num entries (%d) must be a power of two" % (self.table_num_entries))
        if self.table_effective_entry_size > self.ENTRY_SIZE:
            raise Exception("Table entry dimension (%d) must be < %d" %
                            (self.table_effective_entry_size, self.ENTRY_SIZE))
            
        # Pad table
        pad = (0, self.ENTRY_SIZE-self.table_effective_entry_size, 0, 0)
        table = F.pad(table, pad)

        # Init buffers from padded table
        self.buffers = dpf_cpp.eval_init(table)

    def eval_gpu(self, keys):
        effective_batch_size = len(keys)
        
        if self.buffers is None:
            raise Exception("Must call `eval_init` before `eval_gpu`")

        # Process in batches
        all_results = []
        for i in range(0, len(keys), self.BATCH_SIZE):
            # Pad batch size
            cur_keys = keys[i:i+dpf_cpp.BATCH_SIZE]
            cur_keys = cur_keys + [cur_keys[-1]] * (self.BATCH_SIZE-len(cur_keys))            
            result = dpf_cpp.eval_gpu(cur_keys, self.buffers, self.table_num_entries, self.prf_method)
            result = result[:,:self.table_effective_entry_size]
            all_results.append(result)
        all_results = torch.cat(all_results)
        return all_results[:effective_batch_size,:]
        
    def __repr__(self):
        if self.buffers is None:
            return "DPF(_uninitialized_, prf_method=%s)" % self.prf_method_string
        else:
            return "DPF(entries=%d, entry_size=%d, prf_method=%s)" % (self.table_num_entries, self.table_effective_entry_size, self.prf_method_string)

def test_cpu_dpf_one_hot(N=1024):
    dpf = DPF()
    
    ###########################
    # Client
    ###########################
    K = 42
    k1, k2 = dpf.gen(K, N)

    ########################
    # Server
    ########################

    # Sanity check cpu

    # DPF
    v1 = dpf.eval_cpu([k1], one_hot_only=True)
    v2 = dpf.eval_cpu([k2], one_hot_only=True)
    rec = (v1-v2).numpy()

    # Groundtruth
    gt = np.zeros(rec.shape)
    gt[:,K] = 1

    assert(np.linalg.norm(rec - gt) <= 1e-8)    
    
    print("Pass CPU (one-hot only) check.")

def test_cpu_dpf(N=1024):
    dpf = DPF()
    
    ###########################
    # Client
    ###########################
    k1s = []
    k2s = []
    gt_indices = []
    for i in range(64):
        indx = random.randint(0, N-1)
        gt_indices.append(indx)
        k1, k2 = dpf.gen(indx, N)
        k1s.append(k1)
        k2s.append(k2)

    #############################
    # Server
    #############################
    # Generate table and initialize DPF table
    table = torch.zeros((N, 16)).int()
    for i in range(N):
        for j in range(16):
            table[i,j] = i*16+j

    dpf.eval_init(table)

    # DPF
    a = dpf.eval_cpu(k1s)
    b = dpf.eval_cpu(k2s)
    rec = (a-b).numpy()

    # Groundtruth
    gt = table[gt_indices, :].numpy()

    assert(np.linalg.norm(rec-gt) <= 1e-8)

    print("Pass CPU check.")    

def test_gpu_dpf(N=8192):
    dpf = DPF()
    
    ###########################
    # Client
    ###########################
    k1s = []
    k2s = []
    gt_indices = []
    for i in range(64):
        indx = random.randint(0, N-1)
        gt_indices.append(indx)
        k1, k2 = dpf.gen(indx, N)
        k1s.append(k1)
        k2s.append(k2)

    #############################
    # Server
    #############################
    # Generate table and initialize DPF table
    table = torch.zeros((N, 16))
    for i in range(N):
        for j in range(16):
            table[i,j] = i*16+j

    dpf.eval_init(table)

    # DPF
    a = dpf.eval_gpu(k1s)
    b = dpf.eval_gpu(k2s)
    rec = (a-b).numpy()

    # Groundtruth
    gt = table[gt_indices, :].numpy()

    assert(np.linalg.norm(rec-gt) <= 1e-8)

    print("Pass GPU check.")

def test_gpu_dpf_sweep():
    N = [128, 256, 512, 1024, 8192]
    for n in N:
        test_gpu_dpf_nopad(n, batch=random.randint(1, dpf_cpp.BATCH_SIZE*5-1), entrysize=random.randint(1, 16-1))
    print("Pass GPU (sweep) check.")

def test_gpu_dpf_nopad(N=8192, batch=42, entrysize=13):
    dpf = DPF()
    
    ###########################
    # Client
    ###########################
    k1s = []
    k2s = []
    gt_indices = []
    for i in range(batch):
        indx = random.randint(0, N-1)
        gt_indices.append(indx)
        k1, k2 = dpf.gen(indx, N)
        k1s.append(k1)
        k2s.append(k2)

    #############################
    # Server
    #############################
    # Generate table and initialize DPF table
    table = torch.randint(2**31, (N, entrysize)).int()
    dpf.eval_init(table)

    # DPF
    a = dpf.eval_gpu(k1s)
    b = dpf.eval_gpu(k2s)
    rec = (a-b).numpy()

    # Groundtruth
    gt = table[gt_indices, :].numpy()

    assert(np.linalg.norm(rec-gt) <= 1e-8)

    print("Pass GPU (nopad) check.")
    
def test_gpu_dpf_perf(N=2048, batch=dpf_cpp.BATCH_SIZE, entrysize=16, prf=DPF.DEFAULT_PRF):
    dpf = DPF(prf=prf)
    
    ###########################
    # Client
    ###########################
    k1s = []
    k2s = []
    gt_indices = []
    for i in range(batch):
        indx = random.randint(0, N-1)
        gt_indices.append(indx)
        k1, k2 = dpf.gen(indx, N)
        k1s.append(k1)
        k2s.append(k2)

    #############################
    # Server
    #############################
    # Generate table and initialize DPF table
    table = torch.rand(N, entrysize).int()

    dpf.eval_init(table)

    # DPF
    tstart = time.time()
    reps = 10
    for i in range(reps):
        dpf.eval_gpu(k1s)
    tend = time.time()
    elapsed = tend-tstart

    dpfs_per_sec = batch*reps/elapsed
    keysize = np.prod(k1s[0].shape)*4
    print("%s Key Size: %d bytes, Perf: %d dpfs/sec" % (dpf, keysize, dpfs_per_sec))

def test_cpu_dpf_perf(N=2048, batch=dpf_cpp.BATCH_SIZE, entrysize=16, prf=DPF.DEFAULT_PRF):
    dpf = DPF(prf=prf)
    
    ###########################
    # Client
    ###########################
    k1s = []
    k2s = []
    gt_indices = []
    for i in range(batch):
        indx = random.randint(0, N-1)
        gt_indices.append(indx)
        k1, k2 = dpf.gen(indx, N)
        k1s.append(k1)
        k2s.append(k2)

    #############################
    # Server
    #############################
    # Generate table and initialize DPF table
    table = torch.rand(N, entrysize).int()

    dpf.eval_init(table)

    # DPF
    tstart = time.time()
    reps = 10
    for i in range(reps):
        dpf.eval_cpu(k1s)
    tend = time.time()
    elapsed = tend-tstart

    dpfs_per_sec = batch*reps/elapsed
    keysize = np.prod(k1s[0].shape)*4
    print("%s Key Size: %d bytes, Perf: %d dpfs/sec" % (dpf, keysize, dpfs_per_sec))
        
if __name__=="__main__":
    random.seed(time.time())
    test_cpu_dpf()    
    test_cpu_dpf_one_hot()
    test_gpu_dpf()
    test_gpu_dpf_nopad()
    test_gpu_dpf_sweep()
    test_gpu_dpf_perf()
    
    #test_cpu_dpf_perf() # commented out because it is slow
    
