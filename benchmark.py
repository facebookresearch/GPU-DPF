# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
import dpf

for N in [16384, 65536, 262144, 1048576]:
    dpf.test_gpu_dpf_perf(N=N, prf=dpf.DPF.PRF_AES128)
    dpf.test_gpu_dpf_perf(N=N, prf=dpf.DPF.PRF_SALSA20)
    dpf.test_gpu_dpf_perf(N=N, prf=dpf.DPF.PRF_CHACHA20)
