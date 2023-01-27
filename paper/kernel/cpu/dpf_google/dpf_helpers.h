// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <vector>

void *DPFInitialize(int log_domain_size, int bitsize);
void DPFGetKey(void *dpf, int alpha, int beta, void **k1, void **k2);
void DPFExpand(void *dpf_ptr, void *key_data, int N, uint32_t *out);
