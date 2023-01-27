# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

# sample.py
# ------------------------------------
# Example usage of DPF interface.
#
# Problem setting:
# - Client wishes to retrieve an entry from a table held on two non-colluding servers
# - Client does not wish to leak any information about the index they are retrieving
#
# Solution:
# - Client constructs a DPF representing their secret index
# - Client generates two keys k1, k2 from the DPF
# - Client sends k1, k2 to non-colluding servers 1 and 2 respectively
# - Servers 1 and 2 evaluate k1 and k2 returning the result
# - Client adds the shares together to obtain the table entry

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
