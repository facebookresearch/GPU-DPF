# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.  

import sys
import os
import json
import pprint
from collections import namedtuple
import numpy as np
import random
import pprint
import matplotlib.pyplot as plt
import batch_pir_optimization

mode = sys.argv[1]

if mode == "lm":
     import language_model_dataset as dataset
     from language_model_dataset import *
     dir_out = "language_model_sweep_out"

if mode == "movielens":
     import movielens_dataset as dataset
     from movielens_dataset import *
     dir_out = "movielens_sweep_out" 
     initialize_collocate_load_from="initialize_collocate_movielens.json"    

if mode == "taobao":
     import taobao_rec_dataset_v2 as dataset
     from taobao_rec_dataset_v2 import *

     dir_out = "taobao_sweep_out"
     initialize_collocate_load_from="initialize_collocate_taobao.json"

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

dataset.initialize()

# Config params (try to be independent of dataset)
assert(dataset.num_embeddings > 0)

# Get average number of accesses
access_lengths = []
for d in dataset.val_access_pattern:
     access_lengths.append(len(d))
avg_access_length = np.mean(access_lengths)
max_access_length = np.max(access_lengths)
access_length_90 = np.percentile(access_lengths, 90)

hot_cold_ratios = [1, .05, .1, .15, .2, .25, .3]
num_collocation = [0, 1, 2, 3, 4, 5]

#pir_num_bins = [int(x) for x in list(np.arange(1, dataset.num_embeddings, dataset.num_embeddings//10))]

# We are going change num_bins -> bin_fraction (i.e: fraction of total dataset)
pir_num_bins = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

# We got by powers of 2 but not as much since generally most users don't click on many ads
pir_hot_queries = [int(x) for x in list(np.arange(1, access_length_90, max(1, access_length_90//10)))]
pir_cold_queries = [int(x) for x in list(np.arange(1, access_length_90, max(1,access_length_90//10)))]

with open(initialize_collocate_load_from, "r") as f:
     initialize_collocate_load_from = json.load(f)

def run(hot_cold_ratio, num_collocation, pir_num_bins, num_hot_queries, num_cold_queries):
    hotcold_config = batch_pir_optimization.HotColdConfig(hot_cold_ratio)
    collocate_config = batch_pir_optimization.CollocateConfig(num_collocation)
    pir_config = batch_pir_optimization.PIRConfig(pir_num_bins, 200, num_hot_queries, num_cold_queries)

    b = batch_pir_optimization.BatchPIROptimize(dataset.train_access_pattern,
                                                dataset.val_access_pattern,
                                                hotcold_config, collocate_config, pir_config,
                                                initialize_collocate_load_from=initialize_collocate_load_from)
    b.evaluate_real()
    results = b.summarize_evaluation()

    f_out = f"{dir_out}/{hot_cold_ratio}_{num_collocation}_{ pir_num_bins}_{num_hot_queries}_{num_cold_queries}"
    with open(f_out, "w") as f:
        json.dump(results, f)

if __name__=="__main__":

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    args = []
    for r in hot_cold_ratios:
        for c in num_collocation:
            for b in pir_num_bins:
                for q in pir_hot_queries:
                    for x in pir_cold_queries:
                        args_set = (r, c,  b, q, x)
                        args.append(args_set)

    np.random.shuffle(args)

    batch_pir_only = [x for x in args if x[0] == 1 and x[1] == 0]
    batch_pir_with_hot_cold = [x for x in args if x[1] == 0 and x[0] < 1]
    batch_pir_with_hot_cold_and_coll = [x for x in args if x[1] > 1 and x[0] < 1]
    batch_pir_with_coll = [x for x in args if x[1] > 0 and x[0] == 1]


    max_len = max([len(x) for x in [batch_pir_only, batch_pir_with_hot_cold, batch_pir_with_hot_cold_and_coll, batch_pir_with_coll]])

    batch_pir_only = (batch_pir_only*max_len)[:max_len]
    batch_pir_with_hot_cold = (batch_pir_with_hot_cold*max_len)[:max_len]
    batch_pir_with_hot_cold_and_coll = (batch_pir_with_hot_cold_and_coll*max_len)[:max_len]
    batch_pir_with_coll = (batch_pir_with_coll*max_len)[:max_len]

    args = batch_pir_only + batch_pir_with_hot_cold + batch_pir_with_hot_cold_and_coll + batch_pir_with_coll
    np.random.shuffle(args)

    #args.sort(key=lambda x: 0 if (x[0] == 1 and
    #                              x[1] == 0 and
    #                              (x[2] == 1 or x[2] == 4 or x[2] == 256) and
    #                              (x[3] == 1 or x[3] == 4 or x[3] == 16) and
    #                              (x[4] == 1 or x[4] == 4 or x[4] == 16)) else 1)

    #print(args[0:3])
    #args = [x for x in args if x[2] == .1]
    #run(*args[0])
    #run(*args[1])
    #run(*args[2])
    
    with Pool(processes=8) as pool:
        pool.starmap(run, args)

     
