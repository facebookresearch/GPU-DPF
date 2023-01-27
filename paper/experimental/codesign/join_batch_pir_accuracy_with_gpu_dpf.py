# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.  

import sys
import pprint
import os
import glob
import numpy as np
import json

d_in = sys.argv[1]
app = sys.argv[2]
dpf_in = sys.argv[3]
d_out = sys.argv[4]

def application_config(app):
    if app == "lm":
        return 33278
    if app == "movielens":
        return 131263
    if app == "taobao":
        return 846811
    assert(0)
    
def load_acc_data(d_in):
    files = glob.glob(f"{d_in}/*")
    print(len(files))

    d = []
    for f in files:
        with open(f, "r") as ff:
            d.append((f, json.load(ff)))

    return d

def load_dpf_perf_data(dpf_in):
    files = glob.glob(f"{dpf_in}/*")
    d = []

    for f in files:
        with open(f, "r") as ff:
            last_line = ff.readlines()[-1]
            try:
                data = eval(last_line)
                d.append(data)

            except:
                pass
    return d

def compute_joined_data(num_embeddings, pir_stat, dpf_perf_numbers):

    hot_table_embeddings = int(pir_stat["hotcold_config"]["cache_size_fraction"]*num_embeddings)
    cold_table_embeddings = num_embeddings - hot_table_embeddings

    num_collocate = pir_stat["collocate_config"]["num_collocate"]

    queries_to_hot = pir_stat["pir_config"]["queries_to_hot"]
    queries_to_cold = pir_stat["pir_config"]["queries_to_cold"]

    num_bins = pir_stat["pir_config"]["num_bins"]

    if num_bins == 1:
        return None    
    
    if num_bins > 1:
        bin_size_hot = num_bins
        bin_size_cold = num_bins
    elif num_bins < 1:
        bin_size_hot = int(num_bins * hot_table_embeddings)
        bin_size_cold = int(num_bins * cold_table_embeddings)

    num_bins_hot = hot_table_embeddings // bin_size_hot + 1
    num_bins_cold = 0 if cold_table_embeddings == 0 else cold_table_embeddings // bin_size_cold + 1

    hot_numbers = []
    cold_numbers = []
    
    for perf_number in dpf_perf_numbers:
        throughput = perf_number["throughput_queries_per_ms"]
        latency = perf_number["latency_ms"]
        batchsize = perf_number["batch_size"]

        if perf_number["entries"] >= bin_size_hot and perf_number["entry_size_ints"]*128/8 >= num_collocate*16:            
            n_queries_for_hot = num_bins_hot * queries_to_hot
            #print(n_queries_for_hot, bin_size_hot)
            #if n_queries_for_hot > batchsize:
            #    continue
            hot_throughput = batchsize / n_queries_for_hot
            hot_latency = np.ceil(n_queries_for_hot / batchsize)*latency
            hot_numbers.append((hot_throughput, hot_latency))
            
        if perf_number["entries"] >= bin_size_cold and perf_number["entry_size_ints"]*128/8 >= num_collocate*16:
            n_queries_for_cold = num_bins_cold * queries_to_cold
            #if n_queries_for_cold > batchsize or n_queries_for_cold == 0:
            #    continue
            if n_queries_for_cold == 0:
                cold_throughput = float("inf")
                cold_latency = float("0")
            else:
                cold_throughput = batchsize / n_queries_for_cold
                cold_latency = np.ceil(n_queries_for_cold / batchsize)*latency
            cold_numbers.append((cold_throughput, cold_latency))
            

    # Since we need to compute both] hot and cold numbers, assuming we have 2 GV100 gpus
    # we take the maxs of latency and mins of throughputs
    latency_throughputs = []
    for h in hot_numbers:
        for c in cold_numbers:
            #latency = max(h[1],c[1])
            latency = h[1]+c[1]
            #throughput = min(h[0],c[0])
            throughput = min(h[0],c[0])/2
            latency_throughputs.append((latency, throughput))
    latency_throughputs = list(set(latency_throughputs))

    #print(len(latency_throughputs))

    latency_throughputs_final = []
    for latency, throughput in latency_throughputs:
        best_throughput = max([x[1] for x in latency_throughputs if x[0] <= latency])
        #print(best_throughput)
        latency_throughputs_final.append((latency, best_throughput))
        
    if len(latency_throughputs_final) <= 0:
        return None
        
    assert(len(latency_throughputs_final) > 0)
    #print(sorted(latency_throughputs_final))
    #sys.exit(0)    

    return  {**pir_stat, **{"latency_throughputs" : latency_throughputs_final}}

def join_data(num_embeddings, data, dpf_perf_numbers):
    final = []
    for fname, pir_stats in data:
        joined_data = compute_joined_data(num_embeddings, pir_stats, dpf_perf_numbers)
        final.append((fname, joined_data))
        
    for fname, final_d in final:
        bname  = os.path.basename(fname)
        fpath = f"{d_out}/{bname}"

        if not os.path.exists(d_out):
            os.makedirs(d_out)

        #print(fpath)
        if final_d is not None:        
            with open(fpath, "w") as ff:
                json.dump(final_d, ff)

if __name__ == "__main__":
    num_embeddings = application_config(app)
    data = load_acc_data(d_in)
    dpf_perf_numbers = load_dpf_perf_data(dpf_in)

    join_data(num_embeddings, data, dpf_perf_numbers)
