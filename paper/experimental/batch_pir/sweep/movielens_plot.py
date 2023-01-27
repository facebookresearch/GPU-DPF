# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.  

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob
import sys

dir_out = "movielens_sweep_out"
files = glob.glob(dir_out + "/*")

data = []

for f in files:
    with open(f, "r") as ff:
        d = json.load(ff)
        data.append(d)

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs, reverse=False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if not reverse:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)  # Keep any point with a lower cost                
            is_efficient[i] = True  # And keep self
    return is_efficient

def get_pareto_points(xs, ys, reverse=False):
    points = list(zip(xs, ys))
    is_efficient = is_pareto_efficient_simple(np.array(points), reverse=reverse)
    selected = [x for i,x in enumerate(points) if is_efficient[i]]
    selected.sort(key=lambda x:x[0])
    return [x[0] for x in selected], [x[1] for x in selected]

def plot_computation_vs_accuracy(data):
    accuracy_baseline = max([x["accuracy_stats"]["auc"] for x in data])
    
    data = [x for x in data if x["cost"]["upload_communication"] + x["cost"]["download_communication"] <= 300000]
    print(len(data))

    plt.cla()
    plt.clf()

    plt.axhline(accuracy_baseline)
    
    plain = [x for x in data if x["hotcold_config"]["cache_size_fraction"] == 1 and x["collocate_config"]["num_collocate"] == 0]
    collocate_only = [x for x in data if x["hotcold_config"]["cache_size_fraction"] == 1]
    hotcold_only = [x for x in data if x["collocate_config"]["num_collocate"] == 0]
    basic = [x for x in data if x["pir_config"]["num_bins"] == 1 and x["hotcold_config"]["cache_size_fraction"] == 1 and  x["collocate_config"]["num_collocate"] == 0]

    print("plain", len(plain))
    print("coll", len(collocate_only))
    print("hotc", len(hotcold_only))
    print("basic", len(basic))
    
    def plot_single(data, label, marker, color):
        accuracy = [x["accuracy_stats"]["auc"] for x in data]
        computation = [x["cost"]["computation"]/1000 for x in data]    
    

        computation, accuracy = get_pareto_points(computation, [-x for x in accuracy], reverse=False)
        accuracy = [-x for x in accuracy]

        print(label, list(zip(computation, accuracy)))        
        
        #plt.scatter(computation, accuracy, label=label, alpha=.3)
        plt.plot(computation, accuracy, label=label, markersize=15, marker=marker, alpha=1, color=color, linewidth=5)

    plot_single(plain, "batch-pir", "o", "black")
    plot_single(collocate_only, "batch-pir +c", "x", "red")
    plot_single(hotcold_only, "batch_pir +h", "^", "green")
    plot_single(data, "batch-pir +c +h", "v", "blue")

    plt.xlabel("Computation (kPRFs)", fontsize=28)
    plt.ylabel("Accuracy (auc)", fontsize=28)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    #plt.yscale("log")

    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"movielens_computation_vs_auc.pdf", tight_layout=True)        

def plot_communication_vs_accuracy(data):
    accuracy_baseline = max([x["accuracy_stats"]["auc"] for x in data])
    
    data = [x for x in data if x["cost"]["computation"] <= 100000]
    data = [x for x in data if (x["cost"]["upload_communication"]+x["cost"]["download_communication"])/1000 < 1000]
    data = [x for x in data if x["accuracy_stats"]["auc"] >= .75]
    
    print(len(data))

    plt.cla()
    plt.clf()

    plt.axhline(accuracy_baseline)
    
    plain = [x for x in data if x["hotcold_config"]["cache_size_fraction"] == 1 and x["collocate_config"]["num_collocate"] == 0]
    collocate_only = [x for x in data if x["hotcold_config"]["cache_size_fraction"] == 1]
    hotcold_only = [x for x in data if x["collocate_config"]["num_collocate"] == 0]
    basic = [x for x in data if x["pir_config"]["num_bins"] == 1 and x["hotcold_config"]["cache_size_fraction"] == 1 and  x["collocate_config"]["num_collocate"] == 0]

    print("plain", len(plain))
    print("coll", len(collocate_only))
    print("hotc", len(hotcold_only))
    print("basic", len(basic))
    
    def plot_single(data, label, marker, color):
        accuracy = [x["accuracy_stats"]["auc"] for x in data]
        computation = [(x["cost"]["upload_communication"]+x["cost"]["download_communication"])/1000 for x in data]    
    

        computation, accuracy = get_pareto_points(computation, [-x for x in accuracy], reverse=False)
        accuracy = [-x for x in accuracy]

        print(label, list(zip(computation, accuracy)))        
        
        #plt.scatter(computation, accuracy, label=label, alpha=.3)
        plt.plot(computation, accuracy, label=label, markersize=15, marker=marker, alpha=1, color=color, linewidth=5)

    plot_single(plain, "batch-pir", "o", "black")
    plot_single(collocate_only, "batch-pir +c", "x", "red")
    plot_single(hotcold_only, "batch_pir +h", "^", "green")
    plot_single(data, "batch-pir +c +h", "v", "blue")

    plt.xlabel("Communication (kBytes)", fontsize=28)
    plt.ylabel("Accuracy (auc)", fontsize=28)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    #plt.yscale("log")

    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"movielens_communication_vs_auc.pdf", tight_layout=True)            

def plot_communication_vs_computation_cost(data):

    data = [x for x in data if x["accuracy_stats"]["auc"] >= .77]
    print(len(data))

    plt.cla()
    plt.clf()
    plain = [x for x in data if x["hotcold_config"]["cache_size_fraction"] == 1 and x["collocate_config"]["num_collocate"] == 0]
    collocate_only = [x for x in data if x["hotcold_config"]["cache_size_fraction"] == 1]
    hotcold_only = [x for x in data if x["collocate_config"]["num_collocate"] == 0]
    basic = [x for x in data if x["pir_config"]["num_bins"] == 1 and x["hotcold_config"]["cache_size_fraction"] == 1 and  x["collocate_config"]["num_collocate"] == 0]

    print("plain", len(plain))
    print("coll", len(collocate_only))
    print("hotc", len(hotcold_only))
    print("basic", len(basic))

    def plot_single(data, label, marker, color):
        communication = [(x["cost"]["upload_communication"]+x["cost"]["download_communication"])/1000 for x in data]
        computation = [x["cost"]["computation"]/1000 for x in data]        
        communication, computation = get_pareto_points(communication, computation)
        #plt.scatter(communication, computation, label=label)
        plt.plot(communication, computation, label=label, markersize=15, marker=marker, alpha=1, color=color, linewidth=5)

    plot_single(plain, "batch-pir", "o", "black")
    plot_single(collocate_only, "batch-pir +c", "x", "red")
    plot_single(hotcold_only, "batch_pir +h", "^", "green")
    plot_single(data, "batch-pir +c +h", "v", "blue")

    plt.xlabel("Communication (kBytes)", fontsize=28)
    plt.ylabel("Computation (kPRFs)", fontsize=28)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    #plt.yscale("log")

    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"movielens_communication_vs_computation.pdf", tight_layout=True)        

#plot_computation_cost(data)
#plot_communication_cost(data)
plot_communication_vs_computation_cost(data)
plot_computation_vs_accuracy(data)
plot_communication_vs_accuracy(data)
