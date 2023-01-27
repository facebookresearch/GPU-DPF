# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.  

import sys
import pprint
import numpy as np
import matplotlib.pyplot as plt
import glob
import json

app_name = sys.argv[1]

def load_data(files):
    data = []
    for fname in files:
        with open(fname, "r") as f:
            data.append(json.load(f))
    return data

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def get_pareto_points(xs, ys):
    points = list(zip(xs, ys))
    is_efficient = is_pareto_efficient_simple(np.array(points))
    selected = [x for i,x in enumerate(points) if is_efficient[i]]
    selected.sort(key=lambda x:x[0])
    return [x[0] for x in selected], [x[1] for x in selected]

def plot_accuracy_vs_throughput(data):
    plt.cla()
    plt.clf()

    # Requirements: < 100 ms processing latency    
    # Requirements: < 300 KB Communication Cost
    data = [x for x in data if x["cost"]["upload_communication"] + x["cost"]["download_communication"] <= 100000]
    print(len(data))
    #data = [x for x in data if x["accuracy_stats"]["auc"] >= .50]
    accuracy_baseline = max([x["accuracy_stats"]["auc"] for x in data])   

    plain = [x for x in data if x["hotcold_config"]["cache_size_fraction"] == 1 and x["collocate_config"]["num_collocate"] == 0]

    for d in data:
        continue
        pprint.pprint(d)
        sys.stdin.readline()
        
    plt.axhline(accuracy_baseline)

    def plot_single(data, label, marker, color):

        all_accuracies = []
        all_throughputs = []

        for d in data:
            for latency, throughput in d["latency_throughputs"]:
                if latency < 100:
                    all_accuracies.append(d["accuracy_stats"]["auc"])
                    all_throughputs.append(throughput)
                    #all_throughputs.append(d["cost"]["computation"])

        
        #print(list(set(all_accuracies)))
        #sys.exit(0)

        #print(list(set(all_throughputs)))
        
        #zipped = list(zip(all_throughputs, all_accuracies))
        #zipped.sort(key=lambda x:x[0])
        #print(zipped[:10])
        #sys.exit(0)

        """
        all_throughputs_accuracies = list(zip(all_throughputs, all_accuracies))
        np.random.shuffle(all_throughputs_accuracies)
        all_throughputs = [x[0] for x in all_throughputs_accuracies]
        all_accuracies = [x[1] for x in all_throughputs_accuracies]
        all_throughputs = all_throughputs[:1000]
        all_accuracies = all_accuracies[:1000]        
        print(sorted(list(zip(all_throughputs, all_accuracies))))
        """

        #"""
        all_accuracies = [-x for x in all_accuracies]
        all_throughputs = [-x for x in all_throughputs]        
        print(max(all_throughputs))
        all_throughputs, all_accuracies = get_pareto_points(all_throughputs, all_accuracies)
        all_throughputs = [-x for x in all_throughputs]                
        all_accuracies = [-x for x in all_accuracies]
        #all_throughputs = [-x for x in all_throughputs]        
        plt.plot(all_throughputs, all_accuracies, label=label, markersize=15, marker=marker, alpha=1, color=color, linewidth=5)
        #"""

        #print(list(zip(all_throughputs, all_accuracies)))

        print(len(all_throughputs))        
        #plt.scatter(all_throughputs, all_accuracies, label=label, marker=marker, alpha=1, color=color)

    plot_single(plain, "batch-pir", "o", "black")
    plot_single(data, "batch-pir w/ co-design", "x", "blue")    

    plt.xlabel("Throughput (q/ms)", fontsize=28)
    plt.ylabel("Accuracy (auc)", fontsize=28)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.legend(loc="best", fontsize=14);

    _, max_number = plt.xlim()
    #plt.xticks(np.arange(0, max_number, 5)) 

    plt.tight_layout()

    plt.savefig(f"{app_name}_throughpt_auc.pdf", tight_layout=True)
    

d_in = sys.argv[1]
data = load_data(glob.glob(f"{d_in}/*"))

plot_accuracy_vs_throughput(data)

