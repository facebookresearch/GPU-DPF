# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.  

# Generates optimized table entry layouts for batch pir (e.g: partial batch retrieval).
#
# Context:
# - Batch pir "bins" together embeddings, then issues individual DPFs per bin
# - However, this process may fail to obtain DPFs if they fall into the same bin
# - How to minimize fetch failures?
#
# Idea:
# - Optimize where each entry is placed; place entries that occur together frequently in separate bins
# - This can be done using a hypergraph optimization process
#
# Specifics:
# - Each embedding w/ index i:0-N is binned into bin number i/num_bins. These are fixed

import sys
import numpy as np
import batch_pir_test_dataset
#import batch_pir_taobao_dataset as batch_pir_test_dataset
import random
import pprint
import matplotlib.pyplot as plt

batch_pir_test_dataset.initialize()

def dpf_communication_cost(table_size):
    return np.ceil(np.log2(table_size))*2*2*128/8

def plot_assignment_evaluation_fixed_recovery_rate_hotcold(data):
    # x-axis cache size, y-axis computation improvement
    xs = []
    ys = []
    for d in data:
        stats = d["data"]
        xs.append(stats["cache_ratio"])
        ys.append(stats["computation_gain_over_naive"])
        
    plt.clf()
    plt.plot(xs, ys, marker="o", markersize=5)
    plt.xlabel("Cache size (ratio of total table size)")
    plt.ylabel("Computation ratio improvement over naive")
    plt.savefig("hotcold_fixed_recovery_rate_%f.pdf" % data[0]["data"]["target_recovery_rate"])

def plot_assignment_evaluation_fixed_cache_ratio_hotcold(data):
    # x-axis cache size, y-axis computation improvement
    xs = []
    ys = []
    for d in data:
        stats = d["data"]
        xs.append(stats["target_recovery_rate"])
        ys.append(stats["computation_gain_over_naive"])
        
    plt.clf()
    plt.plot(xs, ys, marker="o", markersize=5)
    plt.xlabel("Recovery rate")
    plt.ylabel("Computation ratio improvement over naive")
    plt.savefig("hotcold_fixed_cache_ratio_%f.pdf" % data[0]["data"]["cache_ratio"])
    

def plot_evaluation_comparison(with_collocate, without):

    plt.clf()
    for ev in with_collocate:
        name = ev["name"]
        data = ev["data"]
        cdf = sorted(list(data["average_percent_recovered_after_num_dpfs"].items()), key=lambda x:x[0])
        compute_costs = [data["per_dpf_compute_cost"]*x[0] for x in cdf]
        communication_costs = [data["per_dpf_send_bytes"]*x[0] for x in cdf]
        percentage_recovered = [x[1] for x in cdf]

        if "35" not in name:
            continue
        
        plt.plot(percentage_recovered, compute_costs, label=name, marker="o", markersize=5)

    for ev in without:
        name = ev["name"]
        data = ev["data"]
        cdf = sorted(list(data["average_percent_recovered_after_num_dpfs"].items()), key=lambda x:x[0])
        compute_costs = [data["per_dpf_compute_cost"]*x[0] for x in cdf]
        communication_costs = [data["per_dpf_send_bytes"]*x[0] for x in cdf]
        percentage_recovered = [x[1] for x in cdf]

        if "35" not in name:
            continue        
        
        plt.plot(percentage_recovered, compute_costs, label=name, marker="o", markersize=5, linestyle="dashed")        

    plt.legend(loc="best")
    plt.ylabel("Compute cost")
    plt.xlabel("Percentage of Indices in Query Recovered")
    #plt.yscale("log")
    plt.xscale("log")
    plt.savefig("batch_pir_assignment_evaluation_compute_cmp_naive_collocate.pdf")
    

    plt.clf()

    ######### Pareto curve cmp
    
    target_percentages = [.99, .98, .95, .9]
    for perc in target_percentages:
        plt.clf()        
        pareto = {}
        
        for ev in with_collocate:
            name = ev["name"]
            data = ev["data"]
            cdf = sorted(list(data["average_percent_recovered_after_num_dpfs"].items()), key=lambda x:x[0])
            compute_costs = [data["per_dpf_compute_cost"]*x[0] for x in cdf]
            communication_costs = [data["per_dpf_send_bytes"]*x[0] for x in cdf]
            percentage_recovered = [x[1] for x in cdf]
            datum = list(zip(compute_costs, communication_costs, percentage_recovered))

            datum_p = [x for x in datum if x[-1] >= perc]
            datum_p = datum_p[0]
            if perc not in pareto:
                pareto[perc] = {"xs":[], "ys":[], "labels":[]}

            pareto[perc]["xs"].append(datum_p[0])
            pareto[perc]["ys"].append(datum_p[1])
            pareto[perc]["labels"].append(name)

        pareto_naive = {}
        for ev in without:
            name = ev["name"]
            data = ev["data"]
            cdf = sorted(list(data["average_percent_recovered_after_num_dpfs"].items()), key=lambda x:x[0])
            compute_costs = [data["per_dpf_compute_cost"]*x[0] for x in cdf]
            communication_costs = [data["per_dpf_send_bytes"]*x[0] for x in cdf]
            percentage_recovered = [x[1] for x in cdf]
            datum = list(zip(compute_costs, communication_costs, percentage_recovered))

            datum_p = [x for x in datum if x[-1] >= perc]
            datum_p = datum_p[0]
            if perc not in pareto_naive:
                pareto_naive[perc] = {"xs":[], "ys":[], "labels":[]}

            pareto_naive[perc]["xs"].append(datum_p[0])
            pareto_naive[perc]["ys"].append(datum_p[1])
            pareto_naive[perc]["labels"].append(name)        

        for p, d in pareto.items():
            plt.plot(d["xs"], d["ys"], marker="o", markersize=5, label="%f recovered (with collocation)"% p)

            for i,(x,y) in enumerate(zip(d["xs"], d["ys"])):
                plt.text(x,y, d["labels"][i])

        for p, d in pareto_naive.items():
            plt.plot(d["xs"], d["ys"], marker="o", markersize=5, label="%f recovered (naive)"% p)

            for i,(x,y) in enumerate(zip(d["xs"], d["ys"])):
                plt.text(x,y, d["labels"][i])
                
        plt.legend(loc="best")
        plt.ylabel("DPF Commmunication cost")
        plt.xlabel("DPF Computation cost")
        #plt.yscale("log")
        #plt.xscale("log")
        plt.savefig("batch_pir_assignment_evaluation_compute_vs_communication_cmp_naive_collocate_p=%f.pdf" % (perc))
    

def plot_assignment_evaluation(evals, plt_suffix=""):
    
    plt.clf()
    for ev in evals:
        name = ev["name"]
        data = ev["data"]
        cdf = sorted(list(data["average_percent_recovered_after_num_dpfs"].items()), key=lambda x:x[0])
        compute_costs = [data["per_dpf_compute_cost"]*x[0] for x in cdf]
        communication_costs = [data["per_dpf_send_bytes"]*x[0] for x in cdf]
        percentage_recovered = [x[1] for x in cdf]

        plt.plot(percentage_recovered, compute_costs, label=name, marker="o", markersize=5)

    plt.legend(loc="best")
    plt.ylabel("Compute cost")
    plt.xlabel("Percentage of Indices in Query Recovered")
    #plt.yscale("log")
    plt.xscale("log")
    plt.savefig("batch_pir_assignment_evaluation_compute_%s.pdf" % plt_suffix)

    plt.clf()
    for ev in evals:
        name = ev["name"]
        data = ev["data"]
        cdf = sorted(list(data["average_percent_recovered_after_num_dpfs"].items()), key=lambda x:x[0])
        compute_costs = [data["per_dpf_compute_cost"]*x[0] for x in cdf]
        communication_costs = [(data["per_dpf_send_bytes"] + data["per_dpf_receive_bytes"])*x[0] for x in cdf]
        percentage_recovered = [x[1] for x in cdf]

        plt.plot(percentage_recovered, communication_costs, label=name, marker="o", markersize=5)

    plt.legend(loc="best")
    plt.ylabel("Commmunication cost")
    plt.xlabel("Percentage of Indices in Query Recovered")
    #plt.yscale("log")
    plt.xscale("log")
    plt.savefig("batch_pir_assignment_evaluation_communication_%s.pdf" % plt_suffix)

    plt.clf()
    target_percentages = [.99, .98, .95, .9]
    pareto = {}
    for ev in evals:
        name = ev["name"]
        data = ev["data"]
        cdf = sorted(list(data["average_percent_recovered_after_num_dpfs"].items()), key=lambda x:x[0])
        compute_costs = [data["per_dpf_compute_cost"]*x[0] for x in cdf]
        communication_costs = [(data["per_dpf_send_bytes"] + data["per_dpf_receive_bytes"])*x[0] for x in cdf]
        percentage_recovered = [x[1] for x in cdf]
        datum = list(zip(compute_costs, communication_costs, percentage_recovered))

        for p in target_percentages:
            datum_p = [x for x in datum if x[-1] >= p]
            datum_p = datum_p[0]
            if p not in pareto:
                pareto[p] = {"xs":[], "ys":[], "labels":[]}
                
            pareto[p]["xs"].append(datum_p[0])
            pareto[p]["ys"].append(datum_p[1])
            pareto[p]["labels"].append(name)

    for p, d in pareto.items():
        plt.plot(d["xs"], d["ys"], marker="o", markersize=5, label="%f recovered"% p)
        
        for i,(x,y) in enumerate(zip(d["xs"], d["ys"])):
            plt.text(x,y, d["labels"][i])

    plt.legend(loc="best")
    plt.ylabel("DPF Commmunication cost")
    plt.xlabel("DPF Computation cost")
    #plt.yscale("log")
    #plt.xscale("log")
    plt.savefig("batch_pir_assignment_evaluation_compute_vs_communication_%s.pdf" % plt_suffix)

    plt.clf()

class BatchPirOptimizeHotCold(object):

    def __init__(self, train_data, val_data, num_embeddings, p=.10, target_recovery_rate=.95, entry_bytes=128):
        self.train_data = train_data
        self.val_data = val_data
        self.num_embeddings = num_embeddings
        self.entry_bytes = entry_bytes

        # p = relative size of hot vs cold table
        self.p = p
        self.target_recovery_rate = target_recovery_rate

        self.hot_table_size = int(num_embeddings*p)
        self.cold_table_size = num_embeddings - self.hot_table_size

        # Separate indices into hot or cold table by frequency of access
        embedding_counts = self.count_embedding_accesses(train_data)
        sorted_embeddings_by_count = sorted(embedding_counts.items(), key=lambda x:x[1], reverse=True)
        self.hot_table = set([x[0] for x in sorted_embeddings_by_count[:self.hot_table_size]])
        self.cold_table = set([x[0] for x in sorted_embeddings_by_count[self.hot_table_size:]])

        # Obtain the upper bound on number of queries to obtain all indices
        n_query_upper_bound = max([len(d) for d in train_data])
        self.n_query_upper_bound = n_query_upper_bound

        # Obtain best number of queries to hit target_recovery_rate but also minimize
        # accesses to hot/cold tables
        candidates = []
        for queries_to_cold_table in range(n_query_upper_bound):
            for queries_to_hot_table in range(n_query_upper_bound):
                stats = self.evaluate_assignment(train_data, queries_to_hot_table, queries_to_cold_table)
                if stats is None:
                    continue
                if stats["average_p_recovered"] >= self.target_recovery_rate:
                    candidates.append(stats)
                    pprint.pprint(candidates[0])
                    
            candidates.sort(key=lambda x:x["computation_cost"])
            print(candidates[0])

            # Just return first result, which is pretty good
            if len(candidates) > 0:
                break
            
        self.evaluation_stats = candidates[0]

    def count_embedding_accesses(self, data):
        counts = {}
        for d in data:
            for indx in d:
                if indx not in counts:
                    counts[indx] = 0
                counts[indx] += 1
        return counts

    def evaluate_assignment(self, data,
                            queries_to_hot_table,
                            queries_to_cold_table):
        if queries_to_hot_table + queries_to_cold_table == 0:
            return None
        average_portion_recovered = []
        for ii, d in enumerate(data):
            indices_that_are_hot = set([x for x in d if x in self.hot_table])
            indices_that_are_cold = set([x for x in d if x in self.cold_table])
            n_recovered = min(queries_to_hot_table, len(indices_that_are_hot)) + min(len(indices_that_are_cold), queries_to_cold_table)
            total = len(indices_that_are_hot) + len(indices_that_are_cold)
            p_recovered = n_recovered/total

            average_portion_recovered.append(p_recovered)

        stats = {
            "per_dpf_send_bytes" : queries_to_hot_table*dpf_communication_cost(len(self.hot_table)) + queries_to_cold_table*dpf_communication_cost(len(self.cold_table)),
            "computation_cost" : queries_to_hot_table*len(self.hot_table) + queries_to_cold_table*len(self.cold_table),
            "average_p_recovered" : np.mean(average_portion_recovered),
            "naive_computation_cost" : np.ceil(self.target_recovery_rate*self.n_query_upper_bound)*(self.num_embeddings),
            "cache_ratio" : self.p,
            "target_recovery_rate" : self.target_recovery_rate,
        }
        stats["computation_gain_over_naive"] = stats["naive_computation_cost"] / stats["computation_cost"]
        return stats
    
class BatchPirOptimizeCollocate(object):
    def __init__(self, train_data, val_data, num_embeddings, num_bins, entry_bytes=128):

        self.entry_bytes = entry_bytes
        self.num_bins = num_bins
        self.num_embeddings = num_embeddings
        self.train_data = train_data
        self.val_data = val_data

        self.embedding_count = self.count_embedding_accesses(self.train_data)        
        
        self.entries_per_bin = num_embeddings // num_bins
        self.naive_assignment = {i:min((hash(str(i))%num_embeddings)//self.entries_per_bin, self.num_bins-1)
                                 for i in range(num_embeddings)}        
        self.naive_assignment_evaluation = self.evaluate_assignment(self.naive_assignment, val_data)

        # Collocation assignment
        self.conflict_stats = self.collect_conflict_data(self.naive_assignment, train_data)
        self.evaluation_with_collocation = self.evaluate_assignment(self.naive_assignment, val_data, self.conflict_stats)

    def count_embedding_accesses(self, data):
        counts = {}
        for d in data:
            for indx in d:
                if indx not in counts:
                    counts[indx] = 0
                counts[indx] += 1
        return counts        

    def collect_conflict_data(self, assignment, dataset):
        conflict_stats = {i:{} for i in range(self.num_embeddings)}
        for d in dataset:
            bins = {}
            for indx in d:
                if assignment[indx] not in bins:
                    bins[assignment[indx]] = []
                bins[assignment[indx]].append(indx)

            for b, conflicting_indices in bins.items():
                for i in conflicting_indices:
                    for j in conflicting_indices:
                        if j not in conflict_stats[i]:
                            conflict_stats[i][j] = 0
                        conflict_stats[i][j] += 1
        conflict_stats = {i:sorted([(m,n) for m,n in v.items()], key=lambda x:x[1], reverse=True) for i, v in conflict_stats.items()}
        conflict_stats = {i:[(x,y) for x,y in v if x != i] for i,v in conflict_stats.items()}
        return conflict_stats
                
    def evaluate_assignment(self, assignment, dataset, collocation_stats=None):
        
        # Calculate how many "partial batch retrievals" are required to
        # completely fetch each batch of accesses
        pbr_stats = {
            "num_bins" : self.num_bins,
            "num_embeddings" : self.num_embeddings,
            "per_dpf_send_bytes" : dpf_communication_cost(self.entries_per_bin),
            "per_dpf_compute_cost" : self.entries_per_bin,
            "one_table_compute_cost" : self.num_embeddings,
            "entries_per_bin" : self.entries_per_bin,
            "entry_bytes" : self.entry_bytes,
        }
        pbr_stats["per_dpf_receive_bytes"] = self.entry_bytes
        if collocation_stats is not None:
            pbr_stats["per_dpf_receive_bytes"] *= 4

        pbrs_to_obtain_all_batch_elements = []
        pbr_rounds_to_obtain_all_batch_elements = []
        pbr_average_percent_recovered_after_num_dpfs = {}
        for d in dataset:
            covered = set()
            num_pbrs = 0
            num_pbr_rounds_to_obtain_all_batch_elements = 0

            # While  we haven't obtained every index
            while len(covered.intersection(set(d))) != len(set(d)):
                #print("--")
                num_pbr_rounds_to_obtain_all_batch_elements += 1
                
                # For each bin, find an indx in the batch
                # to obtain from the bin
                for b in range(self.num_bins):
                    num_pbrs += 1

                    # Validation may see new id, these have 0 counts
                    for access in d:
                        if access not in self.embedding_count:
                            self.embedding_count[access] = 0
                            
                    if collocation_stats is not None:
                        indices = sorted(d, key=lambda x:self.embedding_count[x])
                    else:
                        indices = sorted(d, key=lambda x:random.random())
                        
                    for indx in indices:
                        if indx not in covered and assignment[indx] == b:
                            covered.add(indx)
                            #print("Add", indx, assignment[indx])
                            
                            if collocation_stats is not None:
                                if len(collocation_stats[indx]) >= 1:
                                    covered.add(collocation_stats[indx][0][0])
                                if len(collocation_stats[indx]) >= 2:
                                    covered.add(collocation_stats[indx][1][0])
                                if len(collocation_stats[indx]) >= 3:
                                    covered.add(collocation_stats[indx][2][0])
                                #if len(collocation_stats[indx]) >= 4:
                                #    covered.add(collocation_stats[indx][3][0])
                                #if len(collocation_stats[indx]) >= 5:
                                #    covered.add(collocation_stats[indx][4][0])                                                                        
                            break                    

                percent_recovered = len(covered.intersection(set(d))) / len(set(d))
                if num_pbrs not in pbr_average_percent_recovered_after_num_dpfs:
                    pbr_average_percent_recovered_after_num_dpfs[num_pbrs] = []
                pbr_average_percent_recovered_after_num_dpfs[num_pbrs].append(percent_recovered)

            #if collocation_stats is not None:
            #    sys.exit(0)
            pbrs_to_obtain_all_batch_elements.append(num_pbrs)
            pbr_rounds_to_obtain_all_batch_elements.append(num_pbr_rounds_to_obtain_all_batch_elements)
            
        pbr_stats["mean_pbrs_to_obtain_all_batch_elements"] = np.mean(pbrs_to_obtain_all_batch_elements)
        pbr_stats["mean_dpf_send_bytes"] = np.mean(pbrs_to_obtain_all_batch_elements)*pbr_stats["per_dpf_send_bytes"]
        pbr_stats["mean_dpf_compute_cost"] = np.mean(pbrs_to_obtain_all_batch_elements)*pbr_stats["per_dpf_compute_cost"]
        pbr_stats["mean_pbr_rounds_to_obtain_all_batch_elements"] = np.mean(pbr_rounds_to_obtain_all_batch_elements)
        pbr_stats["average_percent_recovered_after_num_dpfs"] = {k:np.mean(v) for k,v in pbr_average_percent_recovered_after_num_dpfs.items()}
        pprint.pprint(pbr_stats)

        return pbr_stats
        
def benchmark_collocation():
    num_embeddings = batch_pir_test_dataset.num_embeddings
    a = BatchPirOptimizeCollocate(batch_pir_test_dataset.train_access_pattern,
                         batch_pir_test_dataset.val_access_pattern,
                                  num_embeddings, 35, 5000) 
    b = BatchPirOptimizeCollocate(batch_pir_test_dataset.train_access_pattern,
                         batch_pir_test_dataset.val_access_pattern,
                         num_embeddings, 15, 5000)
    c = BatchPirOptimizeCollocate(batch_pir_test_dataset.train_access_pattern,
                         batch_pir_test_dataset.val_access_pattern,
                         num_embeddings, 10, 5000)
    d = BatchPirOptimizeCollocate(batch_pir_test_dataset.train_access_pattern,
                         batch_pir_test_dataset.val_access_pattern,
                         num_embeddings, 5, 5000)
    e = BatchPirOptimizeCollocate(batch_pir_test_dataset.train_access_pattern,
                         batch_pir_test_dataset.val_access_pattern,
                         num_embeddings, 1, 5000)    

    data = [{"name": "35_bins_naive", "data": a.naive_assignment_evaluation},
            {"name": "15_bins_naive", "data": b.naive_assignment_evaluation},
            {"name": "10_bins_naive", "data": c.naive_assignment_evaluation},
            {"name": "5_bins_naive", "data": d.naive_assignment_evaluation},
            {"name": "1_bins_naive", "data": e.naive_assignment_evaluation}]

    data_coll = [{"name": "35_bins_collocate", "data": a.evaluation_with_collocation},
                 {"name": "15_bins_collocate", "data": b.evaluation_with_collocation},
                 {"name": "10_bins_collocate", "data": c.evaluation_with_collocation},
                 {"name": "5_bins_collocate", "data": d.evaluation_with_collocation},
                 {"name": "1_bins_collocate", "data": e.evaluation_with_collocation}]

    plot_evaluation_comparison(data_coll, data)    
    plot_assignment_evaluation(data, plt_suffix="naive")
    plot_assignment_evaluation(data_coll, plt_suffix="collocate")    
    #print(batch_pir_test_dataset.wordify([b.top_K_embeddings]))

def benchmark_hot_cold_splitting():
    num_embeddings = batch_pir_test_dataset.num_embeddings        

    def create_batch_pir_opt(p, recovery_rate):
        return BatchPirOptimizeHotCold(batch_pir_test_dataset.train_access_pattern,
                                       batch_pir_test_dataset.val_access_pattern,
                                       num_embeddings,
                                       p=p, target_recovery_rate=recovery_rate,
                                       entry_bytes=128)
        
    
    data_hotcold = [{"name": "cache_size=%f,recover=%f" % (p,r), "data" : create_batch_pir_opt(p, r).evaluation_stats}
                    for (p,r) in [(.01, .95), (.03, .95), (.08, .95), (.1, .95), (.15, .95)]]

    plot_assignment_evaluation_fixed_recovery_rate_hotcold(data_hotcold)

    data_hotcold = [{"name": "cache_size=%f,recover=%f" % (p,r), "data" : create_batch_pir_opt(p, r).evaluation_stats}
                    for (p,r) in [(.1, .5), (.1, .8), (.1, .9), (.1, .95), (.1, .98)]]
    
    plot_assignment_evaluation_fixed_cache_ratio_hotcold(data_hotcold)
    
    
if __name__=="__main__":    
    #benchmark_collocation()
    benchmark_hot_cold_splitting()
