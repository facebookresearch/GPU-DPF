# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.  

import sys
import os
import json
import pprint
from collections import namedtuple
import numpy as np
import movielens_dataset as batch_pir_test_dataset
#from movielens_dataset import *
#import language_model_dataset as batch_pir_test_dataset
import taobao_rec_dataset_v2 as batch_pir_test_dataset
from taobao_rec_dataset_v2 import *
import random
import pprint
import matplotlib.pyplot as plt

HotColdConfig = namedtuple('HotColdConfig', 'cache_size_fraction')
CollocateConfig = namedtuple('CollocateConfig', 'num_collocate')
PIRConfig = namedtuple('PIRConfig', 'num_bins entry_size_bytes queries_to_hot queries_to_cold')

DPFCost = namedtuple("DPFCost", 'computation upload_communication download_communication')

class BatchPIROptimize(object):
    def __init__(self, train_set, validation_set,
                 hotcold_config, collocate_config, pir_config,
                 initialize_collocate_load_from=None):
        print("Init BatchPIROptimize")

        self.hotcold_config = hotcold_config
        self.collocate_config = collocate_config
        self.pir_config = pir_config
        
        self.train = train_set
        self.val = validation_set
        self.initialize()

        # Perform hotcold splitting
        self.hot_table = set()
        self.cold_table = set()
        self.initialize_hotcold()

        # Combine w/ collocation per table
        self.initialize_collocate(load_from=initialize_collocate_load_from)

        # Initialize basic pir
        self.initialize_pir()

    def initialize_pir(self):
        num_bins = self.pir_config.num_bins

        # Num bins is actually fraction of table that represents 1 bin

        hot_table_length = len(self.hot_table)
        self.hot_table_entries_per_bin = int(hot_table_length * num_bins)
        self.hot_table_bins = [set(self.hot_table[i:i+self.hot_table_entries_per_bin]) for i in range(0, len(self.hot_table), self.hot_table_entries_per_bin)]

        cold_table_length = len(self.cold_table)
        if cold_table_length == 0:
            self.cold_table_entries_per_bin = 0
            self.cold_table_bins = []
        else:
            self.cold_table_entries_per_bin = int(cold_table_length * num_bins)
            self.cold_table_bins = [set(self.cold_table[i:i+self.cold_table_entries_per_bin]) for i in range(0, len(self.cold_table), self.cold_table_entries_per_bin)]

    def initialize_hotcold(self):
        print("initialize_hotcold...")
        cache_size_fraction = self.hotcold_config.cache_size_fraction
        self.num_embeddings_hot = int(cache_size_fraction*self.num_embeddings)
        self.num_embeddings_cold = self.num_embeddings - self.num_embeddings_hot

        embedding_indices = list(self.all_embedding_indices)
        embedding_indices.sort(key=lambda x:self.embedding_counts[x], reverse=True)

        self.hot_table = embedding_indices[:self.num_embeddings_hot]
        self.cold_table = embedding_indices[self.num_embeddings_hot:]

        self.hot_table.sort(key=lambda x: hash(str(x)))
        self.cold_table.sort(key=lambda x: hash(str(x)))

        assert(len(self.hot_table) + len(self.cold_table) == self.num_embeddings)

        self.accuracy_stats = None

    def DPF_COST(self, table_size):
        if table_size == 0:
            return 0
        return int(np.ceil((128//8) * 4 * np.log2(table_size)))

    def evaluate(self):
        self.percentage_of_query_recovered = []
        for i, val in enumerate(self.val):
            if len(val) == 0:
                continue
            if i % 1000 == 0:
                print(f"evaluate: {i}/{len(self.val)}")
            #print(val, self.fetch(val))
            recovered, self.cost = self.fetch(val)
            recovered = [x for x in recovered if x in val]
            p_recovered = len(set(recovered)) / len(set(val))
            self.percentage_of_query_recovered.append(p_recovered)

    def evaluate_small(self):
        self.percentage_of_query_recovered = []
        for i, val in enumerate(self.val):
            if i % 1000 == 0:
                print(f"evaluate: {i}/{len(self.val)}")
                break
            recovered, self.cost = self.fetch(val)
            recovered = [x for x in recovered if x in val]
            p_recovered = len(set(recovered)) / len(set(val))
            self.percentage_of_query_recovered.append(p_recovered)

    def evaluate_real(self):
        # Evaluate + do it for real with real infrence to collect acc
        self.evaluate()
        self.accuracy_stats = batch_pir_test_dataset.evaluate(self)
        print(self.accuracy_stats)

    def summarize_evaluation(self):
        summary = {}
        summary["pir_config"] = self.pir_config._asdict()
        summary["hotcold_config"] = self.hotcold_config._asdict()
        summary["collocate_config"] = self.collocate_config._asdict()
        summary["mean_recovered"] = np.mean(self.percentage_of_query_recovered)
        summary["recovered_p_50"] = np.percentile(self.percentage_of_query_recovered, 50)
        summary["recovered_p_90"] = np.percentile(self.percentage_of_query_recovered, 90)
        summary["recovered_p_95"] = np.percentile(self.percentage_of_query_recovered, 95)
        summary["recovered_p_10"] = np.percentile(self.percentage_of_query_recovered, 10)
        summary["recovered_p_5"] = np.percentile(self.percentage_of_query_recovered, 5)
        summary["recovered_p_0"] = np.percentile(self.percentage_of_query_recovered, 0)        
        summary["cost"] = self.cost._asdict()
        summary["accuracy_stats"] = self.accuracy_stats
        summary["extra"] = {
            "hot_table_size" : self.num_embeddings_hot,
            "cold_table_size" : self.num_embeddings_cold,
            "hot_table_entries_per_bin" : self.hot_table_entries_per_bin,
            "cold_table_entries_per_bin" : self.cold_table_entries_per_bin
        }
        
        pprint.pprint(summary)
        return summary

    def fetch(self, batch_indices):
        # Return:
        # - indices that are fetched
        # - Computation / Communication cost

        # Sort indices to recover by frequency (so that we recover as much as possible)
        target_indices_counts = {}
        for indx in batch_indices:
            if indx not in target_indices_counts:
                target_indices_counts[indx] = 0
            target_indices_counts[indx] += 1
        target_indices = set(target_indices_counts.keys())
            
        indices_recovered = set()

        def single_query(table, indices_recovered):
            for i, embedding_bin in enumerate(table):
                embedding_recovery_candidates = embedding_bin.intersection(target_indices)
                if len(embedding_recovery_candidates) == 0:
                    continue
                
                # Can only retrieve one per bin. Choose one that:
                # - Has not been recovered
                # - Has biggest index count
                embedding_recovery_candidates = sorted(list(embedding_recovery_candidates), key=lambda x:float("-inf") if x in indices_recovered else target_indices_counts[x], reverse=True)
                embedding_to_fetch = embedding_recovery_candidates[0]

                indices_recovered.add(embedding_to_fetch)

        for i in range(self.pir_config.queries_to_hot):
            single_query(self.hot_table_bins, indices_recovered)
        for i in range(self.pir_config.queries_to_cold):
            single_query(self.cold_table_bins, indices_recovered)

        # Collocation
        collocated_indices = []
        for indx in indices_recovered:
            collocated_indices += self.embedding_collocation_map[indx]
        collocated_indices = set(collocated_indices)

        all_recovered_indices = collocated_indices | indices_recovered

        # Compute computation and communication cost
        computation_cost = self.pir_config.queries_to_hot * len(self.hot_table) + self.pir_config.queries_to_cold * len(self.cold_table)
        upload_communication_cost = (
            self.pir_config.queries_to_hot * self.DPF_COST(self.hot_table_entries_per_bin) * len(self.hot_table_bins) +
            self.pir_config.queries_to_cold * self.DPF_COST(self.cold_table_entries_per_bin) * len(self.cold_table_bins))
        download_communication_cost = (
            self.pir_config.queries_to_hot * len(self.hot_table_bins) * self.pir_config.entry_size_bytes +
            self.pir_config.queries_to_cold * len(self.cold_table_bins) * self.pir_config.entry_size_bytes
        )

        return all_recovered_indices, DPFCost(computation_cost, upload_communication_cost, download_communication_cost)

    def initialize_collocate(self, load_from="initialize_collocate.json", save_to="initialize_collocate.json"):

        #load_from = None
        if load_from is not None:
            if type(load_from) == type({}):
                cumulative = load_from
            if type(load_from) == str and os.path.exists(load_from):
                with open(load_from, "r") as f:
                    cumulative = json.load(f)
            self.embedding_collocation_counts = cumulative["self.embedding_collocation_counts"]
            self.embedding_collocation_map = cumulative["self.embedding_collocation_map"]

            self.embedding_collocation_counts = {int(k):v for k,v in self.embedding_collocation_counts.items()}
            self.embedding_collocation_map = {int(k):v for k,v in self.embedding_collocation_map.items()}                
            return            
        
        print("initialize_collocate...")
        self.embedding_collocation_counts = {}        

        # Tally up indices that are close to each other
        for ii, index_set in enumerate(self.train):
            if ii % 1000 == 0:
                print(f"{ii}/{len(self.train)}")
            for src_indx in index_set:
                for dst_indx in index_set:
                    if src_indx == dst_indx:
                        continue
                    if src_indx not in self.embedding_collocation_counts:
                        self.embedding_collocation_counts[src_indx] = {}
                    if dst_indx not in self.embedding_collocation_counts[src_indx]:
                        self.embedding_collocation_counts[src_indx][dst_indx] = 0
                    self.embedding_collocation_counts[src_indx][dst_indx] += 1

        self.embedding_collocation_map = {}
        for indx in self.all_embedding_indices:
            if indx not in self.embedding_collocation_counts:
                self.embedding_collocation_map[indx] = []
                continue
            neighbor_indices = self.embedding_collocation_counts[indx].keys()
            neighbor_index_counts = self.embedding_collocation_counts[indx]            
            best_neighbor_indices = sorted(neighbor_indices, key=lambda x: neighbor_index_counts[x], reverse=True)
            selected_neighbor_indices = best_neighbor_indices[:self.collocate_config.num_collocate]
            self.embedding_collocation_map[indx] = selected_neighbor_indices

        if save_to is not None:
            with open(save_to, "w") as f:
                cumulative = {
                    "self.embedding_collocation_map" : self.embedding_collocation_map,
                    "self.embedding_collocation_counts" : self.embedding_collocation_counts
                }
                json.dump(cumulative, f)

    def initialize(self):
        print("initialize...")        
        self.embedding_counts = {}
        
        for index_set in self.train:
            for index in index_set:
                if index not in self.embedding_counts:
                    self.embedding_counts[index] = 0
                self.embedding_counts[index] += 1

        # also track all embeddings
        self.all_embedding_indices = set()
        for index_set in self.train + self.val:
            for indx in index_set:
                self.all_embedding_indices.add(indx)
                if indx not in self.embedding_counts:
                    self.embedding_counts[indx] = 0
        self.num_embeddings = len(self.all_embedding_indices)

if __name__=="__main__":
    batch_pir_test_dataset.initialize()    
    hotcold_config = HotColdConfig(1)
    collocate_config = CollocateConfig(0)
    pir_config = PIRConfig(30, 256, 4, 0)
    
    #b = BatchPIROptimize(batch_pir_test_dataset.train_access_pattern,
    #                     batch_pir_test_dataset.val_access_pattern,
    #                     hotcold_config, collocate_config, pir_config,
    #                     initialize_collocate_load_from="initialize_collocate_language_model.json")

    #b = BatchPIROptimize(batch_pir_test_dataset.train_access_pattern,
    #                     batch_pir_test_dataset.val_access_pattern,
    #                     hotcold_config, collocate_config, pir_config,
    #                     initialize_collocate_load_from="initialize_collocate_movielens.json")

    #b = BatchPIROptimize(batch_pir_test_dataset.train_access_pattern,
    #                     batch_pir_test_dataset.val_access_pattern,
    #                     hotcold_config, collocate_config, pir_config,
    #                     initialize_collocate_load_from="initialize_collocate_taobao.json")

    b = BatchPIROptimize(batch_pir_test_dataset.train_access_pattern,
                         batch_pir_test_dataset.val_access_pattern,
                         hotcold_config, collocate_config, pir_config,
                         initialize_collocate_load_from="initialize_collocate_taobao.json")    
    

    b.evaluate_real()
    #b.evaluate()
    b.summarize_evaluation()
