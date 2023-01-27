# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import sys
import tqdm
import scipy
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

train_access_pattern = []
val_access_pattern = []

SAMPLE_LIMIT = 100000000
#SAMPLE_LIMIT = 10000

user_features_processed = None
user_mappings = None
ad_features_process = None
ad_mappings = None
user_click_history = {}
train_dataset = []
val_dataset = []

embedding_table_lengths = []

split = .8

class RecModel(torch.nn.Module):

    def __init__(self, embedding_table_lengths, em_size=16):
        super().__init__()

        # First embedding table index is the ads embedding 
        self.embedding_table_lengths = embedding_table_lengths
        self.tables = torch.nn.ModuleList([torch.nn.EmbeddingBag(x+1, em_size, mode="sum") for x in embedding_table_lengths])

        self.em_size = em_size

        self.fc1 = torch.nn.Linear(241, 200)
        self.fc2 = torch.nn.Linear(200, 80)
        self.fc3 = torch.nn.Linear(80, 2)        

    def forward(self, sparse_features, dense_features, user_click_history):

        num_embs = sparse_features.shape[1]
        indices = [1+sparse_features[:,i].reshape((-1, 1)) for i in range(num_embs)]
        embs = [self.tables[i](x) for i,x in enumerate(indices)]

        # Make sure index 0 of the embeddings table for ads is 0
        self.tables[0].weight.data[0,:] = 0
        user_click_history_embeddings = self.tables[0](user_click_history+1)

        features = torch.cat(embs + [user_click_history_embeddings] + [dense_features], dim=1)

        x = features

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

def process_features_list(ds):
    columns = len(ds[0])
    rows = len(ds)
    mappings = [{} for i in range(columns)]
    for column in range(columns):
        idx = 0
        for row in range(rows):
            if ds[row][column] not in mappings[column]:
                mappings[column][ds[row][column]] = idx
                idx += 1
        print(idx)

    processed = {}
    for row in ds:
        key = mappings[0][row[0]]
        processed_row = [mappings[i][x] for i,x in enumerate(row)]
        assert(key not in processed)
        processed[key] = processed_row
    return processed, mappings

def initialize():

    # Read ad features
    # adgroup_id,cate_id,campaign_id,customer,brand,price
    ad_features_raw = []
    with open("data/taobao/ad_feature.csv", "r") as f:
        all_lines = f.readlines()[1:]
        for i, line in enumerate(all_lines):
            line = line.split(",")
            vals = [int(line[0]),
                    int(line[1]),
                    int(line[2]),
                    int(line[3]),
                    0 if line[4].strip() == "NULL" else int(line[4]),
                    float(line[5])]
            ad_features_raw.append(vals)

    # Read user features
    # userid,cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level
    user_features_raw = []
    with open("data/taobao/user_profile.csv", "r") as f:
        all_lines = f.readlines()[1:]
        for i, line in enumerate(all_lines):
            line = line.split(",")
            vals = [0 if x.strip() == "" else int(x) for x in line]
            user_features_raw.append(vals)

    global user_features_processed
    global user_mappings
    global ad_features_processed
    global ad_mappings
    user_features_processed, user_mappings = process_features_list(user_features_raw)
    ad_features_processed, ad_mappings = process_features_list(ad_features_raw)

    # Read ad click user access pattern
    # user,time_stamp,adgroup_id,pid,nonclk,clk
    global user_click_history
    dataset = []
    n_skipped = 0
    with open("data/taobao/raw_sample.csv") as f:
        all_lines = f.readlines()[1:]
        LIM = min(SAMPLE_LIMIT, len(all_lines))

        for i, line in enumerate(all_lines):
            if i % 1000 == 0:
                print(f"Reading taobao line={i}/{len(all_lines)}")
            if i >= LIM:
                break
            vals = line.split(",")

            if int(vals[0]) not in user_mappings[0]:
                print("User profile not found... continuing")
                n_skipped += 1
                continue

            if int(vals[2]) not in ad_mappings[0]:
                print("Ad profile not found... continuing")
                n_skipped += 1
                continue

            # Obtain all sparse features
            # - User sparse features
            remapped_user_id = user_mappings[0][int(vals[0])]
            user_sparse_features = user_features_processed[remapped_user_id]
            # - Ad sparse features (everything except price)
            remapped_ad_id = ad_mappings[0][int(vals[2])]
            ad_sparse_features = [x for i,x in enumerate(ad_features_processed[remapped_ad_id]) if i != 5]
            # - Context sparse features
            # TODO (need to do remapping)
            all_sparse_features = ad_sparse_features + user_sparse_features 

            # Obtain all dense features
            ad_dense_features = [x for i,x in enumerate(ad_features_processed[remapped_ad_id]) if i == 5]
            all_dense_features = ad_dense_features

            # Timestamp / target
            timestamp = int(vals[-3])
            click = int(vals[-1])

            # Add to dataset
            dataset.append((all_sparse_features, all_dense_features, timestamp, click))

            # Update user history
            if remapped_user_id not in user_click_history:
                user_click_history[remapped_user_id] = []
            user_click_history[remapped_user_id].append((remapped_ad_id, timestamp))

    print("Skipped", n_skipped)
    global train_dataset
    global val_dataset
    split_indx = int(len(dataset)*split)
    train_dataset = dataset[:split_indx]
    val_dataset = dataset[split_indx:]

    # Obtain table lengths for each sparse feature index
    global embedding_table_lengths
    rows = len(dataset)
    columns = len(dataset[0][0])
    for column in range(columns):
        vals = [dataset[row][0][column] for row in range(rows)]
        print(max(vals))
        embedding_table_lengths.append(max(vals)+1)

    # Obtain train and val access pattern
    print("Extracting access pattern")
    for i, (user_id, click_history) in enumerate(user_click_history.items()):
        hist = [x[0] for x in click_history]
        if i >= int(split*len(user_click_history)):
            val_access_pattern.append(hist)
        else:
            train_access_pattern.append(hist)

def evaluate_model(model, dataset, batch=64, pir_optimize=None):
    groundtruths, preds = [], []

    indices = list(range(len(dataset)))
    for b in range(0, len(indices), batch):
        
        points = [dataset[x] for x in indices[b:b+batch]]
        sparse_features = [x[0] for x in points]
        dense_features = [x[1] for x in points]
        timestamps = [x[2] for x in points]
        targets = [x[3] for x in points]

        # Get historical clicks
        user_ids = [x[5] for x in sparse_features]
        user_history = [get_user_history(user_ids[i], timestamps[i]) for i in range(len(user_ids))]

        sparse_features = torch.from_numpy(np.array(sparse_features)).long()
        dense_features = torch.from_numpy(np.array(dense_features)).float()
        user_history = torch.from_numpy(np.array(user_history)).long()
        targets = torch.from_numpy(np.array(targets)).long()

        sparse_features = sparse_features.to(next(model.parameters()).device)
        dense_features = dense_features.to(next(model.parameters()).device)
        user_history = user_history.to(next(model.parameters()).device)
        targets = targets.to(next(model.parameters()).device)

        ############################
        # PIR
        data_pir = []
        for bbatch in user_history:
            bbatch = bbatch.detach().cpu().numpy().tolist()
            n_fillers = bbatch.count(-1)
            bb = [x for x in bbatch if x != -1]
            if pir_optimize is not None:
                recovered, _ = pir_optimize.fetch(bb)
            else:
                recovered = bb
            # 9 is <unk>
            new_b = [x if x in recovered else -1 for x in bb]
            data_pir.append(new_b + [-1]*n_fillers)
        data_pir = np.array(data_pir)
        data_pir = torch.from_numpy(data_pir)
        data_pir = data_pir.to(next(model.parameters()).device)

        assert(data_pir.shape == user_history.shape)

        user_history= data_pir

        ############################                

        model.zero_grad()
        pred = model(sparse_features, dense_features, user_history)
        
        prob_click = F.softmax(pred, dim=1)[:,1]
        prob_click = prob_click.detach().cpu().numpy().flatten().tolist()

        preds += prob_click
        groundtruths += targets.detach().cpu().numpy().flatten().tolist()

    score = roc_auc_score(groundtruths, preds)
    return score

def get_user_history(user_id, timestamp):
    clicks = [x[0] for x in user_click_history[user_id] if x[1] < timestamp]
    L = 10000
    clicks = clicks[:L]
    if len(clicks) <= L:
        clicks += [-1]*(L-len(clicks))
    return clicks

def train_taobao_rec(epochs=100, batch=64):
    print("Training...")

    model = RecModel(embedding_table_lengths)
    model.to("cuda")
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    # Train on train users
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)

        train_loss = 0        
        for b in tqdm.tqdm(range(0, len(indices), batch)):
            points = [train_dataset[x] for x in indices[b:b+batch]]
            sparse_features = [x[0] for x in points]
            dense_features = [x[1] for x in points]
            timestamps = [x[2] for x in points]
            targets = [x[3] for x in points]

            # Get historical clicks
            user_ids = [x[5] for x in sparse_features]
            user_history = [get_user_history(user_ids[i], timestamps[i]) for i in range(len(user_ids))]

            sparse_features = torch.from_numpy(np.array(sparse_features)).long()
            dense_features = torch.from_numpy(np.array(dense_features)).float()
            user_history = torch.from_numpy(np.array(user_history)).long()
            targets = torch.from_numpy(np.array(targets)).long()

            sparse_features = sparse_features.to("cuda")
            dense_features = dense_features.to("cuda")
            user_history = user_history.to("cuda")
            targets = targets.to("cuda")            

            model.zero_grad()
            pred = model(sparse_features, dense_features, user_history)
            output = loss(pred, targets)

            output.backward()
            optim.step()

            train_loss += output.detach().cpu().item()
            
        score = evaluate_model(model, val_dataset)
        #score = evaluate_model(model, train_dataset)
        print("Eval score", score)

        torch.save(model, f"recmodel_epoch={epoch}.pt")

def evaluate(pir_optimize):
    dir_to_use = os.path.dirname(__file__)
    model = RecModel(embedding_table_lengths)
    #with open(f"{dir_to_use}/recmodel_epoch=0.pt", 'rb') as f:        
    #    model = torch.load(f)
    #    pass
    model.to("cpu")

    auc = evaluate_model(model, val_dataset, pir_optimize=pir_optimize)
    print(f"AUC: {auc}")
    return {"auc" : auc}
                    
if __name__=="__main__":
    initialize()
    #train_taobao_rec()
    evaluate(None)
