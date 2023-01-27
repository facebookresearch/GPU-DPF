# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import sys
import tqdm
import scipy
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

SAMPLE_LIMIT = 10000000000

num_embeddings = -1

train_access_pattern = []
val_access_pattern = []

train_access_pattern_user = []
val_access_pattern_user = []

ad_features = {}
user_features = {}
user_clicks = {}

ad_id_to_embedding_id = {}
user_id_to_embedding_id = {}

# For batch-pir stats
split = .8

####################################

# For training
click_noclick_dataset_train = []
click_noclick_dataset_test = []
train_test_split = .8

#####################################


class RecModel(torch.nn.Module):

    def __init__(self, ad_feature_table, num_ads, em_size=160):
        super().__init__()

        self.num_ads = num_ads
        self.em_size = em_size
        
        self.ad_embeddings = torch.nn.EmbeddingBag(num_ads+1, em_size, mode='sum')

        self.ad_dense_feature_table = torch.nn.EmbeddingBag(ad_feature_table.shape[0], ad_feature_table.shape[1], mode="sum")
        self.ad_dense_feature_table.weight.data[:,:] = torch.from_numpy(ad_feature_table)
        self.ad_dense_feature_table.weight.requires_grad = False

        # 414
        self.fc1 = torch.nn.Linear(340, 200)
        self.fc2 = torch.nn.Linear(200, 80)
        self.fc3 = torch.nn.Linear(80, 2)

    def forward(self, user_features, user_ad_history, target_ad_id):

        # Zero out index 0 of ad_id (0 is special index for 0 vector)
        self.ad_embeddings.weight.data[0,:] = 0

        # Sparse features
        user_ad_history = user_ad_history + 1
        target_ad_id = target_ad_id.reshape((-1, 1))
        target_ad_sparse_feature = self.ad_embeddings(target_ad_id)        
        user_ad_sparse_features = self.ad_embeddings(user_ad_history)

        # Dense features
        user_ad_dense_features = self.ad_dense_feature_table(user_ad_history)
        target_ad_dense_feature = self.ad_dense_feature_table(target_ad_id)


        x = torch.cat([user_ad_dense_features, target_ad_dense_feature, user_ad_sparse_features, target_ad_sparse_feature, user_features], dim=1)

        #print(x.shape)
        #sys.exit(0)
        #print([x.shape for x in [user_ad_dense_features, target_ad_dense_feature, user_ad_sparse_features, target_ad_sparse_feature, user_features]])

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        #print(x.shape)        
        #sys.exit(0)

        return x

def initialize():

    ads = set()    

    # Read ad click user access pattern
    with open("data/taobao/raw_sample.csv") as f:
        all_lines = f.readlines()
        LIM = min(SAMPLE_LIMIT, len(all_lines))

        # Sort by timestamp
        all_lines = all_lines[1:]
        all_lines.sort(key=lambda x:int(x.split(",")[1]))

        for i, line in enumerate(all_lines):
            print(f"Reading taobao line={i}/{len(all_lines)}, num_items_in_dict={len(user_clicks.items())}")
            if len(user_clicks.items()) >= LIM:
                break
            vals = line.split(",")
            user, ad_id, click = vals[0], vals[2], vals[-1].strip()

            ads.add(ad_id)

            if user not in user_clicks:
                user_clicks[user] = set()            

            if i >= train_test_split*LIM:
                click_noclick_dataset_test.append((user, ad_id, int(click)))
            else:
            
                click_noclick_dataset_train.append((user, ad_id, int(click)))

                if int(click) == 1:
                    user_clicks[user].add(ad_id)

    global num_embeddings
    num_embeddings = len(ads)

    # Remap ads id to sequential indices
    count = 0
    for ad_id in sorted(ads):
        if ad_id not in ad_id_to_embedding_id:
            ad_id_to_embedding_id[ad_id] = count
            count += 1

    # train/test pattern (for batch pir)
    for i, (user, clickset) in enumerate(sorted(user_clicks.items())):
        remapped_clickset = [ad_id_to_embedding_id[x] for x in clickset]
        if i <= int(split*len(user_clicks.items())):
            train_access_pattern.append(remapped_clickset)
            train_access_pattern_user.append(user)
        else:
            val_access_pattern.append(remapped_clickset)
            val_access_pattern_user.append(user)

    # Read ad features
    with open("data/taobao/ad_feature.csv", "r") as f:
        all_lines = f.readlines()
        for i, line in enumerate(all_lines):
            if i == 0:
                continue
            vals = line.split(",")
            ad_id = vals[0]

            # This happens when using a limited sample set
            if ad_id not in ad_id_to_embedding_id:
                continue

            #print("Add ad: %s" % ad_id)
            
            remapped_id = ad_id_to_embedding_id[ad_id]            

            vals = [int(vals[0])/1000000,
                    int(vals[1])/100000,
                    int(vals[2])/1000000,
                    int(vals[3])/100,
                    0 if vals[4] == "NULL" else int(vals[4])/1000000,
                    np.log2(float(vals[5]))/100]
            
            feature_vector = np.array(vals)
            ad_features[remapped_id] = feature_vector

    # Read user features
    with open("data/taobao/user_profile.csv", "r") as f:
        all_lines = f.readlines()
        for i, line in enumerate(all_lines):
            if i == 0:
                continue
            vals = line.split(",")
            user_id = vals[0]

            # Happens if using a subset of full data
            #if user_id not in user_clicks.keys():
            #    continue

            #print("Add user: %s" % user_id)

            vals = [0 if x.strip() == "" else float(x) for x in vals[1:]]
            feature_vector = np.array(vals) / 100

            user_features[user_id] = feature_vector

def get_dense_features(d):
    user_id, ad_id = d[0], d[1]

    # Get dense features
    if user_id not in user_features:
        return None
    
    user_features_dense = user_features[user_id]
    target_ad_features_dense = ad_features[ad_id_to_embedding_id[ad_id]]


    return user_features_dense.flatten()

def get_sparse_features(d):
    user_id, ad_id = d[0], d[1]

    user_ad_history = user_clicks[user_id]
    user_ad_history = [ad_id_to_embedding_id[x] for x in user_ad_history]
    target_ad_id = ad_id_to_embedding_id[ad_id]

    # Pad user_ad_history to length 100
    L = 1000
    user_ad_history = user_ad_history[-L:]
    if len(user_ad_history) < L:
        user_ad_history = user_ad_history + [-1]*(L-len(user_ad_history))
    assert(len(user_ad_history) == L)

    return np.array(user_ad_history), target_ad_id

def get_target(d):
    return int(d[2])

def evaluate_model(model, dataset, batch=64):
    groundtruths, preds = [], []

    indices = list(range(len(dataset)))
    for b in range(0, len(indices), batch):
        b_indices = indices[b:b+batch]        
        
        dense_features = [get_dense_features(dataset[i]) for i in b_indices]
        sparse_features = [get_sparse_features(dataset[i]) for i in b_indices]            
        targets = [get_target(dataset[i]) for i in b_indices]

        targets = [x for i,x in enumerate(targets) if dense_features[i] is not None]
        sparse_features = [x for i,x in enumerate(sparse_features) if dense_features[i] is not None]                        
        dense_features = [x for x in dense_features if x is not None]

        targets = torch.from_numpy(np.array(targets))
        user_features = torch.from_numpy(np.array(dense_features)).float()
        user_ad_sparse_features = torch.from_numpy(np.array([x[0] for x in sparse_features])).long()
        target_ad_sparse_feature = torch.from_numpy(np.array([x[1] for x in sparse_features])).long()

        targets = targets.to("cuda")
        user_features = user_features.to("cuda")
        user_ad_sparse_features = user_ad_sparse_features.to("cuda")
        target_ad_sparse_feature = target_ad_sparse_feature.to("cuda")
        
        pred = model(user_features, user_ad_sparse_features, target_ad_sparse_feature)
        
        prob_click = F.softmax(pred, dim=1)[:,1]
        prob_click = prob_click.detach().cpu().numpy().flatten().tolist()

        preds += prob_click
        groundtruths += targets.detach().cpu().numpy().flatten().tolist()

    score = roc_auc_score(groundtruths, preds)
    return score

def train_taobao_rec(epochs=100, batch=64):
    print("Training...")

    ad_features_table = [None for i in range(len(ad_id_to_embedding_id)+1)]
    for ad_id, embedding_id in ad_id_to_embedding_id.items():
        f = ad_features[embedding_id]
        ad_features_table[embedding_id+1] = np.array(f)
    ad_features_table[0] = np.zeros(ad_features_table[1].shape)
    ad_features_table = np.array(ad_features_table)
    
    model = RecModel(ad_features_table, len(ad_id_to_embedding_id.keys())).to("cuda")
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    # Train on train users
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        indices = list(range(len(click_noclick_dataset_train)))
        np.random.shuffle(indices)

        train_loss = 0
        
        for b in tqdm.tqdm(range(0, len(indices), batch)):
            b_indices = indices[b:b+batch]
            
            dense_features = [get_dense_features(click_noclick_dataset_train[i]) for i in b_indices]
            sparse_features = [get_sparse_features(click_noclick_dataset_train[i]) for i in b_indices]            
            targets = [get_target(click_noclick_dataset_train[i]) for i in b_indices]

            targets = [x for i,x in enumerate(targets) if dense_features[i] is not None]
            sparse_features = [x for i,x in enumerate(sparse_features) if dense_features[i] is not None]                        
            dense_features = [x for x in dense_features if x is not None]

            targets = torch.from_numpy(np.array(targets)).long()
            user_features = torch.from_numpy(np.array(dense_features)).float()
            user_ad_sparse_features = torch.from_numpy(np.array([x[0] for x in sparse_features])).long()
            target_ad_sparse_feature = torch.from_numpy(np.array([x[1] for x in sparse_features])).long()

            targets = targets.to("cuda")
            user_features = user_features.to("cuda")
            user_ad_sparse_features = user_ad_sparse_features.to("cuda")
            target_ad_sparse_feature = target_ad_sparse_feature.to("cuda")

            #print(user_features.shape, target_ad_dense_feature.shape,
            #      user_ad_sparse_features.shape, target_ad_sparse_feature.shape)

            model.zero_grad()
            pred = model(user_features, user_ad_sparse_features, target_ad_sparse_feature)
            output = loss(pred, targets)

            output.backward()
            optim.step()

            train_loss += output.detach().cpu().item()
        
        score = evaluate_model(model, click_noclick_dataset_test)
        #score = evaluate_model(model, click_noclick_dataset_train)
        #torch.save(model, "recmodel.pt")
        print("Eval AUC-ROC Score", score)
    
if __name__=="__main__":
    initialize()
    train_taobao_rec()
