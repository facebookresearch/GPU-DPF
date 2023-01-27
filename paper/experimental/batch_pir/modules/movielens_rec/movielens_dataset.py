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

train_dataset = []
val_dataset = []

num_movies = 0
num_users = 0

LIM = 10000000000
#LIM = 10000
split = .8

class RecModel(torch.nn.Module):

    def __init__(self, n_movies, em_size=32):
        super().__init__()

        # First embedding table index is the ads embedding 
        self.table = torch.nn.EmbeddingBag(n_movies+1, em_size, mode="sum")
        self.em_size = em_size

        self.fc1 = torch.nn.Linear(64, 200)
        self.fc2 = torch.nn.Linear(200, 80)
        self.fc3 = torch.nn.Linear(80, 2)

        self.d = torch.nn.Dropout(.5)

    def forward(self, movie_history, target_movie):
        target_movie = target_movie.reshape((-1, 1))

        self.table.weight.data[0,:] = 0

        target_embedding = self.table(target_movie+1)
        movie_history_embedding = self.table(movie_history+1)

        x = torch.cat([target_embedding, movie_history_embedding], dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.d(x)        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x
    
def initialize():

    user_ratings = {}
    
    with open("data/ml-20m/ratings.csv", "r") as f:
        lines = f.readlines()[1:]
        # userId,movieId,rating,timestamp
        for i,line in enumerate(lines):
            if i >= LIM:
                break
            line = line.split(",")
            user_id, movie_id, rating, timestamp = line[0], line[1], line[2], line[3]
            user_id, movie_id, rating, timestamp = int(user_id), int(movie_id), float(rating), int(timestamp)

            click = rating >= 4

            if user_id not in user_ratings:
                user_ratings[user_id] = []

            user_ratings[user_id].append((movie_id, click, timestamp))


    global train_dataset
    global val_dataset
    global num_users
    global num_movies
    train_dataset = []
    val_dataset = []
    num_users = 0
    num_movies = 0
    for i, (user_id, d) in enumerate(user_ratings.items()):
        test = i >= int(split*len(user_ratings))
        user_click_history = [(x[0], x[2]) for x in d if x[1]]
        num_users = max(num_users, user_id)
        for movie_id, click, timestamp in d:
            if test:
                val_dataset.append((user_click_history, movie_id, timestamp, click))
            else:
                train_dataset.append((user_click_history, movie_id, timestamp, click))
            num_movies = max(num_movies, movie_id)

    num_users += 1
    num_movies += 1

    print("movies: ", num_movies)

    # Extract train and val access pattern
    print("Extracting access pattern...")
    for i, (user_id, d) in enumerate(user_ratings.items()):
        test = i >= int(split*len(user_ratings))
        user_click_history = [x[0] for x in d if x[1]]
        if test:
            val_access_pattern.append(user_click_history)
        else:
            train_access_pattern.append(user_click_history)
    
def obtain_click_history(point, timestamp):
    L = 5000
    click_history = point[0]
    click_history = [x[0] for x in click_history if x[1] < timestamp]
    if len(click_history) < L:
        click_history += [-1]*(L-len(click_history))
    if len(click_history) != L:
        print(len(click_history))
    assert(len(click_history) == L)
    return click_history

def evaluate(pir_optimize):
    dir_to_use = os.path.dirname(__file__)
    model = RecModel(num_movies)
    with open(f"{dir_to_use}/recmodel_epoch=1.pt", 'rb') as f:        
        model = torch.load(f)
        pass
    model.to("cpu")

    auc = evaluate_model(model, val_dataset, pir_optimize=pir_optimize)
    print(f"AUC: {auc}")
    return {"auc" : auc}

def evaluate_model(model, dataset, batch=256, pir_optimize=None):
    model.eval()
    groundtruths, preds = [], []

    indices = list(range(len(dataset)//10))
    for b in range(0, len(indices), batch):
        print(f"evaluate_model {b}/{len(indices)}")
        
        # Get user "clicks"
        points = [train_dataset[x] for x in indices[b:b+batch]]
        timestamps = [x[2] for x in points]
        click_history = [obtain_click_history(x, timestamps[i]) for i,x in enumerate(points)]

        ############################
        # PIR
        data_pir = []
        for bbatch in click_history:            
            n_fillers = bbatch.count(-1)
            bb = [x for x in bbatch if x != -1]
            if pir_optimize is not None:
                recovered, _ = pir_optimize.fetch(bb)
            else:
                recovered = bb
            # 9 is <unk>
            new_b = [x if x in recovered else -1 for x in bb]
            data_pir.append(new_b + [-1]*n_fillers)
        #data_pir = np.array(data_pir)
        #data_pir = torch.from_numpy(data_pir)
        #data_pir = data_pir.to(next(model.parameters()).device)

        #assert(data_pir.shape == click_history.shape)

        click_history= data_pir

        ############################        """        
        
        target_movie = [x[1] for x in points]
        targets = [x[-1] for x in points]

        click_history = torch.from_numpy(np.array(click_history)).long()
        target_movie = torch.from_numpy(np.array(target_movie)).long()
        targets = torch.from_numpy(np.array(targets)).long()

        click_history = click_history.to(next(model.parameters()).device)
        target_movie = target_movie.to(next(model.parameters()).device)
        targets = targets.to(next(model.parameters()).device)

        pred = model(click_history, target_movie)
        
        prob_click = F.softmax(pred, dim=1)[:,1]
        prob_click = prob_click.detach().cpu().numpy().flatten().tolist()

        preds += prob_click
        groundtruths += targets.detach().cpu().numpy().flatten().tolist()

    score = roc_auc_score(groundtruths, preds)
    model.train()    
    return score

def train_movielens(epochs=100, batch=64):
    print("Training...")

    model = RecModel(num_movies)
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
            # Get user "clicks"
            points = [train_dataset[x] for x in indices[b:b+batch]]
            timestamps = [x[2] for x in points]
            click_history = [obtain_click_history(x, timestamps[i]) for i,x in enumerate(points)]
            target_movie = [x[1] for x in points]
            targets = [x[-1] for x in points]

            click_history = torch.from_numpy(np.array(click_history)).long()
            target_movie = torch.from_numpy(np.array(target_movie)).long()
            targets = torch.from_numpy(np.array(targets)).long()

            click_history = click_history.to("cuda")
            target_movie = target_movie.to("cuda")
            targets = targets.to("cuda")

            model.zero_grad()
            pred = model(click_history, target_movie)
            output = loss(pred, targets)
        
            output.backward()
            optim.step()

            train_loss += output.detach().cpu().item()
            
        score = evaluate_model(model, val_dataset)
        #score = evaluate_model(model, train_dataset)
        print("Eval score", score)

        torch.save(model, f"recmodel_epoch={epoch}.pt")        

if __name__=="__main__":
    initialize()
    #train_movielens()
    evaluate(None)
