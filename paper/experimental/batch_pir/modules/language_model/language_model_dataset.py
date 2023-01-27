# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data

import os
import sys
from io import open
import torch
import language_model as model_module
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


corpus = Corpus("./data/wikitext-2/")

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.numpy().tolist()
    return data

def batchify_pytorch(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to("cpu")

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = [x[0] for x in source[i:i+seq_len]]
    target = source[i+1:i+1+seq_len]
    return data, target

def get_batch_pytorch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_access_pattern(data_source):
    access_pattern = []
    
    # 35 is the defaul bptt for pytorch lang model
    for i in range(0, len(data_source), bptt):
        data, targets = get_batch(data_source, i)
        access_pattern.append(data)
    return access_pattern
    
train_data = batchify(corpus.train, 1)
val_data = batchify(corpus.valid, 1)
test_data = batchify(corpus.test, 1)
bptt = 35

train_access_pattern = None
test_access_pattern = None
val_access_pattern = None
num_embeddings = None
test_words = None

def wordify(source):
    sentences = []
    for b in source:
        bb = [corpus.dictionary.idx2word[i] for i in b]
        sentences.append(" ".join(bb))
    return sentences


def initialize():
    global train_access_pattern
    global test_access_pattern
    global val_access_pattern
    global num_embeddings

    train_access_pattern = get_access_pattern(train_data)
    test_access_pattern = get_access_pattern(test_data)
    val_access_pattern = get_access_pattern(val_data)

    #test_words = wordify(test_access_pattern)
    #print(wordify([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    #sys.exit(0)
    num_embeddings = len(corpus.dictionary.idx2word)

def evaluate(pir_optimize):
    print("Language model evaluating...")
    ntokens = len(corpus.dictionary)
    emsize = 650
    nhid = 650
    nlayers = 2
    dropout = .5
    tied = True
    eval_batch_size = 64
    bptt = 35

    model = model_module.RNNModel("LSTM", ntokens, emsize, nhid, nlayers,
                                  dropout, tied).to("cpu")

    # Load the best saved model.
    dir_to_use = os.path.dirname(__file__)
    with open(f"{dir_to_use}/model.pt", 'rb') as f:
        model = torch.load(f)
        model.to("cpu")
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        model.rnn.flatten_parameters()
        
    criterion = nn.NLLLoss()

    data_source = batchify_pytorch(corpus.valid, eval_batch_size)
    
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch_pytorch(data_source, i)

            #####################################################
            # PIR optimization: drop according to pir_optimize
            data_pir = []
            for batch in data:
                b = batch.detach().numpy().tolist()
                recovered, _ = pir_optimize.fetch(b)
                # 9 is <unk>
                new_b = [x if x in recovered else 9 for x in b]
                data_pir.append(new_b)
            data_pir = np.array(data_pir)
            data_pir = torch.from_numpy(data_pir)

            assert(data_pir.shape == data.shape)

            data = data_pir
            
            #####################################################
            
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    ppl = total_loss / (len(data_source) - 1)
    print("Language model ppl: %f" % ppl)
    return {"ppl" : math.exp(ppl)}

if __name__=="__main__":
    initialize()
