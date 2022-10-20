#!/usr/bin/env python3

from this import d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import sys
import os
import random
import argparse
import pandas as pd
import gzip
import pickle
import json

from src.ccrec.models.vae_training import VAE_training
from transformers import TrainingArguments
from transformers import TrainerCallback

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=350, type=int)
parser.add_argument("--model_name", type=str, default='distilbert-base-uncased')
parser.add_argument("--save_name", type=str, default='msmarco_VAE_model')

parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--beta", default=2e-3, type=float)

args = parser.parse_args()

# %%
# load MS_MARCO data
def load_corpus():
    data_dir = 'data/ms_marco/collection.tsv'
    dataframe = pd.read_csv(data_dir, sep='\t', header=None, names=['pid', 'passage'])
    print('Number of passages:', len(dataframe))
    return dataframe

def load_dev_data():
    data_dir = 'data/ms_marco/qrels.dev.tsv'
    dataframe = pd.read_csv(data_dir, sep='\t', header=None, usecols=[0, 2], names=['qid', 'pid'])
    dataframe = dataframe.groupby('qid')['pid'].apply(list).to_dict()
    print('Number of development queries:', len(dataframe))
    return dataframe

def load_query():
    df_train = pd.read_csv('data/ms_marco/queries.train.tsv', sep='\t', header=None, names=['qid', 'query'])
    # df_dev = pd.read_csv('data/ms_marco/queries.dev.tsv', sep='\t', header=None, names=['qid', 'query'])
    # df_eval = pd.read_csv('data/ms_marco/queries.eval.tsv', sep='\t', header=None, names=['qid', 'query'])
    # dataframe = pd.concat([df_train, df_dev, df_eval], ignore_index=True)
    dataframe = df_train
    print('Number of queries:', len(dataframe))
    return dataframe

def load_ms_marco():
    corpus = load_corpus()
    queries = load_query()
    corpus = dict(zip(corpus['pid'], corpus['passage']))
    queries = dict(zip(queries['qid'], queries['query']))
    ce_scores = pd.read_pickle('data/ms_marco/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl')
    print("Read hard negatives train file")
    hard_negatives_filepath = 'data/ms_marco/msmarco-hard-negatives.jsonl'
    num_of_negatives = 1
    dataset = {}
    with open(hard_negatives_filepath, 'rt') as fIn:        
        for line in fIn:
            data = json.loads(line)
            qid = data['qid']
            pos_pids = data['pos']

            if len(pos_pids) == 0:
                continue
            pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
            neg_pids = []
            neg_pids_ce_scores = []
            negs_to_use = list(data['neg'].keys())
            if len(negs_to_use) == 0:
                continue
            negs_to_use = negs_to_use[0:1]
            for system_name in negs_to_use:
                system_negs = data['neg'][system_name]
                neg_pids = neg_pids + system_negs
            for pid in neg_pids:
                neg_pids_ce_scores.append(ce_scores[qid][pid])
            neg_pids_ce_scores_ = torch.FloatTensor(neg_pids_ce_scores)
            _, sorting_indices = neg_pids_ce_scores_.sort(descending=True)
            sorting_indices = sorting_indices[0:num_of_negatives]
            neg_pids = [neg_pids[idx.item()] for idx in sorting_indices]
            dataset[qid] = {'pos_pid': pos_pids, 'neg_pid': neg_pids}
    del ce_scores
    titles = []
    for qid in dataset:
        titles.append(queries[qid])
        for pid_pos in dataset[qid]['pos_pid']:
            titles.append(corpus[pid_pos])
        for pid_neg in dataset[qid]['neg_pid']:
            titles.append(corpus[pid_neg])
    # titles = list(queries.values()) + list(corpus.values())
    # titles = list(set(titles))
    # random.shuffle(titles)
    print('Number of training data:', len(titles))
    titles = pd.DataFrame(titles, columns=['TITLE'])
    return titles


def main(opt):
    _batch_size = opt.batch_size
    _max_seq_length = opt.max_seq_length
    _model_name = opt.model_name
    _save_name = opt.save_name

    _epochs = opt.epochs
    _lr = opt.lr
    _weight_decay = opt.weight_decay
    _beta = opt.beta

    _save_name = _save_name + '_' + str(_beta)

    DataFrame = load_ms_marco()
    num_of_data = len(DataFrame)

    training_args = TrainingArguments(
        num_train_epochs=_epochs,
        output_dir=f"{_save_name}",
        overwrite_output_dir=True,
        # evaluation_strategy="epoch",
        learning_rate=_lr,
        weight_decay=_weight_decay,
        per_device_train_batch_size=_batch_size,
        per_device_eval_batch_size=_batch_size,
        push_to_hub=False,
        fp16=True,
        save_strategy="epoch",
        logging_steps = 1,
    )
 
    VAE_training(DataFrame, training_args, train_set_ratio=0.95, model_checkpoint=_model_name, max_length=_max_seq_length, vae_beta=_beta)


if __name__ == '__main__':
    main(args)