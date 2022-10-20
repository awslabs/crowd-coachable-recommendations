#!/usr/bin/env python3

from this import d
from unittest.mock import NonCallableMagicMock
import torch
import torch.nn as nn
import numpy as np

import sys
import os
import random
import argparse
import pandas as pd
import gzip
import pickle
import json
import math

from src.ccrec.models.bert_mt import bmt_main
from src.ccrec.models.bbpr import bbpr_main
from src.ccrec.models.vae_lightning import VAEModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", type=str, default='distilbert-base-uncased')
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--beta", default=2e-3, type=float)
parser.add_argument("--training_method", default='vae', type=str)
parser.add_argument("--freeze_bert", default=0, type=int)
parser.add_argument("--input_type", default='text', type=str)
parser.add_argument("--training_dataset", default=None, type=str)
parser.add_argument("--number_of_negatives", default=1000000, type=int)


args = parser.parse_args()

# %%
# load MS_MARCO data
def load_corpus():
    data_dir = 'data/ms_marco/collection.tsv'
    dataframe = pd.read_csv(data_dir, sep='\t', header=None, names=['pid', 'passage'])
    dataframe = dict(zip(dataframe['pid'], dataframe['passage']))
    print('Number of passages:', len(dataframe))
    return dataframe

def load_query():
    dataframe = pd.read_csv('data/ms_marco/queries.train.tsv', sep='\t', header=None, names=['qid', 'query'])
    dataframe = dict(zip(dataframe['qid'], dataframe['query']))
    print('Number of queries:', len(dataframe))
    return dataframe

def load_training_data(training_data_dir=None, num_of_negative_samples=1000000):
    ce_scores = pd.read_pickle('data/ms_marco/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl')
    if training_data_dir is not None:
        dataset = torch.load(training_data_dir)
        return (dataset, ce_scores)
    print("Read hard negatives train file")
    hard_negatives_filepath = 'data/ms_marco/msmarco-hard-negatives.jsonl'
    ce_score_margin = 3.
    num_neg_per_system = 5
    dataset = {}
    count = 0
    with open(hard_negatives_filepath, 'rt') as fIn:        
        for line in fIn:
            # count += 1
            # if count == 1001:
            #     break

            data = json.loads(line)
            qid = data['qid']
            pos_pids = data['pos']

            if len(pos_pids) == 0:
                continue
            pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
            ce_score_threshold = pos_min_ce_score - ce_score_margin
            neg_pids = set()
            negs_to_use = list(data['neg'].keys())
            for system_name in negs_to_use:
                if system_name not in data['neg']:
                    continue
                system_negs = data['neg'][system_name]
                negs_added = 0
                for pid in system_negs:
                    if ce_scores[qid][pid] > ce_score_threshold:
                        continue
                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= num_neg_per_system:
                            break
            neg_pids = list(neg_pids)
            if (len(pos_pids) > 0 and len(neg_pids) > 0):
                dataset[qid] = {'pos_pid': pos_pids, 'neg_pid': neg_pids[0:num_of_negative_samples]}
    dataset = (dataset, ce_scores)
    return dataset

def load_item_df(dataset, corpus, queries):
    pos_pids_all, neg_pids_all = [], []
    for value in dataset.values():
        for pid in value['pos_pid']:
            pos_pids_all.append(pid)
        for pid in value['neg_pid']:
            neg_pids_all.append(pid)
    pids_all = list(set(pos_pids_all + neg_pids_all))
    passage_all = [corpus[pid] for pid in pids_all]

    qids_all = list(dataset.keys())
    query_all = [queries[qid] for qid in qids_all]
    qids_all = [-qid for qid in qids_all]

    item_id = qids_all + pids_all
    title_all = query_all + passage_all
    item_df = pd.DataFrame({'ITEM_ID': item_id, 'TITLE': title_all})
    item_df = item_df.set_index("ITEM_ID")
    return item_df

def load_user_df(dataset):
    USER_ID = list(range(len(dataset)))
    TEST_START_TIME = [1] * len(USER_ID)
    _hist_items = [[-qid] for qid in list(dataset.keys())]
    _hist_ts = [0] * len(USER_ID)
    user_df = pd.DataFrame({'USER_ID': USER_ID,
                        'TEST_START_TIME': TEST_START_TIME,
                        '_hist_items': _hist_items,
                        '_hist_ts': _hist_ts})
    user_df = user_df.set_index("USER_ID")
    return user_df

def load_expl_response(dataset):
    USER_ID = list(range(len(dataset)))
    request_time = [2] * len(USER_ID)
    _hist_items = [-qid for qid in dataset.keys()]
    cand_items = []
    multi_label = []
    for values in dataset.values():
        candicates = [values['pos_pid'][0]] + values['neg_pid']
        labels = [1.] * len([values['pos_pid'][0]]) + [0.] * len(values['neg_pid'])
        cand_items.append(candicates)
        multi_label.append(labels)
    expl_response = pd.DataFrame({'USER_ID': USER_ID,
                                'request_time': request_time,
                                '_hist_items': _hist_items,
                                'cand_items': cand_items,
                                'multi_label': multi_label})
    expl_response = expl_response.set_index("USER_ID")
    return expl_response

    

def main(opt):
    _batch_size = opt.batch_size
    _max_seq_length = opt.max_seq_length
    _model_name = opt.model_name
    _epochs = opt.epochs
    _lr = opt.lr
    _weight_decay = opt.weight_decay
    _beta = opt.beta
    _freeze_bert = opt.freeze_bert
    _training_method = opt.training_method
    _input_type = opt.input_type
    _training_dataset = opt.training_dataset
    _number_of_negatives = opt.number_of_negatives
    
    corpus = load_corpus()
    queries = load_query()
    dataset = load_training_data(_training_dataset, _number_of_negatives)
    item_df = load_item_df(dataset[0], corpus, queries)
    user_df = load_user_df(dataset[0])
    expl_response = load_expl_response(dataset[0])

    training_arguments = {'max_epochs': _epochs, 'beta': _beta, 'lr': _lr, 'weight_decay':_weight_decay,
                            'model_name': _model_name, 'max_length': _max_seq_length, 'batch_size': _batch_size,
                            'freeze_bert': _freeze_bert, 'input_type': _input_type}

    if _training_method == 'bertmt':
        bmt_main(item_df, expl_response, expl_response, user_df, dataset, train_kw=training_arguments)
    elif _training_method == 'bertbpr':
        bbpr_main(item_df, expl_response, expl_response, user_df, dataset, train_kw=training_arguments)
    elif _training_method == 'vae':
        VAEModule = VAEModel(item_df, model_cls_name='VAEPretrainedModel', **training_arguments)
        VAEModule.fit()

if __name__ == '__main__':
    main(args)