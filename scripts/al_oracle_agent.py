#!/usr/bin/env python3
import torch
import pandas as pd
import os
import sys
import random
from torch.cuda.amp import autocast
import argparse
import warnings
import numpy as np

import inspect
from beir.retrieval.evaluation import EvaluateRetrieval

from transformers import AutoTokenizer, AutoModel
from ccrec.models.bert_mt import bmt_main
from ccrec.models.bert_mt import _BertMT
from ccrec.models.bbpr import _BertBPR
from ccrec.util.data_parallel import DataParallel

from train_bmt_msmarco import (
    load_item_df,
    load_user_df,
    load_expl_response,
)

from ms_marco_eval import ranking, load_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"


N_STEPS = 1
ACCURACY_LEVEL = 1.0
NUM_EPOCHS = 10
DATA_NAME = "nq"
BATCH_SIZE = 865
MODEL_NAME = "facebook/contriever"


# %%
# train model
def training(
    train_dataset,
    epochs,
    model_checkpoint,
    corpus,
    queries,
    model_selection,
):
    item_df = load_item_df(train_dataset, corpus, queries)
    user_df = load_user_df(train_dataset)
    expl_response = load_expl_response(train_dataset)

    training_arguments = {
        "lr": 2e-5,
        "model_name": model_selection,
        "max_length": int(os.environ.get("CCREC_MAX_LENGTH", 300)),
        "pretrained_checkpoint": None,
        "do_validation": False,
    }

    _batch_size = 30
    _epochs = epochs
    _alpha = 1.0
    _beta = 2e-3

    _, _, model = bmt_main(
        item_df,
        expl_response,
        expl_response,
        _epochs,
        _batch_size,
        _alpha,
        _beta,
        user_df,
        train_kw=training_arguments,
    )
    return model


# %%
# evaluate and generate ranking profile
def generate_ranking_profile(
    model, model_name, corpus, queries, qrels, save_dir, block_dict=None
):
    batch_size = 512

    _gpu_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        batch_size = batch_size * len(_gpu_ids)

    tokenizer_kw = {
        "truncation": True,
        "padding": True,
        "max_length": int(os.environ.get("CCREC_MAX_LENGTH", 512)),
        "return_tensors": "pt",
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        model.item_tower
        if hasattr(model, "item_tower")
        else model.model.item_tower
        if hasattr(model, "model")
        else model
    )
    model.eval()
    model = DataParallel(model.cuda(), device_ids=_gpu_ids).cache_replicas()

    embedding_type = os.environ["CCREC_EMBEDDING_TYPE"]
    if embedding_type != "mean_pooling":
        warnings.warn(f"{embedding_type} != mean_pooling for contriever models")

    def embedding_func(x):
        tokens = tokenizer(x, **tokenizer_kw)
        outputs = model(**tokens, output_step=embedding_type)
        return outputs

    ranking_profile = ranking(corpus, queries, embedding_func, batch_size, block_dict)

    evaluator = EvaluateRetrieval(None)
    mrr = evaluator.evaluate_custom(
        qrels, ranking_profile, [1, 5, 10, 100], metric="mrr"
    )
    for name, value in mrr.items():
        print("{}".format(name), ":", value)

    torch.save(ranking_profile, save_dir)
    return ranking_profile


# %%
# generate training data
def generate_train_data(
    qids,
    qrels,
    ranking_profile,
    ranking_profile_2,  # bm25
    corpus_key_list=[],
    rng_seed=None,  # STEP
):
    ranks_rng = np.random.RandomState(rng_seed)
    train_data = dict()
    for qid in qids:
        pids = list(ranking_profile[qid].keys())
        pids = pids[0:2]

        pids_2 = list(ranking_profile_2[qid].keys())
        for pid in pids_2:
            if len(pids) == 4:
                break
            if pid not in pids:
                pids.append(pid)

        if len(corpus_key_list):  # add a random choice for attention checks
            pids = pids[:3]
            while len(pids) < 4:
                pid = corpus_key_list[ranks_rng.choice(len(corpus_key_list))]
                if pid not in pids:
                    pids.append(pid)

        random.shuffle(pids)
        pos_pid = pids[0:1]
        neg_pid = pids[1:]

        labels = list(qrels[qid].keys())
        if any(pid in labels for pid in pids):
            pos_pid = []
            neg_pid = []
            for pid in pids:
                if pid in labels:
                    pos_pid = [pid]
                else:
                    neg_pid.append(pid)
            train_data[qid] = {"pos_pid": pos_pid, "neg_pid": neg_pid}
        elif len(corpus_key_list):
            pass  # in new version, skip n/a class
        else:
            train_data[qid] = {"pos_pid": pos_pid, "neg_pid": neg_pid}
    return train_data


def combine_train_data(train_data_pre, train_data_new):
    for qid, item in train_data_new.items():
        train_data_pre[qid] = item
    return train_data_pre
