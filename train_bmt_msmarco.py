#!/usr/bin/env python3

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

from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pathlib

from transformers import AutoTokenizer, AutoModel
from src.ccrec.models.bert_mt import bmt_main
from src.ccrec.models.bbpr import bbpr_main
from src.ccrec.models.vae_training import VAE_training
from src.ccrec.models.vae_lightning import vae_main

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=24, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--beta", default=2e-3, type=float)
parser.add_argument("--alpha", default=0.1, type=float)
parser.add_argument("--training_method", default="bertmt", type=str)
parser.add_argument("--checkpoint", default=None, type=str)
parser.add_argument("--training_dataset_dir", default=None, type=str)
parser.add_argument("--do_validation", default=False, type=bool)


args = parser.parse_args()

# %%
# load MS_MARCO data
def load_corpus():
    data_dir = "data/ms_marco/collection.tsv"
    dataframe = pd.read_csv(data_dir, sep="\t", header=None, names=["pid", "passage"])
    dataframe = dict(zip(dataframe["pid"], dataframe["passage"]))
    print("Number of passages:", len(dataframe))
    return dataframe


def load_query():
    dataframe_train = pd.read_csv(
        "data/ms_marco/queries.train.tsv", sep="\t", header=None, names=["qid", "query"]
    )
    dataframe_dev = pd.read_csv(
        "data/ms_marco/queries.dev.tsv", sep="\t", header=None, names=["qid", "query"]
    )
    dataframe = pd.concat([dataframe_train, dataframe_dev])
    dataframe = dict(zip(dataframe["qid"], dataframe["query"]))
    print("Number of queries:", len(dataframe))
    return dataframe


def load_training_data(num_of_negative_samples=1000000):
    ce_scores = pd.read_pickle(
        "data/ms_marco/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl"
    )
    print("Read hard negatives train file")
    hard_negatives_filepath = "data/ms_marco/msmarco-hard-negatives.jsonl"
    ce_score_margin = 3.0
    num_neg_per_system = 5
    dataset = {}
    count = 0
    with open(hard_negatives_filepath, "rt") as fIn:
        for line in fIn:
            data = json.loads(line)
            qid = data["qid"]
            pos_pids = data["pos"]

            if len(pos_pids) == 0:
                continue
            pos_min_ce_score = min([ce_scores[qid][pid] for pid in data["pos"]])
            ce_score_threshold = pos_min_ce_score - ce_score_margin
            neg_pids = set()
            negs_to_use = list(data["neg"].keys())
            for system_name in negs_to_use:
                if len(neg_pids) > num_of_negative_samples:
                    break
                if system_name not in data["neg"]:
                    continue
                system_negs = data["neg"][system_name]
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
            if len(pos_pids) > 0 and len(neg_pids) > 0:
                dataset[qid] = {
                    "pos_pid": pos_pids,
                    "neg_pid": neg_pids[0:num_of_negative_samples],
                }
    return dataset


def load_training_data_from_dir(training_data_dir):
    dataset = torch.load(training_data_dir)
    return dataset


def load_item_df(dataset, corpus, queries):
    qids_all, pids_all = [], []
    for qid, pids_pos_neg in dataset.items():
        qids_all.append(qid)
        for pid in pids_pos_neg["pos_pid"]:
            if isinstance(pid, str):
                continue
            pids_all.append(pid)
        for pid in pids_pos_neg["neg_pid"]:
            if isinstance(pid, str):
                continue
            pids_all.append(pid)
    qids_all = list(set(qids_all))
    pids_all = list(set(pids_all))

    query_all = [queries[qid] for qid in qids_all]
    qids_all = ["q_{}".format(qid) for qid in qids_all]
    passage_all = [corpus[pid] for pid in pids_all]
    pids_all = ["p_{}".format(pid) for pid in pids_all]
    item_id_all = qids_all + pids_all
    title_all = query_all + passage_all
    item_df = pd.DataFrame({"ITEM_ID": item_id_all, "TITLE": title_all})
    item_df = item_df.set_index("ITEM_ID")
    return item_df


def load_user_df(dataset):
    USER_ID = list(range(len(dataset)))
    TEST_START_TIME = [1] * len(USER_ID)
    _hist_items = [["q_{}".format(qid)] for qid in dataset]
    _hist_ts = [0] * len(USER_ID)
    user_df = pd.DataFrame(
        {
            "USER_ID": USER_ID,
            "TEST_START_TIME": TEST_START_TIME,
            "_hist_items": _hist_items,
            "_hist_ts": _hist_ts,
        }
    )
    user_df = user_df.set_index("USER_ID")
    return user_df


def load_expl_response(dataset):
    USER_ID = list(range(len(dataset)))
    request_time = [2] * len(USER_ID)
    _hist_items = ["q_{}".format(qid) for qid in dataset]
    cand_items = []
    multi_label = []
    for values in dataset.values():
        if isinstance(values["pos_pid"][0], str):
            values["pos_pid"] = [int(values["pos_pid"][0].split("_")[-1])]
            candicates = ["q_{}".format(values["pos_pid"][0])] + ["p_{}".format(pid) for pid in values["neg_pid"]]
        else:
            candicates = [values["pos_pid"][0]] + values["neg_pid"]
            candicates = ["p_{}".format(pid) for pid in candicates]
        labels = [1.0] * len([values["pos_pid"][0]]) + [0.0] * len(values["neg_pid"])
        cand_items.append(candicates)
        multi_label.append(labels)
    expl_response = pd.DataFrame(
        {
            "USER_ID": USER_ID,
            "request_time": request_time,
            "_hist_items": _hist_items,
            "cand_items": cand_items,
            "multi_label": multi_label,
        }
    )
    expl_response = expl_response.set_index("USER_ID")
    return expl_response


def load_item_df_unsupervised_learning(data_name):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(data_name)
    out_dir = os.path.join(pathlib.Path("./data/scifact/").parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    if data_name == "msmarco":
        data_split = "dev"
    else:
        data_split = "test"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=data_split)
    query_title = [queries[qid] for qid in queries]
    corpus_title = []
    for pid, text in corpus.items():
        passage = text["title"] + ": " + text["text"]
        corpus_title.append(passage)
    title_all = query_title + corpus_title
    item_df = pd.DataFrame({"TITLE": title_all})
    return item_df


def main(opt):
    _batch_size = opt.batch_size
    _max_seq_length = opt.max_seq_length
    _model_name = opt.model_name
    _epochs = opt.epochs
    _lr = opt.lr
    _beta = opt.beta
    _alpha = opt.alpha
    _training_method = opt.training_method
    _training_dataset_dir = opt.training_dataset_dir
    _checkpoint = opt.checkpoint
    _do_validation = opt.do_validation

    if _training_method == "bertbpr" or _training_method == "bertmt":
        corpus = load_corpus()
        queries = load_query()

    if _training_method == "bertbpr":
        dataset = load_training_data()
        item_df = load_item_df(dataset, corpus, queries)
        user_df = load_user_df(dataset)
        expl_response = load_expl_response(dataset)

    elif _training_method == "bertmt":
        dataset_sl = load_training_data_from_dir(_training_dataset_dir)
        # dataset_sl = load_training_data_from_dir("data/ms_marco/train_dataset_human_response_2.pt")
        item_df_sl = load_item_df(dataset_sl, corpus, queries)
        item_id_all = item_df_sl.index.to_list()
        item_title_all = item_df_sl["TITLE"].to_list()
        item_df = pd.DataFrame({"ITEM_ID": item_id_all, "TITLE": item_title_all})
        item_df = item_df.set_index("ITEM_ID")
        user_df = load_user_df(dataset_sl)
        expl_response = load_expl_response(dataset_sl)

        # dataset_sl = load_training_data_from_dir(_training_dataset_dir)
        # # dataset_sl = load_training_data_from_dir("data/ms_marco/train_dataset_human_response.pt")
        # dataset_ul = load_training_data(num_of_negative_samples=1)
        # item_df_sl = load_item_df(dataset_sl, corpus, queries)
        # item_df_ul = load_item_df(dataset_ul, corpus, queries)
        # item_id_all = item_df_ul.index.to_list()
        # item_title_all = item_df_ul["TITLE"].to_list()
        # for item_id, item_title in zip(item_df_sl.index.to_list(), item_df_sl["TITLE"].to_list()):
        #     if item_id in item_id_all:
        #         continue
        #     item_id_all.append(item_id)
        #     item_title_all.append(item_title)
        # item_df = pd.DataFrame({"ITEM_ID": item_id_all, "TITLE": item_title_all})
        # item_df = item_df.set_index("ITEM_ID")
        # user_df = load_user_df(dataset_sl)
        # expl_response = load_expl_response(dataset_sl)
    else:
        item_df = load_item_df_unsupervised_learning(_training_method)

    training_arguments = {
        "lr": _lr,
        "model_name": _model_name,
        "max_length": _max_seq_length,
        "pretrained_checkpoint": _checkpoint,
        "do_validation": _do_validation,
    }

    if _training_method == "bertmt":
        bmt_main(
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
    elif _training_method == "bertbpr":
        bbpr_main(
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
    else:
        item_df = item_df['TITLE']
        VAE_training(
            item_df,
            training_args=None,
            train_set_ratio=0.9,
            max_length=training_arguments["max_length"],
            vae_beta=_beta,
            batch_size=_batch_size,
            max_epochs=_epochs,
            callbacks=None,
        )

if __name__ == "__main__":
    main(args)
