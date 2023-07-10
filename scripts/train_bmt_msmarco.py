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
import inspect

from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pathlib

from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_item_df(dataset, corpus, queries):
    qids_all, pids_all = [], []
    for qid, pids_pos_neg in dataset.items():
        qids_all.append(qid)
        for pid in pids_pos_neg["pos_pid"]:
            pids_all.append(pid)
        for pid in pids_pos_neg["neg_pid"]:
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
