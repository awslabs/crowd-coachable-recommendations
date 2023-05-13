import re
import csv
import torch
import os
import numpy as np
import pandas as pd
import warnings
from torch.cuda.amp import autocast

os.environ.setdefault("CCREC_MAX_LENGTH", "256")
os.environ.setdefault("CCREC_SIM_TYPE", "dot")
os.environ.setdefault("CCREC_EMBEDDING_TYPE", "mean_pooling")
os.environ.setdefault("CCREC_DISPLAY_LENGTH", "250")

from sklearn.metrics import confusion_matrix
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from uuid import uuid1
import matplotlib.pyplot as plt

from al_commons import parse_al_args

import inspect
from beir.retrieval.evaluation import EvaluateRetrieval

from transformers import AutoTokenizer, AutoModel
from ccrec.models.bert_mt import bmt_main
from ccrec.models.bbpr import bbpr_main
from ccrec.models.bert_mt import _BertMT
from ccrec.models.bbpr import _BertBPR
from ccrec.util.data_parallel import DataParallel


from train_bmt_msmarco import (
    load_corpus,
    load_query,
    load_item_df,
    load_user_df,
    load_expl_response,
)
from ms_marco_eval import ranking, load_data

(
    MODEL_NAME,
    DATA_NAME,
    RESULTS_DIR,
    STEP,
    ranking_profile_bm25,
    qids_split,
    N_REPEATS,
    REPEAT_SEED,
    number_of_qid_split_batch,
    NUM_EPOCHS,
    DRYRUN,
) = parse_al_args()


corpus, queries, qrels = load_data(DATA_NAME)

train_pre = torch.load(
    f"{RESULTS_DIR}/data_iteration_{STEP}/train_data_human_response.pt"
)


def training(
    train_dataset,
    epochs,
    model_checkpoint,
    save_dir,
    corpus,
    queries,
    model_selection,
):
    item_df = load_item_df(train_dataset, corpus, queries)
    user_df = load_user_df(train_dataset)
    expl_response = load_expl_response(train_dataset)

    if model_selection == "vae":
        training_arguments = {
            "lr": 2e-5,
            "model_name": "distilbert-base-uncased",
            "max_length": int(os.environ.get("CCREC_MAX_LENGTH", 300)),
            "pretrained_checkpoint": model_checkpoint,
            "do_validation": False,
            "log_directory": save_dir,
        }
    elif "contriever" in model_selection:
        training_arguments = {
            "lr": 2e-5,
            "model_name": model_selection,
            "max_length": int(os.environ.get("CCREC_MAX_LENGTH", 300)),
            "pretrained_checkpoint": None,
            "do_validation": False,
            "log_directory": save_dir,
        }

    _batch_size = 30
    _epochs = epochs
    _alpha = 1.0
    _beta = 2e-3

    train_main = globals()[
        os.environ.get("CCREC_TRAIN_MAIN", "bmt_main")
    ]  # or bbpr_main

    _, _, model = train_main(
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


model = training(
    train_pre,
    NUM_EPOCHS,
    model_checkpoint=None,
    save_dir=None,
    corpus=corpus,
    queries=queries,
    model_selection=MODEL_NAME,
)

print(type(model.model.item_tower))

model_save_path = f"{RESULTS_DIR}/data_iteration_{STEP}/state-dict.pth"
torch.save(
    model.model.item_tower.state_dict(),
    model_save_path,
)
print(model_save_path)
