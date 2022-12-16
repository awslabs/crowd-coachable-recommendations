#!/usr/bin/env python3
import torch
import pandas as pd
import os
import sys
import random
from torch.cuda.amp import autocast
import argparse

import inspect
from beir.retrieval.evaluation import EvaluateRetrieval

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from transformers import AutoTokenizer, AutoModel
from src.ccrec.models.bert_mt import bmt_main
from src.ccrec.models.bbpr import bbpr_main
from src.ccrec.models.bert_mt import _BertMT
from src.ccrec.models.bbpr import _BertBPR

from train_bmt_msmarco import (
    load_corpus,
    load_query,
    load_item_df,
    load_user_df,
    load_expl_response,
)

from ms_marco_eval import ranking, load_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--n_steps", type=int, default=1)
# parser.add_argument("--accu_level", type=float, default=1.)
# parser.add_argument("--accu_level", type=float, default=0.7)
parser.add_argument("--accu_level", type=float, default=0.6)
# parser.add_argument("--accu_level", type=float, default=0.5)
parser.add_argument("--epochs", type=int, default=10)

args = parser.parse_args()

# %%
# train model
def training(
    train_dataset, epochs, model_checkpoint, save_dir, corpus=None, queries=None
):
    if corpus is None or queries is None:
        corpus = load_corpus()
        queries = load_query()

    item_df = load_item_df(train_dataset, corpus, queries)
    user_df = load_user_df(train_dataset)
    expl_response = load_expl_response(train_dataset)

    training_arguments = {
        "lr": 2e-5,
        "model_name": "distilbert-base-uncased",
        "max_length": 300,
        "pretrained_checkpoint": model_checkpoint,
        "do_validation": False,
        "log_directory": save_dir,
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
def generate_ranking_profile(model, save_dir):
    batch_size = 1024

    _gpu_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        batch_size = batch_size * len(_gpu_ids)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenize_func = lambda x: tokenizer(
        x, padding=True, max_length=300, truncation=True, return_tensors="pt"
    )

    model.eval()
    model = torch.nn.DataParallel(model, device_ids=_gpu_ids)
    model = model.cuda(_gpu_ids[0]) if _gpu_ids != [] else model

    def transform(x):
        tokens = tokenize_func(x)
        outputs, _ = model(**tokens, output_step="return_mean_std")
        return outputs

    embedding_func = lambda x: transform(x)

    # corpus, queries, qrels = load_data("msmarco")
    corpus, queries, qrels = load_data("hotpotqa")

    ranking_profile = ranking(corpus, queries, embedding_func, batch_size)

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
    qids, qrels, ranking_profile, ranking_profile_2=None, num_of_samples_from_model=4
):
    train_data = dict()
    for qid in qids:
        pids = list(ranking_profile[qid].keys())
        pids = pids[0:4]
        if ranking_profile_2 is not None:
            pids_2 = list(ranking_profile_2[qid].keys())
            pids = pids[0:num_of_samples_from_model]
            for pid in pids_2:
                if len(pids) == 4:
                    break
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
    return train_data


def generate_train_data_with_accu_level(train_data, accu_level, qrels):
    num_of_correct_data = 0
    qids_correct_all = []
    for qid, item in train_data.items():
        labels = list(qrels[str(qid)].keys())
        pos_pid = item["pos_pid"]
        neg_pid = item["neg_pid"]
        if any(pid in labels for pid in pos_pid):
            num_of_correct_data += 1
            qids_correct_all.append(qid)
    print(
        "number of correct samples: {} / {}".format(
            num_of_correct_data, len(train_data)
        )
    )
    random.shuffle(qids_correct_all)
    number_of_wrong_data = int(num_of_correct_data * (1 - accu_level))
    qids_wrong = qids_correct_all[0:number_of_wrong_data]

    for qid, item in train_data.items():
        if qid not in qids_wrong:
            continue
        pos_pid = item["pos_pid"]
        neg_pid = item["neg_pid"]
        random.shuffle(neg_pid)
        pids = neg_pid + pos_pid
        pos_pid = [pids[0]]
        neg_pid = pids[1:]
        train_data[qid] = {"pos_pid": pos_pid, "neg_pid": neg_pid}
    return train_data


def combine_train_data(train_data_pre, train_data_new):
    for qid, item in train_data_new.items():
        train_data_pre[qid] = item
    return train_data_pre


# %%
# main function
def main(opt):
    n_steps = opt.n_steps
    accu_level = opt.accu_level
    epochs = opt.epochs

    corpus, queries, qrels = load_data("hotpotqa")
    batch_size = 1852
    # batch_size = 865

    qids_all = list(qrels.keys())
    num_of_train_data = len(qids_all)
    qids_split = [
        qids_all[x : x + batch_size] for x in range(0, num_of_train_data, batch_size)
    ]
    number_of_qid_split_batch = len(qids_split)

    print("Accuracy level:", accu_level)
    print("Total number of iterations:", number_of_qid_split_batch * n_steps)

    bm25_dir = "hotpotqa_results_oracle_agent_{}/data_iteration_0/ranking_profile_bm25.pt".format(
        accu_level
    )
    ranking_profile_bm25 = torch.load(bm25_dir)

    # load initial model
    # model_init_dir = "exp_results_oracle_agent_{}/data_iteration_0/model/pytorch_model.bin".format(accu_level)
    model_init_dir = "hotpotqa_results_oracle_agent_{}/data_iteration_0/model/checkpoints/epoch=4-step=20800.ckpt".format(
        accu_level
    )
    state = torch.load(model_init_dir)
    model = _BertMT(None, model_name="distilbert-base-uncased")
    # model.item_tower.ae_model.load_state_dict(state)
    model.load_state_dict(state["state_dict"])

    for ite in range(n_steps):
        for step in range(number_of_qid_split_batch):
            step_ = int(ite * number_of_qid_split_batch + step)
            previous_working_dir = (
                "hotpotqa_results_oracle_agent_{}/data_iteration_{}".format(
                    accu_level, step_ - 1
                )
            )
            current_working_dir = (
                "hotpotqa_results_oracle_agent_{}/data_iteration_{}".format(
                    accu_level, step_
                )
            )

            if not os.path.exists(current_working_dir):
                os.makedirs(current_working_dir)

            # generate ranking profile
            print("Generate ranking profile at step:", step_)
            save_dir = os.path.join(current_working_dir, "ranking_profile.pt")
            if os.path.isfile(save_dir):
                ranking_profile = torch.load(save_dir)
            else:
                model_ = model.item_tower
                with autocast():
                    ranking_profile = generate_ranking_profile(model_, save_dir)

            # generate training data
            print("Generate training data at step:", step_)

            num_of_samples_from_model = 2
            print("using number of samples from model:", num_of_samples_from_model)
            train_data_oracle = generate_train_data(
                qids_split[step],
                qrels,
                ranking_profile,
                ranking_profile_bm25,
                num_of_samples_from_model,
            )

            train_data = generate_train_data_with_accu_level(
                train_data_oracle, accu_level, qrels
            )

            if step_ > 0:
                train_data_prev_dir = os.path.join(
                    previous_working_dir, "training_data.pt"
                )
                train_data_prev = torch.load(train_data_prev_dir)
                train_data = combine_train_data(train_data_prev, train_data)

            train_data_dir = os.path.join(current_working_dir, "training_data.pt")
            torch.save(train_data, train_data_dir)

            # train model
            print("Training model at step:", step_)
            print("Number of training data:", len(train_data))
            save_train_dir = "al_oracle_agent_{}_{}".format(step_, accu_level)
            bertmt = training(
                train_data,
                epochs,
                model_checkpoint=model_init_dir,
                save_dir=save_train_dir,
                corpus=corpus,
                queries=queries,
            )
            model = bertmt.model

    model_ = model.item_tower
    save_dir = os.path.join(current_working_dir, "ranking_profile_final.pt")
    with autocast():
        ranking_profile = generate_ranking_profile(model_, save_dir)


if __name__ == "__main__":
    main(args)
