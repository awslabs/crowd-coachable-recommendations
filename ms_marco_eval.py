#!/usr/bin/env python3
from unittest.mock import NonCallableMagicMock
import torch
from torch.cuda.amp import autocast
import torch.nn as nn

import pandas as pd

import sys
import statistics
from collections import Counter
import os
import math
import time
import random

import numpy as np
import argparse

from transformers import AutoTokenizer, AutoModel
from src.ccrec.models.vae_models import VAEPretrainedModel
from src.ccrec.models.bert_mt import _BertMT
from src.ccrec.models.bbpr import _BertBPR

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--model_type", type=str, default="vaepretrainedmodel")
parser.add_argument("--eval_method", type=str, default="reranking")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MaxMRRRank = 100

# %%
# generate rankings
def load_corpus():
    data_dir = "data/ms_marco/collection.tsv"
    df = pd.read_csv(data_dir, sep="\t", header=None, names=["pid", "passage"])
    data = dict(zip(df["pid"], df["passage"]))
    print("Number of passages:", len(data))
    return data


def load_dev_data():
    data_dir = "data/ms_marco/qrels.dev.tsv"
    df = pd.read_csv(
        data_dir, sep="\t", header=None, usecols=[0, 2], names=["qid", "pid"]
    )
    data = df.groupby("qid")["pid"].apply(list).to_dict()
    print("Number of development queries:", len(data))
    return data


def load_dev_top1000_data():
    data_dir = "data/ms_marco/top1000.dev"
    df = pd.read_csv(
        data_dir, sep="\t", header=None, names=["qid", "pid", "query", "passage"]
    )
    print("Number of development queries:", len(df))
    return df


def load_query():
    df_train = pd.read_csv(
        "data/ms_marco/queries.train.tsv", sep="\t", header=None, names=["qid", "query"]
    )
    df_dev = pd.read_csv(
        "data/ms_marco/queries.dev.tsv", sep="\t", header=None, names=["qid", "query"]
    )
    df_eval = pd.read_csv(
        "data/ms_marco/queries.eval.tsv", sep="\t", header=None, names=["qid", "query"]
    )
    df = pd.concat([df_train, df_dev, df_eval], ignore_index=True)
    data = dict(zip(df["qid"], df["query"]))
    print("Number of queries:", len(data))
    return data


def generate_embeddings(
    data_indices, data_dic, embedding_func, batch_size, embedding_size=768, name=None
):
    num = len(data_indices)
    num_batches = math.ceil(num / batch_size)
    embeddings = torch.zeros(num, embedding_size)
    with torch.no_grad():
        for step in range(num_batches):
            if (step % 100) == 0:
                print("Processing", step, "|", num_batches)
            indices = data_indices[step * batch_size : (step + 1) * batch_size]
            text_batch = [data_dic[index] for index in indices]
            embedding_batch = embedding_func(text_batch)
            embeddings[
                step * batch_size : (step * batch_size + len(indices)), :
            ] = embedding_batch.cpu()
    if name is not None:
        torch.save(embeddings, name)
    return embeddings


def ranking(data_dev, corpus, queries, embedding_func, batch_size):
    embedding_dim = 768
    ranking_profile = {}
    save_top = 100
    num_queries, num_passages = len(data_dev), len(corpus)
    num_query_batches = math.ceil(num_queries / batch_size)
    num_passage_batches = math.ceil(num_passages / batch_size)
    queries_ids, corpus_ids = list(data_dev.keys()), list(corpus.keys())

    queries_embeddings = generate_embeddings(
        queries_ids, queries, embedding_func, batch_size, embedding_size=embedding_dim
    )
    passage_embeddings = generate_embeddings(
        corpus_ids, corpus, embedding_func, batch_size, embedding_size=embedding_dim
    )

    corpus_ids = torch.Tensor(np.array(corpus_ids)).type(torch.int32)
    ranking_matrix = torch.zeros(batch_size, num_passages)
    for step_q in range(num_query_batches):
        if (step_q % 10) == 0:
            print("processing query: ", step_q)
        qids = queries_ids[step_q * batch_size : (step_q + 1) * batch_size]
        queries_embeddings_batch = queries_embeddings[
            step_q * batch_size : (step_q + 1) * batch_size
        ]
        queries_embeddings_batch = queries_embeddings_batch.cuda()
        num_of_rows = queries_embeddings_batch.shape[0]
        for step_p in range(num_passage_batches):
            passage_embeddings_batch = passage_embeddings[
                step_p * batch_size : (step_p + 1) * batch_size
            ]
            passage_embeddings_batch = passage_embeddings_batch.cuda()
            num_of_cols = passage_embeddings_batch.shape[0]
            if len(queries_embeddings_batch.shape) == 1:
                queries_embeddings_batch = queries_embeddings_batch.unsqueeze(0)
            if len(passage_embeddings_batch.shape) == 1:
                passage_embeddings_batch = passage_embeddings_batch.unsqueeze(0)
            scores = torch.mm(
                queries_embeddings_batch, passage_embeddings_batch.transpose(0, 1)
            )
            scores = scores.cpu()
            ranking_matrix[
                0:num_of_rows, step_p * batch_size : (step_p * batch_size + num_of_cols)
            ] = scores
        for idx, qid in enumerate(qids):
            score_array = ranking_matrix[idx]
            score_array = score_array.cuda()
            _, ordering = score_array.sort(descending=True)
            ranking_profile[qid] = corpus_ids[ordering][0:save_top].tolist()
    return ranking_profile


# data_dev is dataframe with 4 columns
# [qid, pid, query, passage]
def re_ranking(data_dev, corpus, queries, embedding_func, batch_size):
    embedding_dim = 768
    ranking_profile = {}
    save_top = 100

    data_dev_dict = data_dev.groupby("qid")["pid"].apply(list).to_dict()
    queries_ids = list(data_dev_dict.keys())
    corpus_ids = set(list(data_dev["pid"]))
    effective_corpus = {}
    for pid, passage in corpus.items():
        if pid in corpus_ids:
            effective_corpus[pid] = passage
    corpus_ids = list(effective_corpus.keys())

    queries_embeddings = generate_embeddings(
        queries_ids, queries, embedding_func, batch_size, embedding_size=embedding_dim
    )
    passage_embeddings = generate_embeddings(
        corpus_ids,
        effective_corpus,
        embedding_func,
        batch_size,
        embedding_size=embedding_dim,
    )

    num_queries, num_passages = len(queries_ids), len(corpus_ids)
    map_pid_localIndex = dict(zip(corpus_ids, list(range(num_passages))))
    queries_embeddings = queries_embeddings.cuda()
    for step_q in range(num_queries):
        if (step_q % 100) == 0:
            print("processing query: ", step_q)
        qid = queries_ids[step_q]
        pids = data_dev_dict[qid]
        if len(pids) == 1:
            ranking_profile[qid] = pids
            continue
        pids_effective = [map_pid_localIndex[pid] for pid in pids]
        queries_embedding_qid = queries_embeddings[step_q]
        passage_embeddings_pid = passage_embeddings[pids_effective]
        passage_embeddings_pid = passage_embeddings_pid.cuda()
        if len(queries_embedding_qid.shape) == 1:
            queries_embedding_qid = queries_embedding_qid.unsqueeze(0)
        if len(passage_embeddings_pid.shape) == 1:
            passage_embeddings_pid = passage_embeddings_pid.unsqueeze(0)
        scores = torch.mm(queries_embedding_qid, passage_embeddings_pid.transpose(0, 1))
        scores = scores.squeeze()
        _, ids_ordered = scores.sort(descending=True)
        pids = [pids[index] for index in ids_ordered]
        ranking_profile[qid] = pids[0:save_top]
    return ranking_profile


def extract_hard_negatives(
    embedding_func,
    corpus,
    queries,
    model_name,
    batch_size=256,
    queries_embeddings=None,
    passage_embeddings=None,
):
    embedding_dim = 768
    queries_ids, corpus_ids = list(queries.keys()), list(corpus.keys())
    if (queries_embeddings == None) or (passage_embeddings == None):
        name_q = "lightning_logs/queries_{}".format(model_name)
        name_p = "lightning_logs/passage_{}".format(model_name)
        queries_embeddings = generate_embeddings(
            queries_ids,
            queries,
            embedding_func,
            batch_size,
            embedding_size=embedding_dim,
            name=name_q,
        )
        passage_embeddings = generate_embeddings(
            corpus_ids,
            corpus,
            embedding_func,
            batch_size,
            embedding_size=embedding_dim,
            name=name_p,
        )

    # construct training dataset from 400M data
    print("Construct training dataset from 400M data")
    all_training_data = pd.read_csv(
        "data/ms_marco/qidpidtriples.train.full.2.tsv",
        sep="\t",
        header=None,
        names=["qid", "pos_pid", "neg_pid"],
    )
    all_training_pos = all_training_data.groupby("qid")["pos_pid"].apply(list).to_dict()
    all_training_neg = all_training_data.groupby("qid")["neg_pid"].apply(list).to_dict()
    train_dataset = {}
    save_top = 10
    qid_to_ptr = dict(zip(queries_ids, range(len(queries_ids))))
    pid_to_ptr = dict(zip(corpus_ids, range(len(corpus_ids))))
    for step, qid in enumerate(all_training_pos.keys()):
        if (step % 10000) == 0:
            print("Processing:", step, "|", len(all_training_pos))
        pos_pids = list(set(all_training_pos[qid]))
        neg_pids = list(set(all_training_neg[qid]))
        if len(pos_pids) == 0 or len(neg_pids) == 0:
            continue
        pos_pids = [pos_pids[0]]
        if len(neg_pids) < save_top:
            train_dataset[qid] = {"pos_pid": pos_pids, "neg_pid": neg_pids}
            continue
        embedding_qid = queries_embeddings[qid_to_ptr[qid]]
        neg_pid_ptr = [pid_to_ptr[idx] for idx in neg_pids]
        embedding_neg_pids = passage_embeddings[neg_pid_ptr]
        embedding_qid, embedding_neg_pids = (
            embedding_qid.cuda(),
            embedding_neg_pids.cuda(),
        )
        if len(embedding_qid.shape) == 1:
            embedding_qid = embedding_qid.unsqueeze(0)
        if len(embedding_neg_pids.shape) == 1:
            embedding_neg_pids = embedding_neg_pids.unsqueeze(0)

        scores = torch.mm(embedding_qid, embedding_neg_pids.transpose(0, 1))
        scores = scores.squeeze()
        _, ids_ordered = scores.sort(descending=True)
        neg_pids = [neg_pids[index] for index in ids_ordered]
        neg_pids = neg_pids[0:save_top]
        train_dataset[qid] = {"pos_pid": pos_pids, "neg_pid": neg_pids}
    torch.save(train_dataset, "data/ms_marco/train_dataset_{}.pt".format(model_name))


def process_dev_data_for_reranking(dev_data, top1000_dev_data):
    effective_queries = set(list(top1000_dev_data["qid"]))
    dataset = {}
    for qid, pids in dev_data.items():
        if qid in effective_queries:
            dataset[qid] = pids
    return dataset


def convert_dev_data_to_msmarco(data, data_folder):
    writing_dir = os.path.join(data_folder, "eval", "dev_data.tsv")
    with open(writing_dir, "w") as w:
        for qid in data:
            pids = data[qid]
            for pid in pids:
                w.write("{}\t{}\n".format(qid, pid))
    return writing_dir


def convert_ranking_to_msmarco(data, data_folder):
    writing_dir = os.path.join(data_folder, "eval", "rankings.tsv")
    with open(writing_dir, "w") as w:
        for qid in data:
            passage_indices = data[qid]
            for rank, pid in enumerate(passage_indices):
                rank += 1
                w.write("{}\t{}\t{}\n".format(qid, pid, rank))
    return writing_dir


# %%
# from files
def load_reference_from_stream(f):
    qids_to_relevant_passageids = {}
    for line in f:
        try:
            line = line.strip().split("\t")
            qid = int(line[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(line[1]))
        except:
            raise IOError('"%s" is not valid format' % line)
    return qids_to_relevant_passageids


def load_reference(path_to_reference):
    with open(path_to_reference, "r") as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids


def load_candidate_from_stream(f):
    qid_to_ranked_candidate_passages = {}
    for line in f:
        try:
            line = line.strip().split("\t")
            qid = int(line[0])
            pid = int(line[1])
            rank = int(line[2])
            if qid in qid_to_ranked_candidate_passages:
                pass
            else:
                tmp = [0] * 1000
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank - 1] = pid
        except:
            raise IOError('"%s" is not valid format' % line)
    return qid_to_ranked_candidate_passages


def load_candidate(path_to_candidate):
    with open(path_to_candidate, "r") as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages


def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    message = ""
    allowed = True
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())
    for qid in qids_to_ranked_candidate_passages:
        duplicate_pids = set(
            [
                item
                for item, count in Counter(
                    qids_to_ranked_candidate_passages[qid]
                ).items()
                if count > 1
            ]
        )
        if len(duplicate_pids - set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0]
            )
            allowed = False
    return allowed, message


def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR += 1 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
    if len(ranking) == 0:
        raise IOError(
            "No matching QIDs found. Are you sure you are scoring the evaluation set?"
        )

    MRR = MRR / len(qids_to_relevant_passageids)
    all_scores["MRR @100"] = MRR
    all_scores["QueriesRanked"] = len(qids_to_ranked_candidate_passages)
    return all_scores


def compute_metrics_from_files(
    path_to_reference, path_to_candidate, perform_checks=True
):
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(
            qids_to_relevant_passageids, qids_to_ranked_candidate_passages
        )
        if message != "":
            print(message)
    return compute_metrics(
        qids_to_relevant_passageids, qids_to_ranked_candidate_passages
    )


# %%
def main(args):
    model_name_ = args.model_name
    model_type = args.model_type
    batch_size = args.batch_size
    model_dir_ = args.model_dir
    eval_method = args.eval_method

    _gpu_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        batch_size = batch_size * len(_gpu_ids)

    tokenizer = AutoTokenizer.from_pretrained(model_name_)
    tokenize_func = lambda x: tokenizer(
        x, padding="max_length", max_length=300, truncation=True, return_tensors="pt"
    )

    state = torch.load(model_dir_) if model_dir_ is not None else None
    if model_type == "automodel":
        model = AutoModel.from_pretrained(model_name_)
        if state is not None:
            model.load_state_dict(state)
    elif model_type == "bertbpr":
        model = _BertBPR(None, model_name=model_name_)
        if state is not None:
            model.load_state_dict(state["state_dict"])
        model = model.item_tower
    elif model_type == "bertmt":
        model = _BertMT(None, model_name=model_name_)
        if state is not None:
            model.load_state_dict(state["state_dict"])
        model = model.item_tower
    elif model_type == "vaepretrainedmodel":
        model = VAEPretrainedModel.from_pretrained(model_name_)
        if state is not None:
            # model.load_state_dict(state['state_dict'])
            model.load_state_dict(state)

    model = torch.nn.DataParallel(model, device_ids=_gpu_ids)
    model = model.cuda(_gpu_ids[0]) if _gpu_ids != [] else model

    if model_type == "automodel":

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        def transform(x):
            tokens = tokenize_func(x)
            model_outputs = model(
                input_ids=tokens["input_ids"].cuda(),
                attention_mask=tokens["attention_mask"].cuda(),
            )
            outputs = mean_pooling(model_outputs, tokens["attention_mask"].cuda())
            return outputs

    elif model_type == "bertbpr":

        def transform(x):
            tokens = tokenize_func(x)
            outputs = model(**tokens)
            return outputs

    elif model_type == "bertmt":

        def transform(x):
            tokens = tokenize_func(x)
            outputs = model(**tokens, testing=True, output_step="embedding")
            return outputs

    elif model_type == "vaepretrainedmodel":

        def transform(x):
            tokens = tokenize_func(x)
            outputs = model(**tokens, testing=False, return_embedding=True)
            return outputs

    embedding_func = lambda x: transform(x)

    data_folder = "data/ms_marco"
    corpus = load_corpus()
    dev_data = load_dev_data()
    dev_top1000_data = load_dev_top1000_data()
    query_data = load_query()

    time_start = time.time()
    with autocast():
        if eval_method == "ranking":
            ranking_profile = ranking(
                dev_data, corpus, query_data, embedding_func, batch_size
            )
        elif eval_method == "reranking":
            ranking_profile = re_ranking(
                dev_top1000_data, corpus, query_data, embedding_func, batch_size
            )
            dev_data = process_dev_data_for_reranking(dev_data, dev_top1000_data)
        elif eval_method == "find_hard_negatives":
            extract_hard_negatives(
                embedding_func, corpus, query_data, model_type, batch_size
            )
    time_end = time.time()
    print("Time used for ranking:", time_end - time_start)

    path_to_reference = convert_dev_data_to_msmarco(dev_data, data_folder)
    path_to_candidate = convert_ranking_to_msmarco(ranking_profile, data_folder)

    all_scores = compute_metrics_from_files(path_to_reference, path_to_candidate)
    print(
        "MRR @100:",
        all_scores["MRR @100"],
        "queries ranked:",
        all_scores["QueriesRanked"],
    )


if __name__ == "__main__":
    main(args)
