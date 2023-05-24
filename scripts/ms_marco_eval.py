#!/usr/bin/env python3
import torch
from torch.cuda.amp import autocast
import torch.nn as nn

import pandas as pd

import os
import math
import time
import sys

import numpy as np
import argparse
import json
import warnings
from collections import Counter

from datasets import Dataset

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
except ImportError:
    warnings.warn("package beir not found")

import pathlib

from transformers import AutoTokenizer, AutoModel
from ccrec.models.bert_mt import _BertMT
from ccrec.models.bbpr import _BertBPR
from bm_25 import BM25

from ccrec.util.amazon_review_prime_pantry import get_item_df

try:
    from beir.retrieval.evaluation import EvaluateRetrieval
except ImportError:
    warnings.warn("package beir not found")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# %%
def load_data(task, data_split=None):
    if task == "prime_pantry":
        data_root = (
            "/".join(os.path.realpath(__file__).split("/")[:-2])
            + "/data/amazon_review_prime_pantry"
        )
        print("data_root", data_root)
        item_df, _ = get_item_df(data_root, shorten_brand_name=False)
        corpus = item_df["TITLE"].to_dict()
        queries = item_df["TITLE"].to_dict()
        block_dict = {
            k: item_df[item_df["BRAND"] == v["BRAND"]].index.tolist()
            for k, v in item_df.iterrows()
        }
        reviews = pd.read_json(f"{data_root}/Prime_Pantry.json.gz", lines=True)
        reviews = reviews[reviews["asin"].isin(item_df.index)].copy()

        reviews["past_asin"] = (
            reviews.sort_values("unixReviewTime", kind="stable")
            .groupby("reviewerID")["asin"]
            .shift(1)
        )

        review_bigram = reviews.dropna(subset=["asin", "past_asin"])
        review_bigram = review_bigram[
            review_bigram.apply(
                lambda x: x["past_asin"] not in block_dict[x["asin"]],
                axis=1,
            )
        ]

        qrels = review_bigram.groupby("past_asin")["asin"].apply(
            lambda x: Counter(x).most_common(3)
        )
        qrels = {
            k: ({kk: vv for kk, vv in qrels.loc[k]} if k in qrels.index else {})
            for k in queries.keys()
        }

        qids_split = [
            item_df.sample(frac=1, random_state=42).index.tolist()[
                i
                * int(np.ceil(len(item_df) / 4)) : (i + 1)
                * int(np.ceil(len(item_df) / 4))
            ]
            for i in range(4)
        ]
        print([x[:5] for x in qids_split])
        return corpus, queries, qrels, block_dict, qids_split, item_df

    elif task == "ms_marco_full_ranking":
        corpus = pd.read_csv(
            "data/ms_marco/collection.tsv",
            sep="\t",
            header=None,
            names=["pid", "passage"],
        )
        corpus = dict(zip(corpus["pid"], corpus["passage"]))
        queries = pd.read_csv(
            "data/ms_marco/queries.dev.tsv",
            sep="\t",
            header=None,
            names=["qid", "query"],
        )
        queries = dict(zip(queries["qid"], queries["query"]))
        qrels = pd.read_csv(
            "data/ms_marco/qrels.dev.tsv",
            sep="\t",
            header=None,
            usecols=[0, 2],
            names=["qid", "pid"],
        )
        qrels = qrels.groupby("qid")["pid"].apply(list).to_dict()
    elif (
        task == "ms_marco_find_hard_negatives"
        or task == "ms_marco_find_hard_negatives_for_devs"
    ):
        corpus = pd.read_csv(
            "data/ms_marco/collection.tsv",
            sep="\t",
            header=None,
            names=["pid", "passage"],
        )
        corpus = dict(zip(corpus["pid"], corpus["passage"]))
        queries_train = pd.read_csv(
            "data/ms_marco/queries.train.tsv",
            sep="\t",
            header=None,
            names=["qid", "query"],
        )
        queries_dev = pd.read_csv(
            "data/ms_marco/queries.dev.tsv",
            sep="\t",
            header=None,
            names=["qid", "query"],
        )
        queries = pd.concat([queries_train, queries_dev], ignore_index=True)
        queries = dict(zip(queries["qid"], queries["query"]))
        qrels = None
    else:
        data_name = task
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            data_name
        )
        out_dir = os.path.join(
            pathlib.Path("./data/scifact/").parent.absolute(), "datasets"
        )
        data_path = util.download_and_unzip(url, out_dir)
        if data_split is None:
            if data_name == "msmarco":
                data_split = "dev"
            else:
                data_split = "test"
        corpus_, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split=data_split
        )
        corpus = dict()
        for pid, passage in corpus_.items():
            if passage["title"] == "":
                corpus[pid] = passage["text"]
            else:
                corpus[pid] = passage["title"] + ": " + passage["text"]
    return corpus, queries, qrels


def generate_embeddings(
    data_indices, data_dic, embedding_func, batch_size, embedding_size=768, name=None
):
    num = len(data_indices)
    num_batches = math.ceil(num / batch_size)
    embeddings = []
    tic = time.time()
    with torch.no_grad():
        for step in range(num_batches):
            if step != 0 and step & (step - 1) == 0:  # power of 2
                print(
                    f"Processed {step * batch_size} | {num}",
                    f"t={time.time() - tic:.1f}s",
                    f"/ {(time.time() - tic) * num / (step * batch_size):.1f}s",
                )
            indices = data_indices[step * batch_size : (step + 1) * batch_size]
            text_batch = [data_dic[index] for index in indices]
            embedding_batch = embedding_func(text_batch)
            embeddings.append(
                torch.as_tensor(embedding_batch).to(
                    "cpu",
                    non_blocking=bool(int(os.environ.get("CCREC_NON_BLOCKING", "1"))),
                )
            )
    torch.cuda.synchronize()
    print(f"Processed total {num} t={time.time() - tic:.1f}s")
    embeddings = torch.vstack(embeddings)
    if name is not None:
        torch.save(embeddings, name)
    return embeddings


def cos_sim(a: torch.Tensor, b: torch.Tensor):
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def ranking_bm25(corpus, queries):
    ranking_profile = {}
    model = BM25()
    print("Fitting BM-25 model")
    model.fit(list(corpus.values()))
    print("Retrieval with BM-25 model")
    queries_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    num_queries = len(queries_ids)
    for step_q in range(num_queries):
        if (step_q % 100) == 0:
            print("processing query: {} | {}".format(step_q, num_queries))
        qid = queries_ids[step_q]
        query_sentence = queries[qid]
        solution = model.transform(query_sentence)
        solution = torch.Tensor(solution)
        ordered_scores, ordering = solution.sort(descending=True)
        ordered_scores, ordering = ordered_scores[0:1001], ordering[0:1001]
        ordered_pids = [corpus_ids[idx] for idx in ordering]
        ordered_scores = ordered_scores.numpy().tolist()
        ranking_profile[qid] = dict(zip(ordered_pids, ordered_scores))
    return ranking_profile


def ranking(corpus, queries, embedding_func, batch_size, block_dict=None):
    embedding_dim = 768
    num_queries, num_passages = len(queries), len(corpus)
    num_query_batches = math.ceil(num_queries / batch_size)
    num_passage_batches = math.ceil(num_passages / batch_size)
    queries_ids, corpus_ids = list(queries.keys()), list(corpus.keys())

    queries_embeddings = generate_embeddings(
        queries_ids, queries, embedding_func, batch_size, embedding_size=embedding_dim
    )
    passage_embeddings = generate_embeddings(
        corpus_ids, corpus, embedding_func, batch_size, embedding_size=embedding_dim
    )

    ranking_profile = {}
    ranking_matrix = torch.zeros(num_queries, num_passages)
    queries_embeddings = queries_embeddings.cuda()
    for step in range(num_passage_batches):
        passage_embeddings_batch = passage_embeddings[
            step * batch_size : (step + 1) * batch_size
        ]
        passage_embeddings_batch = passage_embeddings_batch.cuda()
        num_of_pasg = passage_embeddings_batch.shape[0]
        if os.environ["CCREC_SIM_TYPE"] == "cos":
            scores = cos_sim(queries_embeddings, passage_embeddings_batch)
        else:  # dot
            scores = queries_embeddings @ passage_embeddings_batch.T
        ranking_matrix[
            0:num_queries, step * batch_size : (step * batch_size + num_of_pasg)
        ] = scores.cpu()
    if block_dict is not None:
        print("using block_dict")
    for step, qid in enumerate(queries_ids):
        # print(step, "|", num_queries)
        scores = ranking_matrix[step]
        if block_dict is not None:
            block_ind = pd.Index(corpus_ids).get_indexer(block_dict[qid])
            assert -1 not in block_ind, "block id not found"
            scores[block_ind] = -1e6
        scores = scores.cuda()
        ordered_scores, ordering = scores.sort(descending=True)
        ordered_scores, ordering = ordered_scores[0:1001], ordering[0:1001]
        ordered_pids = [corpus_ids[idx] for idx in ordering]
        ordered_scores, ordered_pids = ordered_scores, ordered_pids
        ordered_scores = ordered_scores.cpu().numpy().tolist()
        ranking_profile[qid] = dict(zip(ordered_pids, ordered_scores))
    return ranking_profile


def load_training_data():
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
                    "neg_pid": neg_pids,
                }
    return dataset


def extract_hard_negatives(
    embedding_func,
    corpus,
    queries,
    model_name,
    batch_size=256,
    devs=None,
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

    save_top = 10
    qid_to_ptr = dict(zip(queries_ids, range(len(queries_ids))))
    pid_to_ptr = dict(zip(corpus_ids, range(len(corpus_ids))))
    if devs is None:
        dataset_old = load_training_data()
        train_dataset = {}
        for step, qid in enumerate(dataset_old.keys()):
            if (step % 10000) == 0:
                print("Processing:", step, "|", len(dataset_old))
            pos_pids = dataset_old[qid]["pos_pid"]
            neg_pids = dataset_old[qid]["neg_pid"]
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
            if os.environ["CCREC_SIM_TYPE"] == "cos":
                scores = cos_sim(embedding_qid, embedding_neg_pids)
            else:  # dot
                scores = embedding_qid @ embedding_neg_pids.T
            scores = scores.squeeze()
            _, ids_ordered = scores.sort(descending=True)
            neg_pids = [neg_pids[index] for index in ids_ordered]
            neg_pids = neg_pids[0:save_top]
            train_dataset[qid] = {"pos_pid": pos_pids, "neg_pid": neg_pids}
        torch.save(
            train_dataset, "data/ms_marco/train_dataset_{}.pt".format(model_name)
        )
    else:
        train_dataset = {}
        num_queries, num_passages = len(devs), len(corpus)
        num_query_batches = math.ceil(num_queries / batch_size)
        num_passage_batches = math.ceil(num_passages / batch_size)
        dev_queries_ids, corpus_ids = list(devs.keys()), list(corpus.keys())
        corpus_ids = torch.Tensor(np.array(corpus_ids)).type(torch.int32)
        ranking_matrix = torch.zeros(batch_size, num_passages)
        for step_q in range(num_query_batches):
            print("processing query: ", step_q, "|", num_query_batches)
            qids = dev_queries_ids[step_q * batch_size : (step_q + 1) * batch_size]
            ptr_qids = [qid_to_ptr[index] for index in qids]
            queries_embeddings_batch = queries_embeddings[ptr_qids]
            queries_embeddings_batch = queries_embeddings_batch.cuda()
            num_of_rows = queries_embeddings_batch.shape[0]
            for step_p in range(num_passage_batches):
                passage_embeddings_batch = passage_embeddings[
                    step_p * batch_size : (step_p + 1) * batch_size
                ]
                passage_embeddings_batch = passage_embeddings_batch.cuda()
                num_of_cols = passage_embeddings_batch.shape[0]
                if os.environ["CCREC_SIM_TYPE"] == "cos":
                    scores = cos_sim(queries_embeddings_batch, passage_embeddings_batch)
                else:  # dot
                    scores = queries_embeddings_batch @ passage_embeddings_batch.T
                scores = scores.cpu()
                ranking_matrix[
                    0:num_of_rows,
                    step_p * batch_size : (step_p * batch_size + num_of_cols),
                ] = scores
            for idx, qid in enumerate(qids):
                score_array = ranking_matrix[idx]
                score_array = score_array.cuda()
                _, ordering = score_array.sort(descending=True)
                neg_pids = corpus_ids[ordering][0:save_top].tolist()
                pos_pid = devs[qid][0]
                if pos_pid in neg_pids:
                    neg_pids.remove(pos_pid)
                train_dataset[qid] = {"pos_pid": [pos_pid], "neg_pid": neg_pids}
        torch.save(train_dataset, "data/ms_marco/dev_dataset_{}.pt".format(model_name))
