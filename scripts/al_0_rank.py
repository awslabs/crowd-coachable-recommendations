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

print("total num queries", len(np.hstack(qids_split)))
print(
    "this batch size", len(qids_split[STEP]) if STEP < number_of_qid_split_batch else -1
)


## evaluation


def generate_ranking_profile(model, model_name, corpus, queries):
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

    if model_name == "vae":  # legacy
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = (
            model.item_tower
            if hasattr(model, "item_tower")
            else model.model.item_tower
            if hasattr(model, "model")
            else model
        )
        model.eval()
        model = DataParallel(model.cuda(), device_ids=_gpu_ids).cache_replicas()

        def embedding_func(x):
            tokens = tokenizer(x, **tokenizer_kw)
            outputs = model(**tokens, output_step=os.environ["CCREC_EMBEDDING_TYPE"])
            return outputs

    elif "contriever" in model_name:
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

    ranking_profile = ranking(corpus, queries, embedding_func, batch_size)

    return ranking_profile


print("RESULTS_DIR", RESULTS_DIR)
previous_working_dir = f"{RESULTS_DIR}/data_iteration_{STEP-1}"
current_working_dir = f"{RESULTS_DIR}/data_iteration_{STEP}"

os.makedirs(current_working_dir, exist_ok=True)


path_to_ranking_profile = f"{current_working_dir}/ranking_profile.pt"
print(path_to_ranking_profile)
if os.path.isfile(path_to_ranking_profile):
    ranking_profile = torch.load(path_to_ranking_profile)
else:
    model = _BertMT(None, model_name=MODEL_NAME)
    if STEP > 0:
        model.item_tower.load_state_dict(
            torch.load(f"{previous_working_dir}/state-dict.pth")
        )
    with autocast():
        ranking_profile = generate_ranking_profile(model, MODEL_NAME, corpus, queries)
    torch.save(ranking_profile, path_to_ranking_profile)


evaluator = EvaluateRetrieval(None)
mrr = evaluator.evaluate_custom(qrels, ranking_profile, [1, 5, 10, 100], metric="mrr")
for name, value in mrr.items():
    print("{}".format(name), ":", value)


## creation

ranks_rng = np.random.RandomState(STEP)
corpus_keys = list(corpus.keys())


def filter_string(text):
    new_text = re.sub(r"[^a-zA-Z0-9 ,:.;?$!()&\[\]]", "", text)
    return new_text[: int(os.environ["CCREC_DISPLAY_LENGTH"])]


header = ["query", "passage-1", "passage-2", "passage-3", "passage-4"]
rows = []

id_track = dict()
for qid in ranking_profile:
    if qid not in qids_split[STEP % number_of_qid_split_batch]:
        continue

    ranks = list(ranking_profile[qid].keys())
    ranks_bm25 = list(ranking_profile_bm25[qid].keys())

    cands = ranks[0:2].copy()
    for pid in ranks_bm25:
        if len(cands) == 3:
            break
        if pid not in cands:
            cands.append(pid)

    while len(cands) < 4:
        pid = corpus_keys[ranks_rng.choice(len(corpus_keys))]
        if pid not in cands:
            cands.append(pid)

    query_text = queries[qid]
    passages = [filter_string(corpus[pid]) for pid in cands]

    row = [query_text, *passages]
    rows.append(row)
    id_track[query_text] = "q_{}".format(qid)
    for pid, passage in zip(cands, passages):
        id_track[passage] = "p_{}".format(pid)

id_track_file_name = f"{current_working_dir}/id_track.pt"
torch.save(id_track, id_track_file_name)

request_orig = pd.DataFrame(rows, columns=header)
print(request_orig.iloc[0])
request_orig.to_csv(f"{current_working_dir}/request_orig.csv", index=False)

rng = np.random.RandomState(REPEAT_SEED)
request_perm = pd.DataFrame(
    [
        [row[0]] + rng.permutation(row[1:]).tolist()
        for i in range(N_REPEATS)
        for row in rows
    ],
    columns=header,
)
print(request_perm.iloc[0])
request_perm.to_csv(f"{current_working_dir}/request_perm.csv", index=False)
