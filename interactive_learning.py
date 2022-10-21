import pandas as pd, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest, rime, ccrec, ccrec.models, ccrec.models.bbpr
from ccrec import InteractiveExperiment, env
from ccrec.env import create_zero_shot, Env, parse_response
from ccrec.env.i2i_env import get_notebook_name
from ccrec.models.vae_models import VAEPretrainedModel

import argparse
import pandas as pd
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--beta", default=2e-3, type=float)
parser.add_argument("--training_method", default="bertmt", type=str)
parser.add_argument("--freeze_bert", default=0, type=int)
parser.add_argument("--input_type", default="text", type=str)
parser.add_argument("--training_dataset", default=None, type=str)
args = parser.parse_args()

# %%
# load MS_MARCO data
def load_corpus(data_root):
    data_dir = os.path.join(data_root, "collection.tsv")
    dataframe = pd.read_csv(data_dir, sep="\t", header=None, names=["pid", "TITLE"])
    print("Number of passages:", len(dataframe))
    return dataframe


def load_query(data_root):
    dataframe_train = pd.read_csv(
        os.path.join(data_root, "queries.train.tsv"),
        sep="\t",
        header=None,
        names=["qid", "TITLE"],
    )
    dataframe_dev = pd.read_csv(
        os.path.join(data_root, "queries.dev.tsv"),
        sep="\t",
        header=None,
        names=["qid", "TITLE"],
    )
    dataframe = pd.concat([dataframe_train, dataframe_dev])
    print("Number of queries:", len(dataframe))
    return dataframe


def load_dev_top1000_data(data_root):
    df = pd.read_csv(
        os.path.join(data_root, "top1000.dev"),
        sep="\t",
        header=None,
        names=["qid", "pid", "query", "passage"],
    )
    print("Number of development queries:", len(df))
    return df


# %%
def get_item_df(data_root="data/ms_marco"):
    corpus_df = load_corpus(data_root)
    queries_df = load_query(data_root)
    queries_df.loc[:, "qid"] = queries_df["qid"].apply(lambda x: -x)
    queries_df["ITEM_TYPE"] = ["query"] * len(queries_df)
    corpus_df["ITEM_TYPE"] = ["passage"] * len(corpus_df)
    corpus_df = corpus_df.set_index("pid")
    queries_df = queries_df.set_index("qid")
    item_df = pd.concat([corpus_df, queries_df])
    return item_df


def get_user_df(data_root="data/ms_marco"):
    user_df = load_dev_top1000_data(data_root)
    user_df.loc[:, "qid"] = user_df["qid"].apply(lambda x: -x)
    user_df = user_df.drop(["pid", "passage"], axis=1)
    user_df = user_df.groupby("qid")["query"].apply(list).to_dict()
    user_df = pd.DataFrame(user_df.items(), columns=["qid", "TITLE"])
    user_df = user_df.set_index("qid")
    return user_df


def load_training_data(data_root="data/ms_marco"):
    ce_scores = pd.read_pickle(
        os.path.join(data_root, "cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl")
    )
    print("Read hard negatives train file")
    hard_negatives_filepath = os.path.join(data_root, "msmarco-hard-negatives.jsonl")
    ce_score_margin = 3.0
    num_neg_per_system = 5
    dataset = {}
    count = 0
    with open(hard_negatives_filepath, "rt") as fIn:
        for line in fIn:
            count += 1
            if count == 101:
                break

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
                dataset[qid] = {"pos_pid": pos_pids, "neg_pid": neg_pids}
    dataset = (dataset, ce_scores)
    return dataset


def get_event_df(data_root="data/ms_marco"):
    dataset, ce_scores = load_training_data(data_root)
    USER_ID = list(range(len(dataset)))
    TEST_START_TIME = [1] * len(USER_ID)

    request_time = [2] * len(USER_ID)
    _hist_items = []
    cand_items = []
    multi_label = []
    for qid, pids in dataset.items():
        _hist_items.append([qid])
        candicates = [pids["pos_pid"][0]] + pids["neg_pid"]
        labels = [1.0] * len([pids["pos_pid"][0]]) + [0.0] * len(pids["neg_pid"])
        cand_items.append(candicates)
        multi_label.append(labels)
    event_df = pd.DataFrame(
        {
            "USER_ID": USER_ID,
            "TEST_START_TIME": TEST_START_TIME,
            "_hist_items": _hist_items,
            "cand_items": cand_items,
            "request_time": request_time,
            "multi_label": multi_label,
        }
    )
    event_df = event_df.set_index(["USER_ID", "TEST_START_TIME"])
    return event_df, dataset


# %%
max_epochs = 0
simulation = True
pretrained_checkpoint = None
train_requests = None
epsilon = 0
role_arn = None
s3_prefix = None
working_model = None
multi_label = False
n_steps = 1
batch_size = 32 * max(1, torch.cuda.device_count())

item_df = get_item_df(data_root="data/ms_marco")
user_df = get_user_df(data_root="data/ms_marco")
zero_shot = create_zero_shot(user_df)
user_df = zero_shot.user_df
gnd_response, dataset = get_event_df(data_root="data/ms_marco")
gnd_events = parse_response(gnd_response)

if working_model is None:
    from ccrec.models.bert_mt import BertMT

    working_model = BertMT(item_df, dataset)
oracle_model = ccrec.models.bbpr.BertBPR(
    item_df,
    dataset,
    batch_size=batch_size,
    model_name="sentence-transformers/msmarco-distilbert-dot-v5",
)

if simulation:
    training_env_kw = {
        "oracle": ccrec.agent.Agent(oracle_model),
        "prefix": "pp-simu-train",
        "soft_label": False,
        "reserve_score": 0.1,
        "test_requests": train_requests,
        "exclude_train": ["ITEM_TYPE"],
    }
else:
    training_env_kw = {
        "oracle": env.I2IConfig(image=True, role_arn=role_arn, s3_prefix=s3_prefix,),
        "prefix": "pp-i2i-train",
        "multi_label": multi_label,
        "test_requests": train_requests,
        "exclude_train": ["ITEM_TYPE"],
    }

all_events = gnd_events  # need to create reranking priors at test time
testing_env_kw = {
    "oracle": rime.Dataset(user_df, item_df, gnd_events),
    "prefix": "pp-simu-test",
    "sample_with_prior": 1e5,  # reranking
}
baseline_models = []  # independent test run w/o competition with other models

iexp = InteractiveExperiment(
    user_df,
    item_df,
    all_events,
    training_env_kw,
    testing_env_kw,
    working_model,
    baseline_models,
    epsilon,
)

iexp.run(n_steps=n_steps, test_every=1, test_before_train=False)
print(pd.DataFrame(iexp.testing_env._reward_by_policy))
