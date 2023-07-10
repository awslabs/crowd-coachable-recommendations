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


## convert labels

df_perm = pd.read_csv(f"{RESULTS_DIR}/data_iteration_{STEP}/human_response.csv")
df_orig = pd.read_csv(f"{RESULTS_DIR}/data_iteration_{STEP}/request_orig.csv")


# normalize response
df_perm["Input.query"] = df_perm["Input.query"].apply(
    lambda x: [y for y in df_orig["query"].values if x.strip() == y.strip()][0]
)


def convert_labels(df_perm, df_orig):
    request_label_map = {
        query: {p.strip(): i for i, p in passages.iteritems()}
        for query, passages in df_orig.set_index("query").iterrows()
    }

    df_converted = df_perm.assign(
        selected_passage=lambda df: df.apply(
            lambda x: None
            if x["Answer.quetion-answering.label"].endswith("None of the above")
            else x["Input.passage-{}".format(x["Answer.quetion-answering.label"])],
            axis=1,
        )
    )

    converted_label = [
        request_label_map[query][passage.strip()] if passage is not None else "zzz"
        for query, passage in df_converted.set_index("Input.query")[
            "selected_passage"
        ].iteritems()
    ]

    df_converted["converted_label"] = converted_label

    return df_converted[
        [
            "Input.query",
            "WorkerId",
            "converted_label",
            "WorkTimeInSeconds",
        ]
    ].copy()


df = convert_labels(df_perm, df_orig)


def entropy(x):
    _, q = np.unique(x, return_counts=True)
    p = q / q.sum()
    return np.exp(-np.log(p).dot(p))


print(
    df.groupby("Input.query")["converted_label"].apply(entropy).mean(),
    np.mean([entropy(x) for x in np.random.choice(4, (1000, 3))]),
    np.mean([entropy(x) for x in np.random.choice(3, (1000, 3))]),
)


print(df.groupby("converted_label").size())

## Dawid Skene

all_tasks = sorted(df["Input.query"].unique())
all_workers = sorted(df["WorkerId"].unique())
all_labels = sorted(df["converted_label"].unique())

data = np.zeros((len(all_tasks), len(all_workers), len(all_labels)))

print("data.shape", data.shape)


for _, row in df.iterrows():
    data[
        all_tasks.index(row["Input.query"]),
        all_workers.index(row["WorkerId"]),
        all_labels.index(row["converted_label"]),
    ] = 1

print(data.sum(axis=1))

z_majority = np.hstack(
    [
        np.argmax(
            (
                data.sum(axis=1)
                + np.random.rand(
                    data.shape[0],
                    data.shape[-1],
                )
                * 0.1
            ),
            axis=1,
        )
        for _ in range(100)
    ]
)
print(z_majority.shape)


I, J, K = data.shape

# create data triplets
ii = list()  # item IDs
jj = list()  # annotator IDs
y = list()  # response

# create data triplets
for i in range(I):
    for j in range(J):
        dat = data[i, j, :]
        if dat.sum() > 0:
            k = np.where(dat == 1)[0][0]
            ii.append(i)
            jj.append(j)
            y.append(k)


## define model


class VqNet(nn.Module):
    """task, worker, response"""

    def __init__(self, I, J, K):
        super().__init__()
        self.snr_logit = torch.nn.Parameter(torch.empty(J).uniform_(-0.07, 0.07))
        self.I = I
        self.set_K(K)

    def set_K(self, K):
        print(f"setting K={K}")
        self.register_buffer("signal_const", torch.eye(K).reshape((1, K, K)))
        self.register_buffer(
            "noise_const",
            torch.ones((K, K)).reshape((1, K, K))
            / K,  # new changes after msmarco step-2-em
        )

    def forward(self, ii, jj, y):
        theta = (
            self.snr_logit.sigmoid().reshape((-1, 1, 1)) * self.signal_const
            + (-self.snr_logit).sigmoid().reshape((-1, 1, 1)) * self.noise_const
        ) / 2
        log_theta = (theta / theta.sum(-1, keepdims=True)).log()  # j, k, k

        complete_log_lik = (
            F.one_hot(ii, self.I).T.float() @ log_theta.swapaxes(-2, -1)[jj, y]
        )
        qz = complete_log_lik.softmax(-1).detach()  # stick to EM; negligible effects
        Vq = (qz * complete_log_lik).sum(-1) - (qz * qz.log()).sum(-1)

        return qz, Vq


class LitModel(pl.LightningModule):
    def __init__(self, vq_net):
        super().__init__()
        self.vq_net = vq_net
        self._loss_hist = []

    def setup(self, stage):
        if stage == "fit":
            print(self.logger.log_dir)

    def training_step(self, batch, batch_idx):
        ii, jj, y = batch[:, 0], batch[:, 1], batch[:, 2]
        _, Vq = self.vq_net(ii, jj, y)
        self.log("loss", -Vq.mean())
        self._loss_hist.append(-Vq.mean().item())
        return -Vq.mean()

    def configure_optimizers(self):
        weight_decay = float(os.environ.get("CCREC_DAWID_SKENE_WEIGHT_DECAY", 0.0005))
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=weight_decay)


def train_vq(I, J, K, ii, jj, y):
    vq_net = VqNet(I, J, K - 1)

    data_tuples = np.asarray([ii, jj, y]).T
    unbiased_data = data_tuples[data_tuples[:, -1] < K - 1]

    train_loader = DataLoader(unbiased_data, batch_size=unbiased_data.shape[0])
    trainer = pl.Trainer(
        max_epochs=500,
        gpus=int(torch.cuda.is_available()),
    )
    model = LitModel(vq_net)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )

    plt.figure(figsize=(3, 2))
    plt.plot(model._loss_hist)
    plt.xlabel("step")
    plt.ylabel("-Vq per task")
    plt.grid()
    plt.show()

    with torch.no_grad():
        vq_net.set_K(K)
        snr = vq_net.snr_logit.sigmoid().detach().cpu().numpy()
        qz, _ = vq_net(torch.as_tensor(ii), torch.as_tensor(jj), torch.as_tensor(y))
        qz = qz.detach().cpu().numpy()
    z_hat = qz.argmax(-1)

    return vq_net, model, snr, qz, z_hat


vq_net, model, snr, qz, z_hat = train_vq(I, J, K, ii, jj, y)


confMat = confusion_matrix(z_majority, np.hstack([z_hat for _ in range(100)])) // 100
print("Dawid-Skene estimate of true category:\n", confMat)
print("majority class distn:", confMat.sum(1))
print("Dawid Skene class distn:", confMat.sum(0))

plt.scatter(
    snr,
    df.groupby("WorkerId")["converted_label"].apply(
        lambda x: np.mean([y == "passage-4" for y in x])
    )[all_workers],
    df.groupby("WorkerId").size()[all_workers],
)
plt.grid()
plt.xlabel("Dawid Skene Inferred SNR")
plt.ylabel("Random click rate")
plt.show()


worker_random_score = (
    df.groupby("WorkerId")
    .agg(
        random=("converted_label", lambda x: np.mean([y == "passage-4" for y in x])),
        na_rate=("converted_label", lambda x: np.mean([y == "zzz" for y in x])),
        size=("converted_label", "size"),
        median_time=("WorkTimeInSeconds", "median"),
    )
    .join(pd.Series(snr, index=all_workers).to_frame("snr"))
    .sort_values("snr")
)

print(worker_random_score.sort_values("snr").head(15))
print(worker_random_score.sort_values("snr").tail(15))

reject_workers = worker_random_score[
    worker_random_score["snr"] < 0.15
]  # change after msmarco step-3
print("reject_workers", reject_workers)


## output

id_track = torch.load(f"{RESULTS_DIR}/data_iteration_{STEP}/id_track.pt")

request_cand_ids = [[id_track[v] for v in r] for r in df_orig.values]

response_label_ids = pd.Series(z_hat, index=all_tasks).loc[df_orig["query"]]


train_dataset = {}
for r, v in zip(request_cand_ids, response_label_ids.values):
    if v < K - 1:
        r = [v[2:] if v.startswith("p_") or v.startswith("q_") else v for v in r]
        qid, pids = r[0], r[1:].copy()
        pos_pid = pids[v]
        del pids[v]
        train_dataset[qid] = {"pos_pid": [pos_pid], "neg_pid": pids}


if isinstance(STEP, int) and STEP > 0:
    train_pre = torch.load(
        f"{RESULTS_DIR}/data_iteration_{STEP-1}/train_data_human_response.pt"
    )
else:
    train_pre = {}

for qid, item in train_dataset.items():
    train_pre[qid] = item

if not DRYRUN:
    save_path = f"{RESULTS_DIR}/data_iteration_{STEP}/train_data_human_response.pt"
    torch.save(train_pre, save_path)

## winning labels and bonus

df["converted_label"].unique()
df_winning = df.join(
    pd.Series(np.array(all_labels)[z_hat], index=all_tasks).to_frame("winning_label"),
    on="Input.query",
).query("converted_label == winning_label")

send_bonus = (
    df_winning.groupby("WorkerId")
    .size()
    .to_frame("winning")
    .join(df_perm.groupby("WorkerId").size().to_frame("submitted"))
    .join(pd.Series(snr, index=all_workers).to_frame("snr"))
    .join(
        df.groupby("WorkerId")["WorkTimeInSeconds"].quantile([0.1, 0.5, 0.9]).unstack()
    )
    .join(
        df.groupby("WorkerId")["converted_label"]
        .apply(lambda x: np.mean([y == "passage-4" for y in x]))
        .to_frame("random_rate")
    )
    .join(
        df.groupby("WorkerId")["converted_label"]
        .apply(lambda x: np.mean([y == "zzz" for y in x]))
        .to_frame("na_rate")
    )
    .assign(
        win_rate=lambda df: df["winning"] / df["submitted"],
        bonus=lambda df: df["winning"] * 0.04 * (~df.index.isin(reject_workers.index)),
    )
    .join(df_perm.groupby("WorkerId")["AssignmentId"].last().to_frame("AssignmentId"))
    .assign(UniqueRequestToken=lambda df: [uuid1() for _ in range(len(df))])
)

print(send_bonus.sort_values("bonus", ascending=False).head(10))
if not DRYRUN:
    df_winning.to_csv(f"{RESULTS_DIR}/data_iteration_{STEP}/df_winning.csv")
    send_bonus.to_csv(f"{RESULTS_DIR}/data_iteration_{STEP}/send_bonus.csv")
