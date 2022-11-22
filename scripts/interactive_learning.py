#!/usr/bin/env python3
import pandas as pd, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import pytest, rime, ccrec, ccrec.models, ccrec.models.bbpr
from ccrec import InteractiveExperiment, env
from ccrec.env import create_zero_shot, Env, parse_response
from ccrec.env.i2i_env import get_notebook_name
from ccrec.models.bert_mt import BertMT
from ccrec.env import create_reranking_dataset

from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pathlib

import argparse
import pandas as pd
import os

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size_train", default=12, type=int)
parser.add_argument("--batch_size_eval", default=512, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--beta", default=2e-3, type=float)
parser.add_argument("--alpha", default=0., type=float)
parser.add_argument("--freeze_bert", default=0, type=int)
parser.add_argument("--input_type", default="text", type=str)
parser.add_argument("--simulation", default="simulation", type=str)
parser.add_argument("--n_steps", default=1, type=int)
parser.add_argument("--task", default="scifact", type=str)
parser.add_argument("--prior_data_dir", default=None, type=str)
parser.add_argument("--working_model_dir", default=None, type=str)
parser.add_argument("--oracle_model_dir", default=None, type=str)
parser.add_argument("--use_ground_truth_oracle", default=True, type=bool)
parser.add_argument("--summarize", default=False, type=bool)
parser.add_argument("--exp_idx", default=0, type=int)



args = parser.parse_args()

# %%
# load MS_MARCO data
def load_data(task):
    data_name = task
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(data_name)
    out_dir = os.path.join(pathlib.Path("./data/scifact/").parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    if data_name == "msmarco":
        data_split = "dev"
    else:
        data_split = "test"
    corpus_, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=data_split)
    corpus = dict()
    for pid, passage in corpus_.items():
        corpus[pid] = passage["text"]
    return corpus, queries, qrels

def load_prior(user_df, item_df, ranking_profile):
    USER_ID = user_df.index.to_list()
    request_time = [2] * len(USER_ID)
    _hist_items = [USER_ID[idx] for idx in range(len(USER_ID))]
    cand_items = []
    multi_label = []
    num = 2
    for qid, cands in ranking_profile.items():
        if not qid.startswith("q"):
            qid = "q_{}".format(qid)
        pids = list(cands.keys())
        if not pids[0].startswith("p"):
            pids = ["p_{}".format(pid) for pid in pids]
        cand_item_temp = pids[0:num]
        multi_label_temp = [1.] * len(cand_item_temp)
        cand_items.append(cand_item_temp)
        multi_label.append(multi_label_temp)
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
    V = create_reranking_dataset(user_df, item_df, expl_response, reranking_prior=100.)
    return V

def get_item_df(task="msmarco"):
    corpus, queries, qrels = load_data(task)
    corpus_df = pd.DataFrame.from_dict({"pid": corpus.keys(), "TITLE": corpus.values()})
    queries_df = pd.DataFrame.from_dict({"qid": queries.keys(), "TITLE": queries.values()})
    queries_df.loc[:, "qid"] = queries_df["qid"].apply(lambda x: "q_{}".format(x))
    corpus_df.loc[:, "pid"] = corpus_df["pid"].apply(lambda x: "p_{}".format(x))
    queries_df.rename(columns = {'qid':'ITEM_ID'}, inplace = True)
    corpus_df.rename(columns = {'pid':'ITEM_ID'}, inplace = True)
    queries_df["ITEM_TYPE"] = ["query"] * len(queries_df)
    corpus_df["ITEM_TYPE"] = ["passage"] * len(corpus_df)
    item_df = pd.concat([queries_df, corpus_df])
    item_df = item_df.assign(landingImage=None)
    return item_df, qrels

def create_information_retrieval(item_df):
    if isinstance(item_df, str):
        item_df = pd.read_csv(item_df)
    if "ITEM_ID" in item_df:
        item_df = item_df.set_index("ITEM_ID")

    zero_shot = create_zero_shot(
        item_df,
        create_user_filter=lambda x: x["ITEM_TYPE"] == "query",
        exclude_train=["ITEM_TYPE"],
    )
    return zero_shot


# %%
def main(args):
    _batch_size_train = args.batch_size_train
    _batch_size_eval = args.batch_size_eval
    _max_seq_length = args.max_seq_length
    _max_epochs = args.epochs
    _simulation = args.simulation
    _working_model_dir = args.working_model_dir
    _oracle_model_dir = args.oracle_model_dir
    _n_steps = args.n_steps
    _task = args.task
    _prior_data_dir = args.prior_data_dir
    _beta = args.beta
    _alpha = args.alpha
    _use_ground_truth_oracle = args.use_ground_truth_oracle
    _lr = args.lr
    _weight_decay = args.weight_decay
    _summarize = args.summarize
    exp_idx = args.exp_idx

    epsilon = 0
    role_arn='arn:aws:iam::000403867413:role/TSinterns',
    s3_prefix=f"s3://yifeim-interns-labeling/{get_notebook_name()}",
    multi_label = False
    exclude_train=["ITEM_TYPE"]
    
    item_df, qrels = get_item_df(_task)
    zero_shot = create_information_retrieval(item_df)
    user_df = zero_shot.user_df
    item_df = zero_shot.item_df
    event_df = zero_shot.event_df

    if _prior_data_dir is not None:
        prior_data = torch.load(_prior_data_dir)
        prior_score = load_prior(user_df, item_df, prior_data)
    else:
        prior_score = None

    if _working_model_dir is not None:
        checkpoint_working_model = _working_model_dir
        # "train_save/msmarco_VAE_model_0.01/checkpoint-98000/pytorch_model.bin"
    else:
        checkpoint_working_model = None
    train_kw = {
        "lr": _lr,
        "weight_decay": _weight_decay,
        "model_name": "distilbert-base-uncased",
        "max_length": _max_seq_length,
        "freeze_bert": 0,
        "input_step": "text",
        "pretrained_checkpoint": checkpoint_working_model,
        "do_validation": False,
    }
    working_model = BertMT(item_df,
            alpha=_alpha,
            beta=_beta,
            max_epochs=_max_epochs,
            batch_size=_batch_size_train * max(1, torch.cuda.device_count()),
            model_cls_name="VAEPretrainedModel",
            **train_kw,
        )

    if _oracle_model_dir is not None:
        checkpoint_oracle_model = _oracle_model_dir
        # 'lightning_logs/bbpr_model_ft/checkpoints/epoch=8-step=14040.ckpt'
    else:
        checkpoint_oracle_model = None
    oracle_kw = {
        "lr": _lr,
        "weight_decay": _weight_decay,
        "model_name": "distilbert-base-uncased",
        "max_length": _max_seq_length,
        "freeze_bert": 0,
        "input_step": "text",
        "pretrained_checkpoint": checkpoint_oracle_model,
    }
    oracle_model = ccrec.models.bbpr.BertBPR(
            item_df,
            max_epochs=_max_epochs,
            batch_size=_batch_size_train * max(1, torch.cuda.device_count()),
            sample_with_prior=True,
            sample_with_posterior=0,
            replacement=False,
            n_negatives=5,
            valid_n_negatives=5,
            training_prior_fcn=lambda x: (x + 1 / x.shape[1]).clip(0, None).log(),
            **oracle_kw,
        )
    if _use_ground_truth_oracle:
        oracle_model.oracle_dir = qrels

    if _summarize:
        model_name = "google/pegasus-large"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
        summarizer = [tokenizer, model]
    else:
        summarizer = None

    if _simulation == "simulation":
        print('Oracle model as SOTA model')
        training_env_kw = {
            "oracle": ccrec.agent.Agent(oracle_model),
            "prefix": "pp-simu-train",
            "soft_label": "greedy",
            "reserve_score": 0.1,
            "test_requests": prior_score,
            "exclude_train": exclude_train,
            "summarizer": summarizer,
        }
    elif _simulation == "human":
        print('Experiment with human feedback')
        training_env_kw = {
                "oracle": env.I2IConfig(
                    image=True,
                    role_arn=role_arn,
                    s3_prefix=s3_prefix,
                ),
                "prefix": "pp-i2i-train",
                "multi_label": multi_label,
                "test_requests": prior_score,
                "exclude_train": exclude_train,
                "summarizer": summarizer,
            }

    testing_env_kw = {
            "oracle": "dummy",
            "prefix": "pp-simu-test",
            "exclude_train": exclude_train,
        }
    baseline_models = []  # independent test run w/o competition with other models

    iexp = InteractiveExperiment(
        user_df,
        item_df,
        event_df,
        training_env_kw,
        testing_env_kw,
        working_model,
        baseline_models,
        epsilon,
    )

    iexp.run(n_steps=_n_steps, test_every=None, test_before_train=False)
    print(iexp.training_env.event_df)
    torch.save(iexp.training_env.event_df, "lightning_logs/iexp_data_{}.pt".format(exp_idx))

if __name__ == "__main__":
    main(args)

# from src.ccrec.env import I2IImageEnv
# from attrdict import AttrDict
# from PIL import Image

# item_df_ = item_df.set_index("ITEM_ID")

# Image.open(I2IImageEnv.image_format(
#     self=AttrDict(item_df=item_df_,
#                 explainer=working_model.to_explainer()),
#     x={'_hist_items': ["q_1"], 'cand_items': ['p_31715818', 'p_14717500', 'p_14717500', 'p_14717500']},
# )).show()