import torch
import csv
import os
import pandas as pd
from collections import Counter
import random


# Description
# Step-1: generate request
#         request is in text form, query text -- [passage text]
# Step-2: generate exp data from ranking profiles
#         Exp data: each query has two candidates from targert model adn two candidates from bm25
# Step-3: upload request to MTurk
# Step-4: measure human response accuracy
# Step-5: construct training data from human response
# Step-6: fine-tune zero-shot model
# Step-7: generate ranking profile from fine-tuned model
# REPEAT

# %%
# Step-1: <generate request>
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# input ranking_profile from our model and ranking_profile from bm-25
# this part generates query -> [passage-1, passage-2, passage-3, passage-4]
# where passage-1 and passage-2 are from our vae model,
# passage-3 and passage-4 are from bm-25
# this part makes 2 outputs:
# 1: request.csv --> query: [passage-1, passage-2, passage-3, passage-4]
# 2: id_track --> dict: query -> qid
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# import re
# import csv

# ranking_profile = torch.load("scripts/hotpotqa_results_human_agent/data_iteration_0/ranking_profile.pt")
# ranking_profile_bm25 = torch.load("scripts/hotpotqa_results_human_agent/data_iteration_0/ranking_profile_bm25.pt")

# def filter_string(text):
#     new_text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
#     return new_text

# qids_all = list(qrels.keys())
# qids_all = qids_all[0:500]
# header = ['query', 'passage-1', 'passage-2', 'passage-3', 'passage-4']
# rows = []

# id_track = dict()
# for qid in ranking_profile:
#     if qid not in qids_all:
#         continue

#     ranks = list(ranking_profile[qid].keys())
#     ranks_bm25 = list(ranking_profile_bm25[qid].keys())

#     cands = ranks[0:2]
#     for pid in ranks_bm25:
#         if len(cands) == 4:
#             break
#         if pid not in cands:
#             cands.append(pid)

#     query_text = queries[qid]
#     passages = [filter_string(corpus[pid]["text"]) for pid in cands]

#     row = [query_text, *passages]
#     rows.append(row)
#     id_track[query_text] = "q_{}".format(qid)
#     for pid, passage in zip(cands, passages):
#         id_track[passage] = "p_{}".format(pid)

# save_dir = os.path.join("scripts", "hotpotqa_results_human_agent", "data_iteration_0")
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# task_name = "request.csv"
# filename = os.path.join(save_dir, task_name)
# # writing to csv file
# with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
#     # creating a csv writer object
#     csvwriter = csv.writer(csvfile)
#     # writing the fields
#     csvwriter.writerow(header)
#         # writing the data rows
#     csvwriter.writerows(rows)
# track_name = "id_track.pt"
# id_track_file_name = os.path.join(save_dir, track_name)
# torch.save(id_track, id_track_file_name)

# %%
# Step-2: <generate exp data>
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# generate exp data
# input as ranking_profile from our model and ranking_profile from bm-25
# exp data acts as a mapping from qid --> [pid-1, pid-2, pid-3, pid-4]
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# import csv
# import torch

# ranking_profile = torch.load("scripts/hotpotqa_results_human_agent/data_iteration_0/ranking_profile.pt")
# ranking_profile_bm25 = torch.load("scripts/hotpotqa_results_human_agent/data_iteration_0/ranking_profile_bm25.pt")

# qids_all = list(qrels.keys())
# header = ["USER_ID", "ITEM_ID", 'VALUE', "_group"]
# rows = []
# for qid in ranking_profile_bm25:
#     if qid not in qids_all:
#         continue
#     ranks = list(ranking_profile[qid].keys())
#     ranks_bm25 = list(ranking_profile_bm25[qid].keys())
#     cands = ranks[0:2]
#     for pid in ranks_bm25:
#         if len(cands) == 4:
#             break
#         if pid not in cands:
#             cands.append(pid)

#     user_id = "q_{}".format(qid)
#     item_id = ["p_{}".format(pid) for pid in cands]
#     for item in item_id:
#         row = [user_id, item, 0, 0,]
#         rows.append(row)
#     row = [user_id, user_id, 0, 0]
#     rows.append(row)

# save_dir = os.path.join("scripts", "hotpotqa_results_human_agent", "data_iteration_0")
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# task_name = "exp_data.csv"
# filename = os.path.join(save_dir, task_name)
# # writing to csv file
# with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
#     # creating a csv writer object
#     csvwriter = csv.writer(csvfile)
#     # writing the fields
#     csvwriter.writerow(header)
#         # writing the data rows
#     csvwriter.writerows(rows)

# %%
# step-3: <upload request to MTurk>

# %%
# step-4: <measure human response accuracy>
# inputs: exp data directory from step-2
#         id_track directory from step-1
#         human response directory
# this part will post-process human response and compute accuracy of majority voting

# import torch
# import pandas as pd
# from collections import Counter

# exp_dir = "scripts/hotpotqa_results_human_agent/data_iteration_2/exp_data.csv"
# id_track_dir = "scripts/hotpotqa_results_human_agent/data_iteration_2/id_track.pt"
# response_dir = "scripts/hotpotqa_results_human_agent/data_iteration_2/human_response.csv"

# def convert_experiment_df_to_dict(df_expl_dir):
#     expl_df = pd.read_csv(df_expl_dir)
# #     expl_df = torch.load(df_expl_dir)
#     expl = dict()
#     for row in expl_df.iterrows():
#         row = row[1]
#         qid = row["USER_ID"].split("_")[-1]
#         pid = row["ITEM_ID"].split("_")[-1]
#         value = row['VALUE']
#         group = row["_group"]
#         if qid not in expl:
#             expl[qid] = {"pid": [], "value": [], "group": []}
#         expl[qid]["pid"].append(pid)
#         expl[qid]["value"].append(value)
#         expl[qid]["group"].append(group)
#     return expl

# def process_human_response(request_dir, id_track_dir, response_dir):
#     request_dict = convert_experiment_df_to_dict(request_dir)
#     id_track_dict = torch.load(id_track_dir)
#     response_csv = pd.read_csv(response_dir)
#     response_dict = dict()
#     query_column_name = "Input.query"
#     passage_column_name = ["Input.passage-1", "Input.passage-2", "Input.passage-3", "Input.passage-4"]
#     answer_column_name = "Answer.quetion-answering.label"
#     for row in response_csv.iterrows():
#         row = row[1]
#         query_text = row[query_column_name]
#         if query_text not in id_track_dict:

#             continue
#         qid = id_track_dict[query_text].split("_")[-1]
#         pids = request_dict[qid]["pid"]
#         ans = row[answer_column_name]
#         approval_rate = int(row["Last7DaysApprovalRate"].split("%")[0])
#         work_time = row["WorkTimeInSeconds"]
#         if ans == "5 -- None of the above":
#             ans = -1
#         else:
#             ans = int(ans)
#             ans -= 1
#             ans = pids[ans]
#         if qid not in response_dict:
#             response_dict[qid] = {"pid": [], "appro": [], "work_time": []}
#         response_dict[qid]["pid"].append(ans)
#         response_dict[qid]["appro"].append(approval_rate)
#         response_dict[qid]["work_time"].append(work_time)
#     return response_dict

# def majority_vote_human_response(response):
#     response_vote = dict()
#     num_correct = 0
#     for qid, item in response.items():
#         pids = item["pid"]
#         x = Counter(pids)
#         index = x.most_common(1)[0][0]
#         if index == -1 and len(x) > 1:
#             index = x.most_common(2)[1][0]
#             count = x.most_common(2)[1][1]

#         labels = list(qrels[qid].keys())
#         if index in labels:
#             num_correct += 1
#         response_vote[qid] = index
#     print("number of correct samples (majority vote):", num_correct)
#     return response_vote

# response = process_human_response(exp_dir, id_track_dir, response_dir)

# exp_data = convert_experiment_df_to_dict(exp_dir)
# qid_effect = []
# num_effect_samples = 0
# for qid in response:
#     candidates = exp_data[qid]["pid"]
#     candidates.remove(qid)
#     labels = list(qrels[qid].keys())
#     if any(pid in labels for pid in candidates):
#         num_effect_samples += 1
#         qid_effect.append(qid)
# print("number of queries that contain true answer:", num_effect_samples)

# num_identified_samples = 0
# for qid, item in response.items():
#     pids = item["pid"]
#     labels = list(qrels[qid].keys())
#     if any(pid in labels for pid in pids):
#         num_identified_samples += 1
# print("number of human identified samples:", num_identified_samples)

# response_vote = majority_vote_human_response(response)


# %%
# step-5: <construct training data from human response>
# inputs: exp data direction from step-2
#         response dict from step-4
#         response_vote from step-4
# this part generates training dataset
# training dataset: qid -> [positive pid, negative pid-1, negative pid-2, ...]


# exp_data = convert_experiment_df_to_dict("scripts/nq_results_human_agent/data_iteration_3/exp_data.csv")

# train_dataset = dict()

# for qid, item in response_vote.items():
#     pos_pid = response_vote[qid]
#     if pos_pid == -1:
#         continue

#     request_pids = exp_data[qid]["pid"]

#     request_pids.remove(qid)
#     for pid in response[qid]:
#         if pid in request_pids:
#             request_pids.remove(pid)

#     train_dataset[qid] = {"pos_pid": [pos_pid], "neg_pid": request_pids}

# train_pre = torch.load("scripts/nq_results_human_agent/data_iteration_2/train_data_human_response.pt")

# for qid, item in train_dataset.items():
#     train_pre[qid] = item

# save_dir = os.path.join("scripts", "nq_results_human_agent", "data_iteration_3", "train_data_human_response.pt")
# torch.save(train_pre, save_dir)


# %%
# step-6: <fine-tune zero-shot model>
# call train_bmt_msmarco.py
# --training_dataset_dir : give the directory of training data generated from step-5
# --checkpoint : give initial vae model directory
# --alpha = 1
# --lr 2e-5
# --epochs 10

# %%
# step-7: <generate ranking profile from fine-tuned model>
# call ms_marco_eval.py
# --model_dir : give the directory of trained model from step-6
# --task : choose a task, e.g. msmarco, nq


# %%
# BM-25 for evaluation

# from beir import util, LoggingHandler
# from beir.retrieval import models
# from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.evaluation import EvaluateRetrieval
# from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
# from beir.retrieval.evaluation import EvaluateRetrieval

# import logging
# import pathlib, os

# # load corpus, quereis, and qrels
# def load_dataset(task="msmarco"):
#     url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(task)
#     out_dir = os.path.join(pathlib.Path("./").parent.absolute(), "datasets")
#     data_path = util.download_and_unzip(url, out_dir)

#     #### Provide the data_path where scifact has been downloaded and unzipped
#     corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
#     return corpus, queries, qrels

# evaluation using bm-25
# from pyserini.search.lucene import LuceneSearcher

# searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

# ranking_profile = dict()

# corpus_list = [item["text"] for key, item in corpus.items()]
# query_list = list(queries.values())
# dataset = corpus_list + query_list
# print("Retrieval with BM-25 model")
# queries_ids = list(queries.keys())
# corpus_ids = list(corpus.keys())
# num_queries = len(queries_ids)
# for step_q in range(num_queries):
#     if (step_q % 1000) == 0:
#         print("processing query: {} | {}".format(step_q, num_queries))
#     qid = queries_ids[step_q]
#     query_sentence = queries[qid]
#     hits = searcher.search(query_sentence, k=1001)
#     ranking, score = [], []
#     if len(hits) < 100:
#         for ii in range(0, len(hits)):
#             ranking.append(hits[ii].docid)
#             score.append(hits[ii].score)
#     else:
#         for ii in range(0, 100):
#             ranking.append(hits[ii].docid)
#             score.append(hits[ii].score)
#     ranking_profile[qid] = dict(zip(ranking, score))

# evaluator = EvaluateRetrieval(None)
# mrr = evaluator.evaluate_custom(qrels, ranking_profile, [10, 100], metric="mrr")
# for name, value in mrr.items():
#     print("{}".format(name), ":", value)
# ndcg, _map, recall, precision = evaluator.evaluate(qrels, ranking_profile, [10, 100])
# results = [ndcg, _map, recall, precision]
# for res in results:
#     print("................................")
#     for name, value in res.items():
#         print("{}".format(name), ":", value)
