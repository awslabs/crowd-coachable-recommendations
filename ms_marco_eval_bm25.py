#!/usr/bin/env python3
import torch

import pandas as pd

import sys
from collections import Counter
import os
import math
import time

from bm_25 import BM25


MaxMRRRank = 100

# %%
# generate rankings
def load_corpus():
    data_dir = 'data/ms_marco/collection.tsv'
    df = pd.read_csv(data_dir, sep='\t', header=None, names=['pid', 'passage'])
    data = dict(zip(df['pid'], df['passage']))
    print('Number of passages:', len(data))
    return data

def load_dev_data():
    data_dir = 'data/ms_marco/qrels.dev.tsv'
    df = pd.read_csv(data_dir, sep='\t', header=None, usecols=[0, 2], names=['qid', 'pid'])
    data = df.groupby('qid')['pid'].apply(list).to_dict()
    print('Number of development queries:', len(data))
    return data

def load_dev_top1000_data():
    data_dir = 'data/ms_marco/top1000.dev'
    df = pd.read_csv(data_dir, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])
    print('Number of development queries:', len(df))
    return df

def load_query():
    df_train = pd.read_csv('data/ms_marco/queries.train.tsv', sep='\t', header=None, names=['qid', 'query'])
    df_dev = pd.read_csv('data/ms_marco/queries.dev.tsv', sep='\t', header=None, names=['qid', 'query'])
    df_eval = pd.read_csv('data/ms_marco/queries.eval.tsv', sep='\t', header=None, names=['qid', 'query'])
    df = pd.concat([df_train, df_dev, df_eval], ignore_index=True)
    data = dict(zip(df['qid'], df['query']))
    print('Number of queries:', len(data))
    return data

def ranking(data_dev, corpus, queries):
    save_top = 100
    ranking_profile = {}
    model = BM25()
    corpus_list = list(corpus.values())
    query_list = list(queries.values())
    dataset = corpus_list + query_list
    print('Fitting BM-25 model')
    model.fit(dataset)
    print('Retrieval with BM-25 model')
    queries_ids = list(data_dev.keys())
    corpus_ids = list(corpus.keys())
    corpus_ids = torch.Tensor(corpus_ids).type(torch.int32)
    num_queries = len(queries_ids)
    for step_q in range(num_queries):
        if (step_q % 100) == 0:
            print('processing query: {} | {}'.format(step_q, num_queries))
        qid = queries_ids[step_q]
        query_sentence = queries[qid]
        solution = model.transform(query_sentence, corpus_list)
        solution = torch.Tensor(solution)
        _, indices = solution.sort(descending=True)
        corpus_ids_sorted = corpus_ids[indices]
        ranking_profile[qid] = corpus_ids_sorted[0:save_top]
    return ranking_profile

# data_dev is dataframe with 4 columns
# [qid, pid, query, passage]
def re_ranking(data_dev, corpus, queries):
    save_top = 100
    ranking_profile = {}
    model = BM25()
    data_dev_dict = data_dev.groupby('qid')['pid'].apply(list).to_dict()
    queries_ids = list(data_dev_dict.keys())
    num_queries = len(queries_ids)

    effective_corpus_list = list(data_dev['passage'])
    effective_corpus_list = list(set(effective_corpus_list))
    effective_query_list = list(queries.values())
    effective_query_list = list(set(effective_query_list))
    dataset = effective_corpus_list + effective_query_list
    print('Fitting BM-25 model')
    model.fit(dataset)
    print('Retrieval with BM-25 model')
    for step_q in range(num_queries):
        if (step_q % 100) == 0:
            print('processing query: {} | {}'.format(step_q, num_queries))
        qid = queries_ids[step_q]
        pids = data_dev_dict[qid]
        passages = [corpus[pid] for pid in pids]
        query_sentence = queries[qid]
        solution = model.transform(query_sentence, passages)
        solution = torch.Tensor(solution)
        _, indices = solution.sort(descending=True)
        pids = torch.Tensor(pids).type(torch.int32)
        pids = pids[indices]
        ranking_profile[qid] = pids[0:save_top]
    return ranking_profile

def process_dev_data_for_reranking(dev_data, top1000_dev_data):
    effective_queries = list(top1000_dev_data['qid'])
    effective_queries = list(set(effective_queries))
    dataset = {}
    for qid, pids in dev_data.items():
        if qid in effective_queries:
            dataset[qid] = pids
    return dataset

def convert_dev_data_to_msmarco(data, data_folder, model_name):
    writing_dir = os.path.join(data_folder, 'eval', 'dev_data_{}.tsv'.format(model_name))
    with open(writing_dir, 'w') as w:
        for qid in data:
            pids = data[qid]
            for pid in pids:
                w.write('{}\t{}\n'.format(qid, pid))
    return writing_dir
                
def convert_ranking_to_msmarco(data, data_folder, model_name):
    writing_dir = os.path.join(data_folder, 'eval', 'rankings_{}.tsv'.format(model_name))
    with open(writing_dir, 'w') as w:
        for qid in data:
            passage_indices = data[qid]
            for rank, pid in enumerate(passage_indices):
                rank += 1
                w.write('{}\t{}\t{}\n'.format(qid, pid, rank))
    return writing_dir

# %%
# from files
def load_reference_from_stream(f):
    qids_to_relevant_passageids = {}
    for line in f:
        try:
            line = line.strip().split('\t')
            qid = int(line[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(line[1]))
        except:
            raise IOError('\"%s\" is not valid format' % line)
    return qids_to_relevant_passageids

def load_reference(path_to_reference):
    with open(path_to_reference, 'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids

def load_candidate_from_stream(f):
    qid_to_ranked_candidate_passages = {}
    for line in f:
        try:
            line = line.strip().split('\t')
            qid = int(line[0])
            pid = int(line[1])
            rank = int(line[2])
            if qid in qid_to_ranked_candidate_passages:
                pass
            else:
                tmp = [0] * 1000
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank-1] = pid
        except:
            raise IOError('\"%s\" is not valid format' % line)
    return qid_to_ranked_candidate_passages

def load_candidate(path_to_candidate):
    with open(path_to_candidate, 'r') as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages

def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    message = ''
    allowed = True
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())
    for qid in qids_to_ranked_candidate_passages:
        duplicate_pids = set([item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])
        if len(duplicate_pids - set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0])
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
                    MRR += 1/(i + 1)
                    ranking.pop()
                    ranking.append(i+1)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    MRR = MRR / len(qids_to_relevant_passageids)
    all_scores['MRR @100'] = MRR
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores
    
def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)
    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

if __name__ == '__main__':
    model_name_ = 'bm25'
    data_folder = 'data/ms_marco'
    corpus = load_corpus()
    dev_data = load_dev_data()
    dev_top1000_data = load_dev_top1000_data()
    query_data = load_query()

    time_start = time.time()
    # ranking_profile = ranking(dev_data, corpus, query_data)
    ranking_profile = re_ranking(dev_top1000_data, corpus, query_data)
    dev_data = process_dev_data_for_reranking(dev_data, dev_top1000_data)
    time_end = time.time()
    print('Time used for ranking:', time_end - time_start)

    path_to_reference = convert_dev_data_to_msmarco(dev_data, data_folder, model_name_)
    path_to_candidate = convert_ranking_to_msmarco(ranking_profile, data_folder, model_name_)
    
    all_scores = compute_metrics_from_files(path_to_reference, path_to_candidate)
    print('MRR @100:', all_scores['MRR @100'], 'queries ranked:', all_scores['QueriesRanked'])


# random model: 0.0016056655968131738
# pre-trained model: 0.018295380095013063
# fine-tuned model: 0.3307866193486566
# vae unsupervised: 0.0491716203796524
