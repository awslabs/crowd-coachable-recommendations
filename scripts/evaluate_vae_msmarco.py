#!/usr/bin/env python3
import pandas as pd
from src.ccrec.models.vae_models import VAEPretrainedModel
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import math
import matplotlib.pyplot as plt
import os

# to measure the precision of vae,
# use ms_marco_eval.py for MRR score
# ./ms_marco_eval.py

# dev: development set e.g. top-1000
# queries: dictionary [qid, query]
# corpus: dictionary [pid, passage]
def VAE_rerank_entropy(
    model,
    dev,
    queries,
    corpus,
    num_ent_sample=20,
    embed_size=768,
    batch_size=64,
    model_checkpoint="distilbert-base-uncased",
):
    dev_indices = list(dev.keys())
    num_item = len(dev)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    entropy = 0

    for i in range(num_item):
        if (i % 100) == 0:
            print("processing item: {} | {}".format(i, num_item))
        source_idx = dev_indices[i]
        candidates_ids = dev[source_idx]
        source_sentence = [queries[source_idx]]
        candidates_sentences = [corpus[idx] for idx in candidates_ids]

        # TODO: implement batch version when source_sentences are large
        num_of_candicates = len(candidates_ids)
        num_candicates_batches = math.ceil(num_of_candicates / batch_size)
        item_embed_mean = torch.empty([num_of_candicates, embed_size]).cuda()
        item_embed_std = torch.empty([num_of_candicates, embed_size]).cuda()
        with torch.no_grad():
            tokenized_source = tokenizer(
                source_sentence,
                truncation=True,
                padding="max_length",
                max_length=350,
                return_tensors="pt",
            )
            source_mean, source_std = model.forward(
                input_ids=tokenized_source["input_ids"].cuda(),
                attention_mask=tokenized_source["attention_mask"].cuda(),
                return_mean_std=True,
            )
            for step in range(num_candicates_batches):
                batch = candidates_sentences[
                    step * batch_size : (step + 1) * batch_size
                ]
                tokenized_candidates = tokenizer(
                    batch,
                    truncation=True,
                    padding="max_length",
                    max_length=350,
                    return_tensors="pt",
                )
                cand_mean, cand_std = model.forward(
                    input_ids=tokenized_candidates["input_ids"].cuda(),
                    attention_mask=tokenized_candidates["attention_mask"].cuda(),
                    return_mean_std=True,
                )
                item_embed_mean[
                    step * batch_size : step * batch_size + cand_mean.shape[0], :
                ] = cand_mean
                item_embed_std[
                    step * batch_size : step * batch_size + cand_std.shape[0], :
                ] = cand_std
        dist = [0.0] * len(candidates_ids)
        for _ in range(num_ent_sample):
            source_eps = torch.randn_like(source_mean)
            cand_eps = torch.randn_like(item_embed_mean)
            source_emb = source_eps * source_std + source_mean
            cand_emb = cand_eps * item_embed_std + item_embed_mean

            # computing the true embeddings based on the mu and std
            with torch.no_grad():
                source_emb = model.standard_layer_norm(
                    model.activation(model.vocab_transform(source_emb))
                )
                cand_emb = model.standard_layer_norm(
                    model.activation(model.vocab_transform(cand_emb))
                )

            score = cand_emb @ source_emb.permute(1, 0)
            j = int(torch.argmax(score))
            dist[j] += 1.0

        for j in dist:
            j_norm = j / num_ent_sample
            if j_norm > 0:
                entropy = entropy - j_norm * math.log(j_norm)
    return entropy / num_item


# data_dev is dataframe with 4 columns
# [qid, pid, query, passage]
def VAE_rerank_unique_items(
    model,
    data_dev,
    queries,
    corpus,
    num_ent_sample=20,
    embedding_dim=768,
    batch_size=64,
    model_checkpoint="distilbert-base-uncased",
):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_dev_dict = data_dev.groupby("qid")["pid"].apply(list).to_dict()
    effective_corpus_indices = list(data_dev["pid"])
    effective_corpus_indices = set(effective_corpus_indices)
    effective_corpus = {}
    for pid, passage in corpus.items():
        if pid in effective_corpus_indices:
            effective_corpus[pid] = passage
    effective_corpus_indices = list(effective_corpus.keys())
    effective_corpus_list = list(effective_corpus.values())

    queries_ids = list(data_dev_dict.keys())
    num_queries, num_passages = len(queries_ids), len(effective_corpus_indices)
    num_passage_batches = math.ceil(num_passages / batch_size)
    num_query_batches = math.ceil(num_queries / batch_size)
    queries_embeddings_mean = torch.zeros(num_queries, embedding_dim)
    queries_embeddings_std = torch.zeros(num_queries, embedding_dim)
    passage_embeddings_mean = torch.zeros(num_passages, embedding_dim)
    passage_embeddings_std = torch.zeros(num_passages, embedding_dim)
    with torch.no_grad():
        for step_q in range(num_query_batches):
            if (step_q % 100) == 0:
                print("processing batch: {} | {}".format(step_q, num_query_batches))
            indices = queries_ids[step_q * batch_size : (step_q + 1) * batch_size]
            queries_batch = [queries[qid] for qid in indices]
            tokens = tokenizer(
                queries_batch,
                padding="max_length",
                max_length=350,
                truncation=True,
                return_tensors="pt",
            )
            embedding_mean, embedding_std = model(
                input_ids=tokens["input_ids"].cuda(),
                attention_mask=tokens["attention_mask"].cuda(),
                return_mean_std=True,
            )
            queries_embeddings_mean[
                step_q * batch_size : (step_q * batch_size + len(indices)), :
            ] = embedding_mean.cpu()
            queries_embeddings_std[
                step_q * batch_size : (step_q * batch_size + len(indices)), :
            ] = embedding_std.cpu()

        for step_p in range(num_passage_batches):
            if (step_p % 100) == 0:
                print("processing batch: {} | {}".format(step_p, num_passage_batches))
            passage_batch = effective_corpus_list[
                step_p * batch_size : (step_p + 1) * batch_size
            ]
            tokens = tokenizer(
                passage_batch,
                padding="max_length",
                max_length=350,
                truncation=True,
                return_tensors="pt",
            )
            embedding_mean, embedding_std = model(
                input_ids=tokens["input_ids"].cuda(),
                attention_mask=tokens["attention_mask"].cuda(),
                return_mean_std=True,
            )
            passage_embeddings_mean[
                step_p * batch_size : (step_p * batch_size + len(passage_batch)), :
            ] = embedding_mean.cpu()
            passage_embeddings_std[
                step_p * batch_size : (step_p * batch_size + len(passage_batch)), :
            ] = embedding_std.cpu()
    map_pid_localIndex = dict(zip(effective_corpus_indices, list(range(num_passages))))
    sol = 0
    for i in range(num_queries):
        if (i % 100) == 0:
            print("processing item: {} | {}".format(i, num_queries))
        qid = queries_ids[i]
        pids = data_dev_dict[qid]
        if len(pids) == 1:
            continue
        pids_effective = [map_pid_localIndex[pid] for pid in pids]
        source_mean = queries_embeddings_mean[i].unsqueeze(0).cuda()
        source_std = queries_embeddings_std[i].unsqueeze(0).cuda()
        item_embed_mean = passage_embeddings_mean[pids_effective].cuda()
        item_embed_std = passage_embeddings_std[pids_effective].cuda()

        diff_match_set = set()
        for _ in range(num_ent_sample):
            source_eps = torch.randn_like(source_mean)
            cand_eps = torch.randn_like(item_embed_mean)
            source_emb = source_eps * source_std + source_mean
            cand_emb = cand_eps * item_embed_std + item_embed_mean
            with torch.no_grad():
                source_emb = model.standard_layer_norm(
                    model.activation(model.vocab_transform(source_emb))
                )
                cand_emb = model.standard_layer_norm(
                    model.activation(model.vocab_transform(cand_emb))
                )

            similarity = cand_emb @ source_emb.permute(1, 0)
            similarity = similarity.squeeze()
            best_match = (torch.topk(similarity, 1).indices)[0].item()
            diff_match_set.add(int(best_match))
        num_of_unique_items = len(diff_match_set)
        sol += num_of_unique_items
    return sol / num_queries


# item_df contains all passages
def VAE_full_retrieval_unique_items(
    model,
    item_df,
    num_ent_sample=20,
    embed_size=768,
    model_checkpoint="distilbert-base-uncased",
    batch_size=64,
    num_of_recommendation=10,
):
    num_item = len(item_df)
    num_of_batch = num_item // batch_size + 1

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    ds = Dataset.from_pandas(item_df)
    eval_dataloader = DataLoader(ds["TITLE"], batch_size=batch_size)

    item_embed_mean = torch.empty([num_item, embed_size]).cuda()
    item_embed_std = torch.empty([num_item, embed_size]).cuda()

    for batch_i, batch in enumerate(eval_dataloader):
        if (batch_i % 100) == 0:
            print("processing batch: {} | {}".format(batch_i, len(eval_dataloader)))
        tokenized_batch = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=350,
            return_tensors="pt",
        )
        with torch.no_grad():
            batch_mean, batch_std = model.forward(
                input_ids=tokenized_batch["input_ids"].cuda(),
                attention_mask=tokenized_batch["attention_mask"].cuda(),
                return_mean_std=True,
            )
        item_embed_mean[
            batch_i * batch_size : batch_i * batch_size + batch_mean.shape[0], :
        ] = batch_mean
        item_embed_std[
            batch_i * batch_size : batch_i * batch_size + batch_std.shape[0], :
        ] = batch_std

    diff_match_set = [set() for _ in range(num_item)]

    for _ in range(num_of_recommendation):

        item_embed = torch.empty([len(ds["TITLE"]), embed_size]).cuda()
        for i in range(num_of_batch):

            if (i + 1) * batch_size <= num_item:
                batch_length = batch_size
            else:
                batch_length = num_item - i * batch_size

            batch_mean = item_embed_mean[
                i * batch_size : i * batch_size + batch_length, :
            ]
            batch_std = item_embed_std[
                i * batch_size : i * batch_size + batch_length, :
            ]
            eps = torch.randn_like(batch_mean)
            batch_embed_in = batch_mean + eps * batch_std

            with torch.no_grad():
                batch_embed_out = model.standard_layer_norm(
                    model.activation(model.vocab_transform(batch_embed_in))
                )
            item_embed[
                i * batch_size : i * batch_size + batch_length, :
            ] = batch_embed_out
        item_embed = item_embed.cpu()
        similarity = item_embed @ torch.transpose(item_embed, 0, 1)
        best_match = (torch.topk(similarity, 2).indices)[:, 1]

        for i in range(num_item):
            diff_match_set[i].add(int(best_match[i]))
    return sum([len(diff_match_set[i]) for i in range(num_item)]) / num_item


def plot_losses(beta, checkpoint=98360):
    data_dir = "msmarco_VAE_model_{}/checkpoint-{}/trainer_state.json".format(
        beta, checkpoint
    )
    losses = []
    data = pd.read_json(data_dir)
    for data_ii in data["log_history"]:
        losses.append(data_ii["loss"])
    plt.plot(losses)


def measure_vae_embedding_loss(data, corpus, queries, model, batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    query_sentences, passage_sentences = [], []
    for qid, pids in data.items():
        query_sentences.append(queries[qid])
        passage_sentences.append(corpus[pids[0]])

    num_data = len(data)
    num_batches = math.ceil(num_data / batch_size)
    loss_quereis_total = 0.0
    loss_passages_total = 0.0
    with torch.no_grad():
        for step in range(num_batches):
            query_batch = query_sentences[step * batch_size : (step + 1) * batch_size]
            passage_batch = passage_sentences[
                step * batch_size : (step + 1) * batch_size
            ]
            tokens_query = tokenizer(
                query_batch,
                padding="max_length",
                max_length=350,
                truncation=True,
                return_tensors="pt",
            )
            tokens_passage = tokenizer(
                passage_batch,
                padding="max_length",
                max_length=350,
                truncation=True,
                return_tensors="pt",
            )
            outputs_query = model(
                input_ids=tokens_query["input_ids"].cuda(),
                attention_mask=tokens_query["attention_mask"].cuda(),
            )
            outputs_passage = model(
                input_ids=tokens_passage["input_ids"].cuda(),
                attention_mask=tokens_passage["attention_mask"].cuda(),
            )
            loss_quereis_total += outputs_query["loss"].mean().item()
            loss_passages_total += outputs_passage["loss"].mean().item()

    loss_quereis_total /= num_batches
    loss_passages_total /= num_batches
    print(
        "Mean embedding loss query: {:.4f}, passage: {:.4f}".format(
            loss_quereis_total, loss_passages_total
        )
    )
