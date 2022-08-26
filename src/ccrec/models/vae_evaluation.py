from .vae_models import VAEPretrainedModel
import sys
import os
import joblib
import math
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
sys.modules['sklearn.externals.joblib'] = joblib
sys.path.append("recurrent-intensity-model-experiments/src")
sys.path.append("InteractiveRecommendation/src")

import rime, irec
from rime.models.zero_shot import ItemKNN

# need a VAE model. For example, 
# model = VAEPretrainedModel.from_pretrained("./VAE_model_prime_beta_0.002/checkpoint-500")

# return the precison evaluated on the prime_pantry data set
def VAE_precision(model, item_df, event_df, user_df, 
                    embed_size = 768,
                    model_checkpoint = "distilbert-base-uncased",
                    batch_size=64):
    D = rime.dataset.Dataset(user_df, item_df, event_df, sample_with_prior=1e5)
    D._k1 = 1
    relevance = rime.Experiment(D)

    df = item_df['TITLE']
    ds = Dataset.from_pandas(df)
    eval_dataloader = DataLoader(ds['TITLE'], batch_size=batch_size)
    item_embed = torch.empty([len(ds['TITLE']), embed_size])
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    for batch_i, batch in enumerate(eval_dataloader):
        tokenized_batch = tokenizer(batch,truncation=True,padding='max_length',max_length=32,return_tensors = "pt")
        with torch.no_grad():
            batch_token_logits = model.forward(input_ids = tokenized_batch["input_ids"].cuda(),
                                 attention_mask = tokenized_batch["attention_mask"].cuda())
        item_embed[batch_i*batch_size:batch_i*batch_size+batch_token_logits.shape[0],:] = batch_token_logits.clone()
    
    agent = ItemKNN(item_df.assign(_hist_len=1, embedding=item_embed.tolist()))
    relevance.run({"new_model": model})

    return relevance.item_rec["new_model"]

# return the average local entropy of the distribution on the candidate items for each query
def VAE_rerank_entropy(model, item_df, event_df,  
                    num_ent_sample = 20,
                    embed_size = 768,
                    model_checkpoint = "distilbert-base-uncased"):
    
    num_item = len(item_df)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    entropy = 0

    for i in range(num_item):
        source = event_df['ITEM_ID'].iloc[i]
        candidates = event_df["ITEM_ID"][num_item + 4*i : num_item + 4*(i+1)].tolist()
    
        tokenized_source = tokenizer(item_df["TITLE"][source],truncation=True,padding='max_length',max_length=32,return_tensors = "pt")
        tokenized_candidates = tokenizer(item_df["TITLE"][candidates].tolist(),truncation=True,padding='max_length',max_length=32,return_tensors = "pt")

        with torch.no_grad():
            source_mean, source_std = model.forward(input_ids = tokenized_source["input_ids"].cuda(),attention_mask = tokenized_source["attention_mask"].cuda())
            cand_mean, cand_std = model.forward(input_ids = tokenized_candidates["input_ids"].cuda(),attention_mask = tokenized_candidates["attention_mask"].cuda())

        dist = [0.0] * len(candidates)

        for _ in range(num_ent_sample):
            source_eps = torch.randn_like(source_mean)
            cand_eps = torch.randn_like(cand_mean)
            source_emb = source_eps * source_std + source_mean
            cand_emb = cand_eps * cand_std + cand_mean

            #computing the true embeddings based on the mu and std
            source_emb = model.standard_layer_norm(model.activation(model.vocab_transform(source_emb)))
            cand_emb = model.standard_layer_norm(model.activation(model.vocab_transform(cand_emb)))

            score = cand_emb @ source_emb[0]
            j = int(torch.argmax(score))  
            dist[j] += 1.0
        
        for j in dist:
            j_norm = j / num_ent_sample
            if j_norm > 0:
                entropy = entropy - j_norm * math.log(j_norm)
        
    return entropy / num_item

# return the average number of unique items of full retrieval for each query
def VAE_full_retrieval_unique_items(model, item_df,
                    num_ent_sample = 20,
                    embed_size = 768,
                    model_checkpoint = "distilbert-base-uncased",
                    batch_size = 64,
                    num_of_recommendation = 10):

    num_item = len(item_df)
    num_of_batch = num_item//batch_size + 1

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    df = item_df['TITLE']
    ds = Dataset.from_pandas(df)
    eval_dataloader = DataLoader(ds['TITLE'], batch_size=batch_size)
    
    item_embed_mean = torch.empty([num_item, embed_size])
    item_embed_std = torch.empty([num_item, embed_size])

    for batch_i, batch in enumerate(eval_dataloader):
        tokenized_batch = tokenizer(batch,truncation=True,padding='max_length',max_length=32,return_tensors = "pt")
        with torch.no_grad():
            batch_mean, batch_std = model.forward(input_ids = tokenized_batch["input_ids"].cuda(),
                                 attention_mask = tokenized_batch["attention_mask"].cuda())
        item_embed_mean[batch_i*batch_size:batch_i*batch_size+batch_mean.shape[0],:] = batch_mean.clone()
        item_embed_std[batch_i*batch_size:batch_i*batch_size+batch_std.shape[0],:] = batch_std.clone()
    
    diff_match_set = [set() for _ in range(num_item)]

    for _ in range(num_of_recommendation):

        item_embed = torch.empty([len(ds['TITLE']), embed_size])
        for i  in range(num_of_batch):
            
            if (i+1) * batch_size <= num_item:
                batch_length = batch_size
            else:
                batch_length = num_item - i*batch_size

            batch_mean = item_embed_mean[i*batch_size:i*batch_size+batch_length,:].clone().cuda()
            batch_std = item_embed_std[i*batch_size:i*batch_size+batch_length,:].clone().cuda()
            eps = torch.randn_like(batch_mean) 
            batch_embed_in = batch_mean + eps * batch_std

            with torch.no_grad():
                batch_embed_out = model.standard_layer_norm(model.activation(model.vocab_transform(batch_embed_in)))
            item_embed[i*batch_size:i*batch_size+batch_length,:] = batch_embed_out
        
        similarity = item_embed @ torch.transpose(item_embed, 0, 1)
        best_match = (torch.topk(similarity, 2).indices)[:,1]

        for i in range(num_item):
            diff_match_set[i].add(int(best_match[i]))
    
    return sum([len(diff_match_set[i]) for i in range(num_item)])/num_item

    


