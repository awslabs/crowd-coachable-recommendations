import dataclasses, torch, time, os, operator, typing, functools, tqdm
import pandas as pd, numpy as np, scipy.sparse as sps
import torch.nn.functional as F
from rime.util import _assign_topk, auto_device, auto_cast_lazy_score, empty_cache_on_exit, RandScore
from irec.agent.base import Agent
from irec.util import merge_unique
from transformers import AutoTokenizer
from irec.models.vae_models import EmbeddingModel


@dataclasses.dataclass
class LatentNoiseAgentBase(Agent):
    batch_size: int = 1  # small batch_size => large between-query diversity

    def _model_transform(self, D):
        """ return left = U and right = VT such that score = U @ VT, before adding the prior_score """

    def _add_noise(self, x, num_samples=0):
        """ return random embeddings; expand on first dimension if num_samples > 0 """

    def _process_batch(self, left, right, k, prior_score=None):
        left = self._add_noise(left, k)  # k,b,h
        right = self._add_noise(right.T, k).swapaxes(-2, -1)  # k,h,v
        R = torch.bmm(left, right)  # k,b,v
        if prior_score is not None:
            R = R + auto_cast_lazy_score(prior_score).as_tensor(R.device)
        batch_first = R.swapaxes(0, 1)  # b,k,v
        return batch_first.topk(k).indices.cpu()  # b, k (draws), k (ranked)

    def __call__(self, D, k):
        left, right = self._model_transform(D)
        topk_indices = []
        for i in tqdm.tqdm(range(0, left.shape[0], self.batch_size)):
            batch_left = left[i:i + self.batch_size]
            batch_prior = D.prior_score[i:i + self.batch_size] if D.prior_score is not None else None
            batch_topk = self._process_batch(batch_left, right, k, batch_prior)
            topk_indices.append(batch_topk)
        topk_indices = torch.cat(topk_indices).numpy()  # q, k (draws), k (ranked)

        num_per_list = [1] * k
        return np.vstack([merge_unique(lol, num_per_list, k)[0] for lol in topk_indices])


@dataclasses.dataclass
class LatentNoiseAgent(LatentNoiseAgentBase):
    std: float = 2e-2

    def _model_transform(self, D):
        S = self.model.transform(D)  # TODO: extract before layer norm
        assert hasattr(S, "op") and S.op == operator.matmul, "only work for low-rank scores"
        return S.children[0].as_tensor(auto_device()), S.children[1].as_tensor(auto_device())

    def _add_noise(self, x, num_samples=0):
        in_shape = x.shape
        out_shape = in_shape if num_samples == 0 else [num_samples, *in_shape]
        return F.layer_norm(x + self.std * torch.randn_like(x.expand(out_shape)), in_shape[-1:])


@dataclasses.dataclass
class VAEAgent(LatentNoiseAgentBase):
    item_df: pd.DataFrame = None
    model_name: str = "distilbert-base-uncased"
    max_length: int = 32

    @property
    def vae_model(self):
        model = self.model
        while not isinstance(model, EmbeddingModel):
            if hasattr(model, 'model'):
                model = model.model
            elif hasattr(model, 'item_tower'):
                model = model.item_tower
            else:
                raise ValueError(f"not understanding {model}")
        return model

    @functools.cached_property
    @torch.no_grad()
    def _cached_mu_std(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        mu_std = []
        for i in tqdm.tqdm(range(0, len(self.item_df), 10)):
            batch_inputs = tokenizer(
                self.item_df.iloc[i:i + 10]['TITLE'].tolist(),
                truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            batch_mu_std = self.vae_model(**batch_inputs.to(self.vae_model.device), return_mean_std=True)
            mu_std.append(torch.cat(batch_mu_std, -1))
        return torch.cat(mu_std)  # auto_device

    def _model_transform(self, D):
        i_to_ptr = self.item_df.index.get_indexer(D.user_in_test['_hist_items'].apply(lambda x: x[0]).tolist())
        j_to_ptr = self.item_df.index.get_indexer(D.item_in_test.index.values)
        return self._cached_mu_std[i_to_ptr], self._cached_mu_std[j_to_ptr].T

    def _add_noise(self, x, num_samples=0):
        mu, std = x.split(x.shape[-1] // 2, -1)
        in_shape = mu.shape
        out_shape = in_shape if num_samples == 0 else [num_samples, *in_shape]
        hidden_states = mu + std * torch.randn_like(mu.expand(out_shape))
        hidden_states = self.vae_model.vocab_transform(hidden_states)
        hidden_states = self.vae_model.activation(hidden_states)
        return self.vae_model.standard_layer_norm(hidden_states)

    def __call__(self, D, k):
        self.vae_model.to(auto_device()).eval()
        out = super().__call__(D, k)
        self.vae_model.to('cpu')
        return out
