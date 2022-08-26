import dataclasses, torch, time, os
import pandas as pd, numpy as np, scipy.sparse as sps
from rime.util import _assign_topk, auto_device, auto_cast_lazy_score, empty_cache_on_exit, RandScore
from .base import Agent
from rime.metrics.dual import Dual
from ..util import merge_unique
from .boltzmann_agent import BoltzmannAgent
from .latent_noise_agent import LatentNoiseAgent, VAEAgent


@dataclasses.dataclass
class RandomAgent(Agent):
    def __call__(self, D, k):
        S = self._get_scores(D, RandScore.create(D.shape))  # respect prior filters
        return _assign_topk(S, k, device=auto_device()).indices.reshape((len(D), k))


@dataclasses.dataclass
class GreedyAgent(Agent):
    def __call__(self, D, k):
        S = self._get_scores(D)
        return _assign_topk(S, k, device=auto_device()).indices.reshape((len(D), k))


@dataclasses.dataclass
class EpsAgent(Agent):
    epsilon: float = 0.0

    def __call__(self, D, k):
        S = self._get_scores(D)
        R = self._get_scores(D, RandScore.create(D.shape))  # respect prior filters
        topk_indices = _assign_topk(S, k, device=auto_device()).indices.reshape((len(D), k))
        rand_indices = _assign_topk(R, k, device=auto_device()).indices.reshape((len(D), k))

        num_per_list = [int(np.ceil(k * (1 - self.epsilon))), int(np.ceil(k * self.epsilon))]
        out = np.vstack([merge_unique([t, r], num_per_list, k)[0]
                         for t, r in zip(topk_indices, rand_indices)])
        return out


@dataclasses.dataclass
class DualAgent(Agent):
    alpha_lb: float = -1.0
    alpha_ub: float = 2.0
    beta_lb: float = -1.0
    beta_ub: float = 2.0

    @empty_cache_on_exit
    def __call__(self, D, k, V=None):
        T = self._get_scores(D)
        S = self._get_scores(V) if V is not None else T

        dual = Dual(S, self.alpha_lb, self.alpha_ub, self.beta_lb, self.beta_ub, device=auto_device())
        dual.fit(S)
        T = auto_cast_lazy_score(T) / dual.score_max - dual.model.v[None, :]
        return _assign_topk(T, k, device=auto_device()).indices.reshape((len(D), k))
