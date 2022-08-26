import dataclasses
import numpy as np
import rime
from rime.util import LazyScoreModel, auto_cast_lazy_score
from .graph_vae import GraphVAE


@dataclasses.dataclass
class EmpiricalAverageModel(LazyScoreModel):
    item_pseudo: float = 0.1
    tie_breaker: float = 0.001

    def fit(self, D=None):
        if D is not None:
            V = rime.dataset.Dataset(D.user_df, D.item_df, D.event_df, sample_with_prior=1)
            prior_score = auto_cast_lazy_score(V.prior_score).apply(
                lambda x: x.clip(self.item_pseudo, None))
            self.lazy_score = auto_cast_lazy_score(V.target_csr) / prior_score
        return self
