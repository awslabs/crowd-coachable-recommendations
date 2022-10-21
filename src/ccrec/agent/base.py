import os, warnings, dataclasses, torch, time
import pandas as pd, numpy as np, scipy.sparse as sps
from rime.util import empty_cache_on_exit, LazyExpressionBase, RandScore


@dataclasses.dataclass
class Agent:
    """ each fit of the model forgets about previous fits

    graph_conv_factory = lambda: rime.models.GraphConv(
        D, sample_with_prior=True, sample_with_posterior=0, #user_rec=False,
        user_conv_model='plain_average', truncated_input_steps=10, max_epochs=max_epochs,
        training_prior_fcn = lambda x: (x + 0.1 / x.shape[1]).clip(0, None).log())
    """

    model: object
    training: bool = False  # dropout / vae

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @empty_cache_on_exit
    def fit(self, data):
        self.model.fit(data)

    def _get_scores(self, D, S=None):
        if S is None:
            S = self.model.transform(D)
        if D.prior_score is not None:
            S = S + D.prior_score
        if isinstance(S, LazyExpressionBase):
            S.train() if self.training else S.eval()
        return S

    def __call__(self, D, k):
        raise NotImplementedError("return indices")
