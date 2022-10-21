import dataclasses, torch, time, os, tqdm, typing
import pandas as pd, numpy as np, scipy.sparse as sps
from torch.distributions.categorical import Categorical
from rime.util import auto_device, empty_cache_on_exit, score_op
from ccrec.agent.base import Agent


def softmax_sample(S, k, replacement=False, shuffle=True):
    indices = (
        score_op(
            S,
            lambda x: torch.multinomial(x.softmax(1), k, replacement=replacement),
            auto_device(),
            lambda x, y: torch.vstack([x, y]),
        )
        .cpu()
        .numpy()
    )
    if shuffle:
        indices = np.array([np.random.permutation(x) for x in indices])
    return indices


def search_temperature(S, k, target_ppl, left, right, n_steps=50):
    left = np.broadcast_to(np.ravel(left), S.shape[0])
    right = np.broadcast_to(np.ravel(right), S.shape[0])

    for _ in range(n_steps):
        temp = (left + right) / 2
        ppl = (
            score_op(
                S / temp.reshape((-1, 1)),
                lambda x: Categorical(logits=x).entropy(),
                auto_device(),
                lambda x, y: torch.hstack([x, y]),
            )
            .exp()
            .cpu()
            .numpy()
        )

        # left < ppl < target < right
        left = np.where(ppl <= target_ppl, temp, left)

        # left < target < ppl < right
        right = np.where(ppl > target_ppl, temp, right)

    print(f"temp={np.mean(temp):.3f}+-{np.std(temp):.3f}, ppl={np.mean(ppl):.1f}")
    return temp


@dataclasses.dataclass
class BoltzmannAgent(Agent):
    target_ppl: float = 50
    min_temp: float = 0
    max_temp: float = 100
    shuffle: bool = True
    _last_S: typing.Any = None  # store last S to quickly resample and visualize the diversity
    _last_temperature: list = None  # reflect the confidence of recommendation

    @empty_cache_on_exit
    def __call__(self, D, k, _use_last_S=False):
        if not _use_last_S or self._last_S is None:
            S = self.model.transform(D)
            if (
                hasattr(S, "op")
                and hasattr(S.op, "__name__")
                and S.op.__name__ == "softplus"
            ):
                S = S.children[0]
            if hasattr(D, "prior_score"):
                S = S + D.prior_score
            self._last_temperature = search_temperature(
                S, k, self.target_ppl, self.min_temp, self.max_temp
            )
            self._last_S = S / self._last_temperature.reshape((-1, 1))
        return softmax_sample(self._last_S, k, shuffle=self.shuffle)
