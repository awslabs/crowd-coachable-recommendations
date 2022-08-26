import pandas as pd, numpy as np, scipy.sparse as sps, torch, dataclasses, warnings
import torch.nn.functional as F
from rime.util import indices2csr, auto_tensor
from rime.dataset.base import Dataset
from ..agent.base import Agent
from .base import Env, create_zero_shot, parse_response, _expand_na_class
from .i2i_env import I2IEnv, I2IConfig, I2IImageEnv, ExpInfo, download_labels


@dataclasses.dataclass
class DummyEnv(Env):
    oracle: str = None

    def _invoke(self, request, D, step_idx):
        return request.assign(multi_label=request['_group'].apply(np.ones_like))


@dataclasses.dataclass
class HoldoutEnv(Env):
    """ get ground truth from holdout target_csr """
    oracle: Dataset = "required"

    def _invoke(self, request, D, step_idx):
        target_csr = self.oracle.reindex(request.index, axis=0).target_csr
        request_J = request['cand_items'].apply(lambda x: [self._tokenize[y] for y in x])

        target_csr = target_csr.copy()
        target_csr.sum_duplicates()
        indices = np.split(target_csr.indices, target_csr.indptr[1:-1])
        data = np.split(target_csr.data, target_csr.indptr[1:-1])
        indices_to_data = [dict(zip(t, d)) for t, d in zip(indices, data)]
        multi_label = np.asarray([[i2d.get(e, 0.0) for e in j]
                                  for j, i2d in zip(request_J, indices_to_data)])

        return request.assign(multi_label=multi_label.tolist())


@dataclasses.dataclass
class SimuEnv(Env):
    """ get ground truth from the probability scores of an oracle model """
    oracle: Agent = "required"
    soft_label: bool = True
    reserve_score: float = 0

    def _invoke(self, request, D, step_idx):
        score = self.oracle._get_scores(D)
        request_J = request['cand_items'].apply(lambda x: [self._tokenize[y] for y in x])

        multi_label = []
        for i in range(0, len(score), score.batch_size):
            iloc = slice(i, min(len(score), i + score.batch_size))
            gnd = auto_tensor(score[iloc])
            J = torch.as_tensor(np.vstack(request_J.iloc[iloc]), device=gnd.device)

            probs = torch.gather(gnd, 1, J).clip(0, None)
            if self.reserve_score:
                probs = F.pad(probs, (0, 1, 0, 0), value=self.reserve_score)
                request = _expand_na_class(request)
            if probs.sum(1).min() == 0:
                warnings.warn("weird things happened here; probs sum = 0")
                probs = probs.clip(1e-10, None)

            if self.soft_label:
                probs = (probs / probs.sum(1, keepdims=True)).cpu().numpy()
                multi_label.append(np.round(probs, 1))
            else:
                label = torch.multinomial(probs, 1).ravel()
                multi_label = F.one_hot(label, probs.shape[1]).cpu().numpy()

        return request.assign(multi_label=np.vstack(multi_label).tolist())


def auto_env_select(*args, oracle=None, **kw):
    if oracle is None:
        cls = Env
    elif isinstance(oracle, I2IConfig):
        cls = I2IImageEnv if oracle.image else I2IEnv
    elif isinstance(oracle, Dataset):
        cls = HoldoutEnv
    elif isinstance(oracle, Agent):
        cls = SimuEnv
    else:
        raise NotImplementedError(f"oracle not understood: {oracle}")
    return cls(*args, oracle=oracle, **kw)
