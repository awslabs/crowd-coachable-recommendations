import pandas as pd, numpy as np, scipy.sparse as sps, torch, dataclasses, warnings
import torch.nn.functional as F
from rime.util import indices2csr, auto_device
from rime.dataset.base import Dataset
from ..agent.base import Agent
from .base import Env, create_zero_shot, create_reranking_dataset, _expand_na_class, parse_response
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
        expl_candidate_index = request['cand_items'].apply(lambda x: [self._tokenize[y] for y in x])

        target_csr = target_csr.copy()
        target_csr.sum_duplicates()
        indices = np.split(target_csr.indices, target_csr.indptr[1:-1])
        data = np.split(target_csr.data, target_csr.indptr[1:-1])
        indices_to_data = [dict(zip(t, d)) for t, d in zip(indices, data)]
        multi_label = np.asarray([[i2d.get(e, 0.0) for e in j]
                                  for j, i2d in zip(expl_candidate_index, indices_to_data)])

        return request.assign(multi_label=multi_label.tolist())


@dataclasses.dataclass
class SimuEnv(Env):
    """ get ground truth from the probability scores of an oracle model """
    oracle: Agent = "required"
    soft_label: bool = True
    reserve_score: float = 0

    def _label_candidate_probs(self, cand_item_ids, oracle_score_ast):
        expl_candidate_index = cand_item_ids.apply(lambda x: [self._tokenize[y] for y in x])
        oracle_score_tensor = oracle_score_ast.as_tensor(auto_device())

        J = torch.as_tensor(np.vstack(expl_candidate_index), device=oracle_score_tensor.device)
        probs = torch.gather(oracle_score_tensor, 1, J).clip(0, None)
        return probs

    def _label_post_processing(self, probs):
        if self.reserve_score:  # expand na class
            probs = F.pad(probs, (0, 1, 0, 0), value=self.reserve_score)

        if probs.sum(1).min() == 0:
            warnings.warn("none of the exploration candidates are available with >=0 probs; defaulting to random labeling")
            probs = probs.clip(1e-10, None)

        if self.soft_label:
            probs = (probs / probs.sum(1, keepdims=True)).cpu().numpy()
            multi_label = np.round(probs, 1)
        else:
            label = torch.multinomial(probs, 1).ravel()
            multi_label = F.one_hot(label, probs.shape[1]).cpu().numpy()

        return multi_label

    def _invoke(self, request, D, step_idx):
        oracle_score_ast = self.oracle._get_scores(D)

        multi_label = []
        for i in range(0, len(oracle_score_ast), oracle_score_ast.batch_size):
            iloc = slice(i, min(len(oracle_score_ast), i + oracle_score_ast.batch_size))

            probs = self._label_candidate_probs(request['cand_items'].iloc[iloc], oracle_score_ast[iloc])

            multi_label_batch = self._label_post_processing(probs)
            multi_label.append(multi_label_batch)

        if self.reserve_score:
            request = _expand_na_class(request)
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
