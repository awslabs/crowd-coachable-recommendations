import numpy as np, pandas as pd, scipy.sparse as sps, warnings, torch, operator
from torch.utils.data import DataLoader
from ..util import (
    perplexity,
    _assign_topk,
    empty_cache_on_exit,
    score_op,
    LazyScoreBase,
)


def _multiply(x, y):
    """lazy, sparse, or numpy array"""
    if isinstance(x, LazyScoreBase):
        return x * y
    elif isinstance(y, LazyScoreBase):
        return y * x
    elif sps.issparse(x):
        return x.multiply(y)
    elif sps.issparse(y):
        return y.multiply(x)
    else:  # numpy dense
        return x * y


def _sum(x, axis, device):
    if isinstance(x, LazyScoreBase):
        if axis is None:
            return float(score_op(x, "sum", device))
        elif axis == 0:
            return (
                score_op(x, lambda x: torch.sum(x, axis), device, operator.add)
                .cpu()
                .numpy()
            )
        else:  # axis == 1
            return (
                score_op(
                    x,
                    lambda x: torch.sum(x, axis),
                    device,
                    lambda x, y: torch.hstack([x, y]),
                )
                .cpu()
                .numpy()
            )
    else:
        return x.sum(axis)


@empty_cache_on_exit
def evaluate_assigned(
    target_csr, assigned_csr, score_mat=None, axis=None, min_total_recs=0, device="cpu"
):
    """compare targets and recommendation assignments on user-item matrix

    target_csr: sparse or numpy array
    assigned_csr: sparse, LazyScoreBase, or numpy array
    score_mat: LazyScoreBase or numpy array
    axis: [None, 0, 1]
    """
    hit_axis = _sum(_multiply(target_csr, assigned_csr), axis, device)
    assigned_sum_0 = _sum(assigned_csr, 0, device)
    assigned_sum_1 = _sum(assigned_csr, 1, device)
    min_total_recs = max(min_total_recs, assigned_sum_0.sum())

    out = {
        "prec": np.sum(hit_axis) / min_total_recs,
        "recs/user": assigned_sum_1.mean(),
        "item_cov": (assigned_sum_0 > 0).mean(),  # 1 by n_items
        "item_ppl": perplexity(assigned_sum_0),
        "user_cov": (assigned_sum_1 > 0).mean(),  # n_users by 1
        "user_ppl": perplexity(assigned_sum_1),
    }

    if score_mat is not None:
        obj_sum = _sum(_multiply(score_mat, assigned_csr), None, device)
        out["obj_mean"] = float(obj_sum / min_total_recs)

    if axis is not None:
        ideal = np.ravel(target_csr.sum(axis=axis))
        out["recall"] = (hit_axis / np.fmax(1, ideal)).mean()

    return out


def evaluate_item_rec(target_csr, score_mat, topk, device="cpu", **kw):
    assigned_csr = _assign_topk(score_mat, topk, device=device, **kw)
    return evaluate_assigned(target_csr, assigned_csr, score_mat, axis=1, device=device)
