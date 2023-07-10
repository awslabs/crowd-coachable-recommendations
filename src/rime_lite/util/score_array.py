import numpy as np, pandas as pd
import torch, dataclasses, warnings, operator, builtins, numbers, os, itertools, functools
from typing import List, ClassVar
from torch.utils.data import DataLoader
import scipy.sparse as sps


def get_batch_size(shape):
    """round to similar batch sizes"""
    n_users, n_items = shape
    frac = float(os.environ.get("BATCH_SIZE_FRAC", 0.1))
    if torch.cuda.device_count():
        total_memory = torch.cuda.get_device_properties(0).total_memory
    else:
        total_memory = 16e9
    max_batch_size = total_memory / 8 / n_items * frac
    n_batches = int(n_users / max_batch_size) + 1
    return int(np.ceil(n_users / n_batches))


def find_iloc(old_index, new_index, allow_missing=False):
    if not isinstance(old_index, pd.Index):
        old_index = pd.Index(old_index)
    iloc = old_index.get_indexer(new_index)
    if not allow_missing and -1 in iloc:
        raise IndexError("missing indices detected in a disallowed case")
    return iloc


def matrix_reindex(csr, old_index, new_index, axis, fill_value=0):
    """pandas.reindex functionality on sparse or dense matrices as well as 1d arrays"""
    if axis == 1:
        return matrix_reindex(csr.T, old_index, new_index, 0, fill_value).T.copy()
    assert axis == 0, "axis must be 0 or 1"
    assert csr.shape[0] == len(old_index), "shape must match between csr and old_index"

    if isinstance(csr, LazyScoreBase):
        pass  # does not support extrapolation
    elif sps.issparse(csr):
        csr = sps.vstack([csr, csr[:1] * 0 + fill_value], "csr")
        csr.eliminate_zeros()
    else:
        csr = np.concatenate([csr, csr[:1] * 0 + fill_value], axis=0)

    iloc = find_iloc(old_index, new_index, allow_missing=True)
    return csr[iloc]


def sps_to_torch(x, device):
    """convert scipy.sparse to torch.sparse"""
    coo = x.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device)


def auto_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def auto_tensor(x, device=None):
    if device is None:
        device = auto_device()
    if hasattr(x, "as_tensor"):
        return x.as_tensor(device)
    elif sps.issparse(x):
        return sps_to_torch(x, device).to_dense()
    else:
        return torch.as_tensor(x, device)


class LazyScoreBase:
    """low-rank, sparse, and other expressions with deferred evaluations after subsampling

    * Support expressions like (a @ b.T + c).exp() * d + e
    * Support operations T, __getitem__, collate_fn
    * Lazy evaluation until as_tensor(device) or numpy()
    Drop support for reindex because it is nontrivial to find fill_values through activation.
    """

    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return f"<{type(self).__name__} {self.shape}>"

    def numpy(self):
        return self.as_tensor().numpy()

    def as_tensor(self, device=None):
        raise NotImplementedError

    @property
    def T(self):
        raise NotImplementedError

    def __getitem__(self, key):
        """slice by rows for pytorch dataloaders"""
        raise NotImplementedError

    @staticmethod
    def collate_fn(D):
        """collate by rows for pytorch dataloaders"""
        raise NotImplementedError

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def batch_size(self):
        return get_batch_size(self.shape)

    def __matmul__(self, other):
        return MatMulExpression(operator.matmul, [self, other])

    def __add__(self, other):
        return ElementWiseExpression(operator.add, [self, other])

    def __mul__(self, other):
        return ElementWiseExpression(operator.mul, [self, other])

    def __sub__(self, other):
        return ElementWiseExpression(operator.sub, [self, other])

    def __truediv__(self, other):
        return ElementWiseExpression(operator.truediv, [self, other])

    def exp(self):
        return ElementWiseExpression(torch.exp, [self])

    def softplus(self):
        return ElementWiseExpression(torch.nn.functional.softplus, [self])

    def sigmoid(self):
        return ElementWiseExpression(torch.sigmoid, [self])

    def apply(self, op):
        return ElementWiseExpression(op, [self])

    def vae_module(self, module):
        op = _VAEOp(module.training, module.prior, module.beta)
        rand_shape = (self.shape[0], self.shape[1] // 2)
        return VAEExpression(op, [self, RandScore.create(rand_shape, "randn")])


def auto_cast_lazy_score(other):
    if other is None:
        return None  # prior_score=None -> None
    elif isinstance(other, LazyScoreBase):
        return other
    elif sps.issparse(other):
        return LazySparseMatrix(other)
    elif isinstance(other, pd.DataFrame):
        return LazyDenseMatrix(other.values)
    elif torch.is_tensor(other):
        return LazyDenseMatrix(other.detach().cpu().numpy())
    else:  # scalar or numpy arrays of 0, 1, 2 dimensions
        return LazyDenseMatrix(other)


class LazySparseMatrix(LazyScoreBase):
    def __init__(self, c):
        self.c = c.tocsr()
        self.shape = c.shape

    def numpy(self):
        return self.c.toarray()

    def as_tensor(self, device=None):
        return sps_to_torch(self.c, device).to_dense()

    @property
    def T(self):
        return self.__class__(self.c.T)

    def __getitem__(self, key):
        if np.isscalar(key):
            slc = slice(self.c.indptr[key], self.c.indptr[key + 1])
            _dict = {
                "values": self.c.data[slc],
                "keys": self.c.indices[slc],
                "shape": self.c.shape[1],
            }
            return _LazySparseDictFast(_dict)
        else:
            return self.__class__(self.c[key])

    @staticmethod
    def collate_fn(D):
        return D[0].__class__(sps.vstack([d.c for d in D]))


class _LazySparseDictFast(LazyScoreBase):
    def __init__(self, c):
        self.c = c
        self.shape = (1, self.c["shape"])

    @staticmethod
    def collate_fn(D):
        C = [d.c for d in D]
        csr = sps.csr_matrix(
            (
                np.hstack([c["values"] for c in C]),  # data
                np.hstack([c["keys"] for c in C]),  # indices
                np.hstack([[0], np.cumsum([len(c["keys"]) for c in C])]),  # indptr
            ),
            shape=(len(C), C[0]["shape"]),
        )
        return LazySparseMatrix(csr)


class LazyDenseMatrix(LazyScoreBase):
    """cast scalar and arrays to 2-d arrays"""

    def __init__(self, c):
        self.c = np.array(c, ndmin=2)
        self.shape = self.c.shape

    def numpy(self):
        return self.c

    def as_tensor(self, device=None):
        return torch.as_tensor(self.c, device=device)

    @property
    def T(self):
        return self.__class__(self.c.T)

    def __getitem__(self, key):
        """take modulos for broadcasting purposes"""
        if isinstance(key, slice):
            key = range(key.stop)[key]
        key = np.array(key, ndmin=1) % self.shape[0]
        return self.__class__(self.c[key])

    @staticmethod
    def collate_fn(D):
        return D[0].__class__(np.vstack([d.c for d in D]))


def _get_op_name(op):
    if hasattr(op, "__name__"):
        return op.__name__
    elif hasattr(op, "training"):
        return f"{op}({op.training})"
    else:
        return repr(op)


class LazyExpressionBase:
    def __init__(self, op, children):
        self.op = op
        self.children = [auto_cast_lazy_score(c) for c in children]
        self.__post_init__()

    def __post_init__(self):
        pass

    def traverse(self, op_func=_get_op_name):
        builder = ""
        for i, c in enumerate(self.children):
            if hasattr(c, "traverse"):
                builder = builder + f"({c.traverse(op_func)})"
            else:
                builder = builder + f"{c}"
            if i == 0:
                builder = builder + f" {op_func(self.op)} "
        return builder

    def train(self):
        self.traverse(
            lambda op: setattr(op, "training", True)
            if hasattr(op, "training")
            else None
        )

    def eval(self):
        self.traverse(
            lambda op: setattr(op, "training", False)
            if hasattr(op, "training")
            else None
        )

    def __repr__(self):
        return self.traverse()

    def as_tensor(self, device=None):
        children = [c.as_tensor(device) for c in self.children]
        return self.op(*children)


class ElementWiseExpression(LazyExpressionBase, LazyScoreBase):
    """Tree representation of an element-wise expression; auto-broadcast"""

    def __post_init__(self):
        shape = np.transpose([c.shape for c in self.children])
        self.shape = (max(shape[0]), max(shape[1]))  # consider broadcast

    @property
    def T(self):
        children = [c.T for c in self.children]
        return self.__class__(self.op, children)

    def __getitem__(self, key):
        children = [c[key] for c in self.children]
        return self.__class__(self.op, children)

    @staticmethod
    def collate_fn(batch):
        op, template = batch[0].op, batch[0].children
        data = zip(*[b.children for b in batch])
        children = [c.collate_fn(D) for c, D in zip(template, data)]
        return batch[0].__class__(op, children)


class MatMulExpression(LazyExpressionBase, LazyScoreBase):
    def __post_init__(self):
        self.left, self.right = self.children
        self.shape = (self.left.shape[0], self.right.shape[1])

        assert (
            self.left.shape[1] == self.right.shape[0]
        ), f"matmul shape check fail: {self.left.shape} vs {self.right.shape}"

    @property
    def T(self):
        return self.__class__(self.op, [self.right.T, self.left.T])

    def __getitem__(self, key):
        return self.__class__(self.op, [self.left[key], self.right])

    @staticmethod
    def collate_fn(batch):
        left = batch[0].left.__class__.collate_fn([b.left for b in batch])
        return batch[0].__class__(batch[0].op, [left, batch[0].right])


class VAEExpression(LazyExpressionBase, LazyScoreBase):
    """op=_VAEOp, [weight (n, 2d), randn (n, d)], auto-broadcast on d-axis"""

    @property
    def axis(self):
        return int(isinstance(self.right, RandScore) and self.right.distn == "randn")

    def __post_init__(self):
        self.left, self.right = self.children
        self.shape = (self.left.shape[0], self.right.shape[1])

        assert _vae_shape_check(
            self.left.shape, self.right.shape, self.axis
        ), f"vae(axis={self.axis}) shape check fail: {self.left.shape} vs {self.right.shape}"

    @property
    def T(self):
        return self.__class__(self.op, [self.right.T, self.left.T])

    def __getitem__(self, key):
        if self.axis == 1:
            return ElementWiseExpression.__getitem__(self, key)
        else:
            return MatMulExpression.__getitem__(self, key)

    @staticmethod
    def collate_fn(batch):
        if batch[0].axis == 1:
            return ElementWiseExpression.collate_fn(batch)
        else:
            return MatMulExpression.collate_fn(batch)

    def as_tensor(self, device):
        if self.axis == 1:
            weight = self.left.as_tensor(device)
            randn = self.right.as_tensor(device)
            return self.op(weight, randn).expand(self.shape)
        else:
            return self.T.as_tensor(device).T


def _vae_shape_check(left_shape, right_shape, axis):
    if axis == 1:
        ndim = left_shape[1] // 2
        return left_shape[0] == right_shape[0] and right_shape[1] % ndim == 0
    else:
        return _vae_shape_check(right_shape[::-1], left_shape[::-1], 1 - axis)


class _VAEOp:
    def __init__(self, training=True, prior=0, beta=1):
        self.training = training
        self.beta = beta
        self.prior_std = np.exp(prior / 2)
        self.__name__ = f"vae({training}*{beta})"

    def __call__(self, weight, randn):
        mean, logvar = torch.split(weight, weight.shape[-1] // 2, -1)
        noise = randn * (logvar / 2).exp() * self.training * self.beta
        return (mean + noise) * self.prior_std


@dataclasses.dataclass(repr=False)
class RandScore(LazyScoreBase):
    """add random noise to break ties"""

    row_seeds: list  # np.array for fast indexing
    col_seeds: list  # np.array for fast indexing
    distn: str = "rand"
    _DEFAULT_MAX_SEED: ClassVar[int] = 10000

    @property
    def shape(self):
        return (len(self.row_seeds), len(self.col_seeds))

    @classmethod
    def create(cls, shape, distn="rand"):
        return cls(
            np.random.choice(cls._DEFAULT_MAX_SEED) + np.arange(shape[0]),
            np.random.choice(cls._DEFAULT_MAX_SEED) + np.arange(shape[1]),
            distn,
        )

    def numpy(self):
        return np.vstack(
            [
                getattr(np.random.RandomState(int(s)), self.distn)(len(self.col_seeds))
                for s in self.row_seeds
            ]
        )

    def as_tensor(self, device=None):
        rows = []
        for s in self.row_seeds:
            generator = torch.Generator(device).manual_seed(int(s))
            new = getattr(torch, self.distn)(
                len(self.col_seeds), device=device, generator=generator
            )
            rows.append(new)
        return torch.vstack(rows)

    @property
    def T(self):
        warnings.warn("transpose changes rand seed; only for evaluate_user_rec")
        return self.__class__(self.col_seeds, self.row_seeds, self.distn)

    def __getitem__(self, key):
        if isinstance(key, slice):
            key = range(key.stop)[key]
        key = np.array(key, ndmin=1) % self.shape[0]
        return self.__class__(self.row_seeds[key], self.col_seeds, self.distn)

    @staticmethod
    def collate_fn(batch):
        row_seeds = np.hstack([b.row_seeds for b in batch])
        return batch[0].__class__(row_seeds, batch[0].col_seeds, batch[0].distn)


def batch_op_iter(S, op, device=None):
    if isinstance(op, str):
        op = getattr(torch, op)  # max, min, sum
    for i in range(0, len(S), S.batch_size):
        batch = S[i : min(len(S), i + S.batch_size)]
        val = batch.as_tensor(device)
        yield op(val)


def score_op(S, op, device=None, reduce_fn=None):
    """aggregation operations (e.g., max, min, sum) across entire matrix"""
    if reduce_fn is None:
        reduce_fn = {"max": max, "min": min, "sum": operator.add}[op]
    iterable = batch_op_iter(S, op, device)
    return functools.reduce(reduce_fn, iterable)


@dataclasses.dataclass
class LazyScoreModel:
    user_index: pd.Index
    item_index: pd.Index
    lazy_score: LazyScoreBase = None
    tie_breaker: float = 0

    def __post_init__(self):
        shape = (len(self.user_index), len(self.item_index))
        if self.lazy_score is None:
            self.lazy_score = auto_cast_lazy_score(sps.csr_matrix(shape))
        assert (
            self.lazy_score.shape == shape
        ), f"shape inconsistent: {shape}, {self.lazy_score.shape}"

    def transform(self, D):
        old_index = self.user_index
        new_index = D.test_requests.index
        while old_index.nlevels > new_index.nlevels:
            old_index = old_index.droplevel(-1)
        while new_index.nlevels > old_index.nlevels:
            new_index = new_index.droplevel(-1)

        row_ind = old_index.get_indexer(new_index)
        col_ind = self.item_index.get_indexer(D.item_in_test.index)
        S = self.lazy_score.T[col_ind].T[row_ind]
        if self.tie_breaker > 0:
            S = S + RandScore.create(S.shape) * self.tie_breaker
        return S
