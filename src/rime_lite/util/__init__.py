import pandas as pd, numpy as np, scipy as sp
import functools, collections, time, contextlib, torch, gc, warnings, json
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property
from .score_array import *  # noqa: F401, F403
from tensorboard.backend.event_processing import event_accumulator


class timed(contextlib.ContextDecorator):
    def __init__(self, name="", inline=True):
        self.name = name
        self.inline = inline

    def __enter__(self):
        self.tic = time.time()
        print("timing", self.name, end=" " if self.inline else "\n")

    def __exit__(self, *args, **kw):
        print(
            "done",
            "." if self.inline else self.name,
            "time {:.1f}s".format(time.time() - self.tic),
        )


def warn_nan_output(func):
    @functools.wraps(func)
    def wrapped(*args, **kw):
        out = func(*args, **kw)
        values = getattr(out, "values", out)

        if hasattr(values, "has_nan"):
            has_nan = values.has_nan()
        else:
            has_nan = np.isnan(values).any()

        if has_nan:
            warnings.warn(f"{func.__name__} output contains NaN", stacklevel=2)
        return out

    return wrapped


def _empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def _get_cuda_objs():
    objs = []
    for obj in gc.get_objects():
        try:
            flag = torch.is_tensor(obj)  # or \
            # (hasattr(obj, 'data') and torch.is_tensor(obj.data))
        except Exception:
            flag = False
        try:
            if flag and torch.device(obj.device) != torch.device("cpu"):
                objs.append(obj)
        except RuntimeError as e:
            warnings.warn(str(e))
    return objs


def empty_cache_on_exit(func):
    @functools.wraps(func)
    def wrapped(*args, **kw):
        _empty_cache()
        start_list = _get_cuda_objs()

        tic = time.time()
        out = func(*args, **kw)
        print(func.__name__, "time {:.1f}s".format(time.time() - tic))

        _empty_cache()
        end_list = _get_cuda_objs()
        for obj in set(end_list) - set(start_list):
            print(
                func.__name__,
                "memory leak",
                type(obj),
                obj.size(),
                obj.device,
                flush=True,
            )

        del start_list
        del end_list
        _empty_cache()
        return out

    return wrapped


@contextlib.contextmanager
def _to_cuda(model):
    if torch.cuda.is_available():
        orig_device = model.device
        print("running model on cuda device")
        yield model.to("cuda")
        model.to(orig_device)
    else:
        yield model


def perplexity(x):
    x = np.ravel(x) / x.sum()
    return float(np.exp(-x @ np.log(np.where(x > 0, x, 1e-10))))


@empty_cache_on_exit
def _assign_topk(S, k, tie_breaker=1e-10, device="cpu", batch_size=None):
    """Return a sparse matrix where each row contains k non-zero values.

    Used for both ItemRec (if S is user-by-item) and UserRec (if S is transposed).
    Expect the shape to align with (len(D.user_in_test), len(D.item_in_test))
    or its transpose.
    """
    indices = []
    if hasattr(S, "collate_fn"):
        if batch_size is None and hasattr(S, "batch_size"):
            batch_size = S.batch_size
        batches = map(
            lambda i: S[i : min(len(S), i + batch_size)], range(0, len(S), batch_size)
        )
    else:
        batches = [S]
    for s in batches:
        if hasattr(s, "as_tensor"):
            s = s.as_tensor(device)
        else:
            s = torch.tensor(s, device=device)
        if tie_breaker:
            s = s + torch.rand(*s.shape, device=device) * tie_breaker
        s_topk = s.topk(k).indices.cpu().numpy()
        indices.append(s_topk)
    indices = np.vstack(indices)

    return sp.sparse.csr_matrix(
        (
            np.ones(indices.size),
            np.ravel(indices),
            np.arange(0, indices.size + 1, indices.shape[1]),
        ),
        shape=S.shape,
    )


assign_topk = _assign_topk


@empty_cache_on_exit
def _argsort(S, tie_breaker=1e-10, device="cpu"):
    print(f"_argsort {S.size:,} scores on device {device}; ", end="")
    if hasattr(S, "batch_size") and S.batch_size < S.shape[0]:
        warnings.warn(f"switching numpy.argsort due to {S.batch_size}<{S.shape[0]}")
        device = None

    if hasattr(S, "as_tensor"):
        S = S.as_tensor(device)

    shape = S.shape

    if device is None:
        if tie_breaker > 0:
            S = S + np.random.rand(*S.shape) * tie_breaker
        S = -S.reshape(-1)
        _empty_cache()
        argsort_ind = np.argsort(S)
    else:
        S = torch.as_tensor(S, device=device)
        if tie_breaker > 0:
            S = S + torch.rand(*S.shape, device=device) * tie_breaker
        S = -S.reshape(-1)
        _empty_cache()
        argsort_ind = torch.argsort(S).cpu().numpy()

    return np.unravel_index(argsort_ind, shape)


def extract_user_item(event_df):
    user_df = event_df.groupby("USER_ID").agg(
        _Tmin=("TIMESTAMP", min), _Tmax=("TIMESTAMP", max)
    )
    item_df = event_df.groupby("ITEM_ID").size().to_frame("_siz")
    return (user_df, item_df)


def groupby_unexplode(series, index=None, return_type="series"):
    """
    assume the input is an exploded dataframe with block-wise indices
    >>> groupby_unexplode(pd.Series([1,2,3,4,5], index=[1,1,2,3,3])).to_dict()
    {1: [1, 2], 2: [3], 3: [4, 5]}
    >>> groupby_unexplode(pd.Series([1,2,3,4,5], index=[1,1,2,3,3]), index=[0,1,-1,2,3,4]).to_dict()
    {0: [], 1: [1, 2], -1: [], 2: [3], 3: [4, 5], 4: []}
    """
    if index is None:
        splits = (
            np.where(
                np.array(series.index.values[1:])
                != np.array(series.index.values[:-1])  # 1, 2, 3, 3  # 1, 1, 2, 3
            )[0]
            + 1
        )  # [2, 3]
        index = series.index.values[np.hstack([[0], splits])]  # 1, 2, 3
    elif len(series) == 0:
        splits = [0 for _ in range(len(index) - 1)]
    else:  # something like searchsorted, but unordered
        splits = []
        tape = enumerate(series.index)
        N = len(series)
        i, value = next(tape)

        for key in index:
            splits.append(i)
            while i < N and key == value:
                i, value = next(tape, (N, None))  # move past the current chunk
        assert i == N, f"mismatch between series and index. {series.index} vs {index}"
        splits = splits[1:]

    if return_type == "splits":
        return splits
    else:
        return pd.Series(
            [x.tolist() for x in np.split(series.values, splits)], index=index
        )


def indices2csr(indices, shape1, data=None):
    indptr = np.cumsum([0] + [len(x) for x in indices])
    if data is None:
        data = [np.ones(indptr[-1])]
    return sps.csr_matrix(
        (np.hstack(data), np.hstack(indices), indptr), shape=(len(indices), shape1)
    )


def extract_past_ij(user_df, item_index):
    item_map = {k: j for j, k in enumerate(item_index)}
    past_event_df = (
        user_df.reset_index()[user_df["_hist_len"].values > 0][  # drop empty users
            "_hist_items"
        ]
        .explode()
        .apply(lambda x: item_map.get(x, -1))
    )
    past_event_df = past_event_df[past_event_df > -1]  # ignore oov items
    return (past_event_df.index.values, past_event_df.values)


def fill_factory_inplace(df, isna, kv):
    for k, v in kv.items():
        if k is None:  # series
            df[:] = [v() if do else x for do, x in zip(isna, df.values)]
        elif k in df.columns:
            df[k] = [v() if do else x for do, x in zip(isna, df[k])]
        else:
            warnings.warn(f"fill_factory_inplace missing {k}")


def sample_groupA(user_df, frac=0.5, seed=888):
    return user_df.index.isin(user_df.sample(frac=frac, random_state=seed).index)


def filter_min_len(event_df, min_user_len, min_item_len):
    """CAVEAT: use in conjunction with dataclass filter to avoid future-leaking bias"""
    users = event_df.groupby("USER_ID").size()
    items = event_df.groupby("ITEM_ID").size()
    return event_df[
        event_df["USER_ID"].isin(users[users >= min_user_len].index)
        & event_df["ITEM_ID"].isin(items[items >= min_item_len].index)
    ]


def explode_user_titles(
    user_hist, item_titles, gamma=0.5, min_gamma=0.1, pad_title="???"
):
    """explode last few user events and match with item titles;
    return splits and discount weights; empty user_hist will be turned into a single pad_title.
    """

    keep_last = (
        int(np.log(min_gamma) / np.log(np.clip(gamma, 1e-10, 1 - 1e-10))) + 1
    )  # default=4

    explode_titles = (
        pd.Series([x[-keep_last:] for x in user_hist.values])
        .explode()
        .to_frame("ITEM_ID")
        .join(item_titles.to_frame("TITLE"), on="ITEM_ID")["TITLE"]
    )
    explode_titles = pd.Series(
        [
            x if not na else pad_title
            for x, na in zip(explode_titles.tolist(), explode_titles.isna().tolist())
        ],
        index=explode_titles.index,
    )

    splits = (
        np.where(
            np.array(explode_titles.index.values[1:])
            != np.array(explode_titles.index.values[:-1])
        )[0]
        + 1
    )

    weights = np.hstack(
        [
            gamma ** (np.cumsum(x) - np.sum(x))  # -2, -1, 0
            for x in np.split(np.ones(len(explode_titles)), splits)
        ]
    )
    weights = np.hstack([x / x.sum() for x in np.split(weights, splits)])

    return explode_titles.values, splits, weights


def _get_loss_value(loss):
    if isinstance(loss, dict):
        return loss["loss"]
    else:
        return loss


class _LitValidated(LightningModule):
    def training_step(self, batch, batch_idx):
        if hasattr(self, "training_and_validation_step"):
            loss = self.training_and_validation_step(batch, batch_idx)
            self.log(
                "loss",
                _get_loss_value(loss),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return loss  # value or dictionary
        else:
            raise NotImplementedError("alternatively, define training_step in subclass")

    def validation_step(self, batch, batch_idx):
        training_and_validation_step = getattr(
            self, "training_and_validation_step", self.training_step
        )
        loss = training_and_validation_step(batch, batch_idx)
        self.log(
            "val_loss",
            _get_loss_value(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return _get_loss_value(loss)

    @cached_property
    def _checkpoint(self):
        return ModelCheckpoint(monitor="val_loss", save_weights_only=True)

    def _load_best_checkpoint(self, msg="loading"):
        best_model_path = self._checkpoint.best_model_path
        best_model_score = self._checkpoint.best_model_score
        if best_model_score is not None:
            print(f"{msg} checkpoint {best_model_path} with score {best_model_score}")
            self.load_state_dict(torch.load(best_model_path)["state_dict"])
        else:
            warnings.warn("No checkpoints found!")


class _ReduceLRLoadCkpt(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, model, **kw):
        super().__init__(*args, **kw)
        self.model = model

    def _reduce_lr(self, epoch):
        super()._reduce_lr(epoch)
        self.model._load_best_checkpoint()


def default_random_split(dataset):
    N = len(dataset)
    if N >= 5:
        return random_split(dataset, [N * 4 // 5, N - N * 4 // 5])
    else:
        warnings.warn(
            f"short dataset len={len(dataset)}; "
            "setting valid_set identical to train_set"
        )
        return dataset, dataset


@dataclasses.dataclass
class MissingModel:
    name: str
    err: Exception
    verbose: bool = True

    def __post_init__(self):
        if self.verbose:
            warnings.warn(f"Model {self.name} is missing due to {self.err}")


def export_jsondump(writer):
    # minor modifications from https://discuss.pytorch.org/t/tensorboard-json-dump-of-all-scalars/69334

    if hasattr(writer, "log_dir"):
        log_dir = writer.log_dir
    elif hasattr(writer, "_get_file_writer"):
        log_dir = writer._get_file_writer().get_logdir()
    else:
        warnings.warn("export jsondump failure")
        return

    tf_files = []  # -> list of paths from writer.log_dir to all files in that directory
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            tf_files.append(
                os.path.join(root, file)
            )  # go over every file recursively in the directory

    for file_id, file in enumerate(tf_files):
        path = os.path.join(
            "/".join(file.split("/")[:-1])
        )  # determine path to folder in which file lies
        # seperate file created by add_scalar from add_scalars
        name = (
            os.path.join(file.split("/")[-2])
            if file_id > 0
            else os.path.join("log_events")
        )

        event_acc = event_accumulator.EventAccumulator(file)
        event_acc.Reload()
        data = {}

        hparam_file = False  # I save hparam files as 'hparam/xyz_metric'
        for tag in sorted(event_acc.Tags()["scalars"]):
            if tag.split("/")[0] == "hparam":
                hparam_file = True  # check if its a hparam file
            step, value = [], []

            for scalar_event in event_acc.Scalars(tag):
                step.append(scalar_event.step)
                value.append(scalar_event.value)

            data[tag] = (step, value)

        if not hparam_file and bool(
            data
        ):  # if its not a hparam file and there is something in the data -> dump it
            with open(f"{path}/{name}.json", "w") as f:
                json.dump(data, f)
