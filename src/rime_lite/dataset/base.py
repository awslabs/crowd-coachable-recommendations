import pandas as pd, numpy as np
import scipy.sparse as sps
import warnings, dataclasses, functools, os, typing
from ..util import (
    perplexity,
    timed,
    groupby_unexplode,
    indices2csr,
    matrix_reindex,
    fill_factory_inplace,
    LazyScoreBase,
    cached_property,
)


def _sanitize_events(event_df, user_df, item_df):
    event_old = event_df
    event_df = event_df[
        event_df["USER_ID"].isin(user_df.index)
        & event_df["ITEM_ID"].isin(item_df.index)
    ].copy()
    if len(event_df) < len(event_old):
        warnings.warn(
            f"dropping unknown USER_ID or ITEM_ID, #events {len(event_old)} -> {len(event_df)}"
        )

    if "VALUE" not in event_df:
        event_df["VALUE"] = 1  # implicit feedback

    if int(os.environ.get("RIME_WARN_REPEATS", "1")):
        with timed("checking for repeated user-item events"):
            nunique = len(set(event_df.set_index(["USER_ID", "ITEM_ID"]).index))
            if nunique < len(event_df):
                warnings.warn(f"user-item repeat rate {1 - nunique / len(event_df):%}")

    return event_df


def stable_join(left, right, inner=True):
    if inner:
        left = left[left.index.get_level_values(0).isin(set(right.index))]
    return (
        left.assign(left_index=left.index.get_level_values(0))
        .join(right, on="left_index")
        .drop("left_index", axis=1)
    )


def aggregate_user_history(user_df, joined):
    splits = groupby_unexplode(joined, user_df.index, "splits")

    return user_df.assign(
        _hist_items=[x.tolist() for x in np.split(joined["ITEM_ID"].values, splits)],
        _hist_ts=[x.tolist() for x in np.split(joined["TIMESTAMP"].values, splits)],
        _hist_values=[x.tolist() for x in np.split(joined["VALUE"].values, splits)],
    ).assign(_hist_len=lambda x: x["_hist_items"].apply(len))


@dataclasses.dataclass
class DatasetBase:
    """base class contains the global set of users, items, and events for auto-regressive training"""

    user_df: pd.DataFrame  # a global registry of unique users
    item_df: pd.DataFrame  # a global registry of unique items
    event_df: pd.DataFrame = (
        None  # a dataframe containing USER_ID, ITEM_ID, TIMESTAMP, VALUE
    )

    def __post_init__(self):
        """sanitize events and aggregate histories if applicable"""
        assert (
            self.user_df.index.nlevels == 1
        ), "please set USER_ID as simple index on user_df"
        assert (
            "TEST_START_TIME" in self.user_df
        ), "please set TEST_START_TIME as a column in user_df"
        assert (
            self.user_df.index.is_unique
        ), "user_df should contain globally unique USER_ID"
        assert (
            self.item_df.index.is_unique
        ), "item_df should contain globally unique ITEM_ID"

        if self.event_df is not None:
            self.event_df = _sanitize_events(self.event_df, self.user_df, self.item_df)

        if "_hist_len" not in self.user_df and self.event_df is not None:
            self.user_df = aggregate_user_history(self.user_df, self._training_events)

        if "_hist_len" not in self.item_df and self.event_df is not None:
            self.item_df = self.item_df.assign(
                _hist_len=self._training_events.groupby("ITEM_ID")
                .size()
                .reindex(self.item_df.index, fill_value=0)
            )

    @cached_property
    @timed("inferring training events")
    def _training_events(self):
        """indexed by USER_ID and columned by ITEM_ID, TIMESTAMP, VALUE"""
        return (
            stable_join(self.user_df, self.event_df.set_index("USER_ID"))
            .query("TIMESTAMP < TEST_START_TIME")
            .assign(USER_ID=lambda x: x.index.get_level_values(0))[
                ["USER_ID", "ITEM_ID", "TIMESTAMP", "VALUE"]
            ]
        )


@dataclasses.dataclass
class Dataset(DatasetBase):
    """Dataset with holdout targets; allows multiple requests by the same user"""

    test_requests: pd.DataFrame = (
        None  # user requests indexed by USER_ID, TEST_START_TIME, etc. combinations
    )
    item_in_test: pd.DataFrame = None  # candidate items as a subset of item_df
    horizon: float = float(
        "inf"
    )  # construct target_csr; ignored if target_csr is provided
    target_csr: sps.spmatrix = None
    exclude_train: typing.Union[
        bool, list
    ] = True  # add negative priors for repeated items or item columns
    add_prior_to_form_reranking_task: float = (
        0  # add priors on target candidates to form a reranking task
    )
    prior_score: sps.spmatrix = None
    _skip_init: dataclasses.InitVar[bool] = False  # skip init during reindex
    test_update_history: bool = (
        True  # update relative history vs. frozen training history (False)
    )
    sample_with_prior: float = None  # renaming to add_prior_to_form_reranking_task

    @property
    def user_in_test(self):
        """alias with simple index"""
        user_in_test = self.test_requests.assign(
            TEST_START_TIME=lambda x: x.index.get_level_values(1)
        )
        while user_in_test.index.nlevels > 1:
            user_in_test = user_in_test.droplevel(-1)
        return user_in_test

    def __post_init__(self, _skip_init):
        """sanitize events and aggregate histories if applicable"""
        if _skip_init:
            return

        super().__post_init__()

        if self.sample_with_prior is not None:
            warnings.warn(
                "Consider renaming `sample_with_prior` to `add_prior_to_form_reranking_task` in future versions",
                PendingDeprecationWarning,
            )
            self.add_prior_to_form_reranking_task = self.sample_with_prior
        if self.test_requests is None:
            self.test_requests = self.user_df.set_index("TEST_START_TIME", append=True)
        if self.item_in_test is None:
            self.item_in_test = self.item_df

        assert (
            self.test_requests.index.nlevels >= 2
        ), "test_requests should be indexed by USER_ID, TEST_START_TIME, etc."
        assert (
            self.test_requests.index.is_unique
        ), "test_requests should contain unique index"
        assert (
            self.item_in_test.index.is_unique
        ), "item_in_test should contain unique ITEM_ID"
        assert self.horizon >= 0, "horizon should be nonnegative"

        if "_hist_len" not in self.test_requests:
            if self.test_update_history:
                self.test_requests = aggregate_user_history(
                    self.test_requests,
                    self._test_joined[
                        self._test_joined["TIMESTAMP"]
                        < self._test_joined.index.get_level_values(1)
                    ],
                )
            else:
                self.test_requests = stable_join(
                    self.test_requests,
                    self.user_df[
                        ["_hist_items", "_hist_ts", "_hist_values", "_hist_len"]
                    ],
                )

        if "_hist_len" not in self.item_in_test:
            self.item_in_test = self.item_in_test.assign(
                _hist_len=self.item_df["_hist_len"].reindex(
                    self.item_in_test.index, fill_value=0
                )
            )

        test_item_tokenize = {k: j for j, k in enumerate(self.item_in_test.index)}

        if self.target_csr is None:
            with timed("creating target_csr"):
                self.target_csr = indices2csr(
                    groupby_unexplode(
                        self._test_targets["ITEM_ID"].apply(test_item_tokenize.get),
                        self.test_requests.index,
                    ),
                    shape1=len(self.item_in_test),
                    data=groupby_unexplode(
                        self._test_targets["VALUE"], self.test_requests.index
                    ),
                )
                self.target_csr.eliminate_zeros()

        if self.prior_score is None and (
            self.exclude_train or self.add_prior_to_form_reranking_task
        ):
            self.prior_score = 0

            if self.exclude_train:
                # handles both outcomes of test_update_history
                _test_histories = self.test_requests[
                    self.test_requests["_hist_len"] > 0
                ]["_hist_items"].explode()
                with timed("creating prior_score"):
                    exclude_csr = indices2csr(
                        groupby_unexplode(
                            _test_histories[
                                _test_histories.isin(self.item_in_test.index)
                            ].apply(test_item_tokenize.get),
                            self.test_requests.index,
                        ),
                        shape1=len(self.item_in_test),
                    )
                    self.prior_score = self.prior_score + exclude_csr * -1e10

                if isinstance(self.exclude_train, list):
                    for _cat_name in self.exclude_train:
                        with timed(f"creating prior_score for {_cat_name}"):
                            _cat_codes = (
                                self.item_df[self.item_df[_cat_name].notnull()][
                                    _cat_name
                                ]
                                .astype("category")
                                .cat.codes
                            )
                            _test_requests_csr = indices2csr(
                                groupby_unexplode(
                                    _test_histories[
                                        _test_histories.isin(_cat_codes.index)
                                    ].apply(_cat_codes.to_dict().get),
                                    self.test_requests.index,
                                ),
                                shape1=_cat_codes.max() + 1,
                            )
                            _item_in_test_csr = indices2csr(
                                groupby_unexplode(
                                    stable_join(
                                        self.item_in_test[[]],
                                        _cat_codes.to_frame("code"),
                                    )["code"],
                                    self.item_in_test.index,
                                ),
                                shape1=_cat_codes.max() + 1,
                            )
                            exclude_csr = _test_requests_csr @ _item_in_test_csr.T
                            self.prior_score = self.prior_score + exclude_csr * -1e10

            if self.add_prior_to_form_reranking_task:
                with timed(
                    "creating reranking tasks by adding prior_score from target candidates"
                ):
                    cand_csr = indices2csr(
                        groupby_unexplode(
                            self._test_targets["ITEM_ID"].apply(test_item_tokenize.get),
                            self.test_requests.index,
                        ),
                        shape1=len(self.item_in_test),
                    )
                    self.prior_score = (
                        self.prior_score
                        + cand_csr * self.add_prior_to_form_reranking_task
                    )

        print(f"{repr(self)} created!")

    @cached_property
    @timed("joining testing events by multi-indexed requests")
    def _test_joined(self):
        return stable_join(self.test_requests, self.event_df.set_index("USER_ID"))

    @property
    def _test_targets(self):
        """based on _test_joined, horizon, item_in_test"""
        if self.horizon == 0:
            return self._test_joined[
                (
                    self._test_joined["TIMESTAMP"]
                    == self._test_joined.index.get_level_values(1)
                )
                & self._test_joined["ITEM_ID"].isin(self.item_in_test.index)
            ]
        return self._test_joined[
            (
                self._test_joined["TIMESTAMP"]
                >= self._test_joined.index.get_level_values(1)
            )
            & (
                self._test_joined["TIMESTAMP"]
                < self._test_joined.index.get_level_values(1) + self.horizon
            )
            & self._test_joined["ITEM_ID"].isin(self.item_in_test.index)
        ]

    @property
    def shape(self):
        return self.target_csr.shape

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        prior_nnz = self.prior_score.nnz if self.prior_score is not None else None
        return f"Dataset{self.shape} with {self.target_csr.nnz} target events and {prior_nnz} prior scores"

    @cached_property
    def auto_regressive(self):
        return DatasetBase(self.user_df, self.item_df, self._training_events)

    @property
    def user_ppl_baseline(self):
        return perplexity(self.test_requests["_hist_len"])

    @property
    def item_ppl_baseline(self):
        return perplexity(self.item_in_test["_hist_len"])

    def get_stats(self):
        avg_hist_span = (
            self.user_in_test[  # test users with finite history
                (self.user_in_test["TEST_START_TIME"] < np.inf)
                & (self.user_in_test["_hist_ts"].apply(len) > 0)
            ]
            .apply(lambda x: x["TEST_START_TIME"] - x["_hist_ts"][0], axis=1)
            .mean()
        )

        return {
            "user_df": {
                "# test user-time instances": len(self.test_requests),
                "# train users": len(self.user_df),
                "avg hist len": self.test_requests["_hist_len"].mean(),
                "avg hist span": avg_hist_span,
                "avg target len": self.target_csr.sum(axis=1).mean(),
            },
            "item_df": {
                "# test items": len(self.item_in_test),
                "# train items": len(self.item_df),
                "avg hist len": self.item_in_test["_hist_len"].mean(),
                "avg target len": self.target_csr.sum(axis=0).mean(),
            },
            "event_df": {
                "# train events": self.user_df["_hist_len"].sum(),
                "# test events": self.target_csr.sum(),
                "horizon": self.horizon,
                "user_ppl_baseline": self.user_ppl_baseline,
                "item_ppl_baseline": self.item_ppl_baseline,
            },
        }

    def print_stats(self, verbose=True):
        print(pd.DataFrame(self.get_stats()).T.stack().apply("{:.2f}".format))
        if verbose:
            print("+++sample test_requests")
            print(self.test_requests.sample().iloc[0])
            print("+++sample item_in_test")
            print(self.item_in_test.sample().iloc[0])

    def reindex(self, index, axis):
        if axis == 0:
            old_index = self.test_requests.index
            while old_index.nlevels > index.nlevels:
                old_index = old_index.droplevel(-1)
            while index.nlevels > old_index.nlevels:
                index = index.droplevel(-1)

            test_requests = self.test_requests.iloc[old_index.get_indexer(index)]
            item_in_test = self.item_in_test

        else:
            old_index = self.item_in_test.index
            test_requests = self.test_requests
            item_in_test = self.item_in_test.reindex(index, fill_value=0)  # hist_len

        target_csr = matrix_reindex(
            self.target_csr, old_index, index, axis, fill_value=0
        )

        if self.prior_score is None:
            prior_score = None
        elif isinstance(self.prior_score, LazyScoreBase):
            prior_score = self.prior_score.reindex(
                index, axis, fill_value=0, old_index=old_index
            )
        else:
            prior_score = matrix_reindex(
                self.prior_score, old_index, index, axis, fill_value=0
            )

        return Dataset(
            self.user_df,
            self.item_df,
            self.event_df,
            test_requests,
            item_in_test,
            self.horizon,
            target_csr,
            self.exclude_train,
            prior_score=prior_score,
            _skip_init=True,
        )

    def reindex_unbiased(
        self, min_user_len=1, min_item_len=1, allow_inf_test_start_time=False
    ):
        keep_row = (self.test_requests["_hist_len"] >= min_user_len).values & (
            (self.test_requests.index.get_level_values(1) < float("inf"))
            | allow_inf_test_start_time
        )
        keep_col = (self.item_in_test["_hist_len"] >= min_item_len).values
        return self.reindex(self.test_requests.index[keep_row], 0).reindex(
            self.item_in_test.index[keep_col], 1
        )

    def sample(self, *, axis, **kw):
        df = self.test_requests if axis == 0 else self.item_in_test
        return self.reindex(df.sample(**kw).index, axis)


def create_dataset_unbiased(
    event_df,
    user_df,
    item_df,
    horizon=float("inf"),
    min_user_len=1,
    min_item_len=1,
    allow_inf_test_start_time=False,
    **kw,
):
    """create unbiased Dataset with potentially repeated user_df"""
    all_users = user_df.groupby(level=0, sort=False).first()
    _request_id = (
        "_request_id"
        if "_request_id" in user_df
        else pd.RangeIndex(len(user_df), name="_request_id")
    )
    test_requests = user_df.set_index(["TEST_START_TIME", _request_id], append=True)
    D = Dataset(
        all_users, item_df, event_df, test_requests=test_requests, horizon=horizon, **kw
    )
    return D.reindex_unbiased(min_user_len, min_item_len, allow_inf_test_start_time)


def create_temporal_splits(
    event_df,
    user_df,
    item_df,
    TEST_START_TIME,
    horizon,
    validating_horizon,
    num_V_extra=0,
    **kw,
):
    testing_data = create_dataset_unbiased(
        event_df,
        user_df.assign(TEST_START_TIME=TEST_START_TIME),
        item_df,
        horizon,
        **kw,
    )
    testing_data.print_stats()
    validating_datasets = [
        create_dataset_unbiased(
            event_df,
            user_df.assign(
                TEST_START_TIME=TEST_START_TIME - validating_horizon * (k + 1)
            ),
            item_df,
            validating_horizon,
            **kw,
        )
        for k in range(num_V_extra + 1)
    ]
    return (testing_data, *validating_datasets)


def create_user_splits(
    event_df, user_df, item_df, test_start_rel, horizon, num_V_extra=0, **kw
):
    assert "_in_GroupA" in user_df, "requires _in_GroupA"
    test_start_abs = user_df.get("_Tmin", 0) + test_start_rel
    D = create_dataset_unbiased(
        event_df,
        user_df.assign(
            TEST_START_TIME=lambda x: np.where(
                x["_in_GroupA"], float("inf"), test_start_abs
            )
        ),
        item_df,
        horizon,
        **kw,
    )
    D.print_stats()
    V = create_dataset_unbiased(
        event_df,
        user_df.assign(
            TEST_START_TIME=lambda x: np.where(x["_in_GroupA"], test_start_abs, 0.0)
        ),
        item_df,
        horizon,
        **kw,
    )
    if num_V_extra:
        V0 = create_dataset_unbiased(
            event_df,
            user_df.assign(TEST_START_TIME=test_start_abs - horizon / 2),
            item_df,
            horizon / 2,
            **kw,
        )
        return D, V, V0
    else:
        return D, V
