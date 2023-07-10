import os, warnings, dataclasses, collections, itertools, time, functools, typing
import pandas as pd, numpy as np, scipy.sparse as sps
from pytorch_lightning.loggers import TensorBoardLogger
from ccrec.util import merge_unique
from rime_lite.dataset import Dataset


def create_zero_shot(
    item_df, self_training=False, copy_item_id=True, create_user_filter=None, **kw
):
    """example create_user_filter=lambda x: x['ITEM_TYPE'] == 'query'"""
    if not copy_item_id:
        warnings.warn(
            "Changing default to use item_id as user_id in the future",
            DeprecationWarning,
        )

    if create_user_filter is None:
        user_item_id_list = item_df.index.values
    else:
        user_item_id_list = item_df[create_user_filter(item_df)].index.values

    user_df = pd.DataFrame(
        [
            {
                "USER_ID": item_id if copy_item_id else natural_id,
                "TEST_START_TIME": 1,
                "_hist_items": [item_id],
                "_hist_ts": [0],
            }
            for natural_id, item_id in enumerate(user_item_id_list)
        ]
    ).set_index("USER_ID")

    event_df = user_df["_hist_items"].explode().to_frame("ITEM_ID")
    event_df["TIMESTAMP"] = user_df["_hist_ts"].explode()
    event_df = event_df.reset_index()  # ITEM_ID, TIMESTAMP, USER_ID

    if self_training:  # legacy
        target_df = (
            user_df.set_index("TEST_START_TIME", append=True)["_hist_items"]
            .explode()
            .to_frame("ITEM_ID")
        )
        target_df["USER_ID"] = target_df.index.get_level_values(0)
        target_df["TIMESTAMP"] = target_df.index.get_level_values(-1)
        event_df = pd.concat([event_df, target_df], ignore_index=True)
    return Dataset(user_df, item_df, event_df, **kw)


def _sanitize_response(response):
    if "request_time" in response:
        response = response.set_index("request_time", append=True)

    request_time = response.index.get_level_values(-1)
    while request_time.max() > time.time():
        warnings.warn(
            "Sanitizing request_time by the unit of the second;"
            " please make sure that the unit stays consistent with all other parts of the code."
        )
        response = (
            response.reset_index(level=-1, drop=True)
            .assign(request_time=request_time / 1e3)
            .set_index("request_time", append=True)
        )
        request_time = response.index.get_level_values(-1)
    return response


def create_reranking_dataset(
    user_df,
    item_df,
    response=None,
    reranking_prior=1,  # use 1 for training and 1e5 for testing
    horizon=0.1,
    test_update_history=False,  # keep at default values
    **kw,
):
    """require user_df to be indexed by USER_ID and contains _hist_items and _hist_ts columns
    use reranking_prior=1 for training and reranking_prior=1e5 for testing
    keep horizon and test_update_hisotory at the default values.
    use exclude_train=['ITEM_TYPE'] to separate queries and passages; see test_information_retrieval.
    """
    past_event_df = user_df["_hist_items"].explode().to_frame("ITEM_ID")
    past_event_df["TIMESTAMP"] = user_df["_hist_ts"].explode().values

    past_event_df["USER_ID"] = past_event_df.index.get_level_values(0)
    past_event_df["VALUE"] = 1  # ITEM_ID, TIMESTAMP, USER_ID

    if response is None:
        event_df = past_event_df.reset_index(drop=True)
        test_requests = None
    else:
        response = _sanitize_response(response)
        event_df = pd.concat(
            [past_event_df, parse_response(response)], ignore_index=True
        )
        test_requests = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [
                    response.index.get_level_values(0),
                    response.index.get_level_values(-1),
                ]
            )
        )

    return Dataset(
        user_df,
        item_df,
        event_df,
        test_requests,
        sample_with_prior=reranking_prior,
        horizon=horizon,
        test_update_history=test_update_history,
        **kw,
    )


def create_retrieval_dataset(user_df, item_df, response=None, reranking_prior=0, **kw):
    return create_reranking_dataset(
        user_df, item_df, response=response, reranking_prior=reranking_prior, **kw
    )


def _sanitize_inputs(event_df, user_df, item_df, clear_future_events=None):
    assert user_df.index.is_unique, "require unique user ids"
    if event_df is None:
        user_non_empty = user_df[user_df["_hist_len"] > 0]
        event_df = (
            user_non_empty["_hist_items"]
            .explode()
            .to_frame("ITEM_ID")
            .assign(
                TIMESTAMP=user_non_empty["_hist_ts"].explode().values,
                VALUE=user_non_empty["_hist_values"].explode().values,
                USER_ID=lambda x: x.index.get_level_values(0),
            )[["USER_ID", "ITEM_ID", "TIMESTAMP", "VALUE"]]
            .reset_index(drop=True)
        )

    assert (
        event_df["TIMESTAMP"].max() < time.time()
    ), "require TIMESTAMP < current request_time"
    if "VALUE" not in event_df:
        event_df = event_df.assign(VALUE=1)

    event_old, event_df = (
        event_df,
        event_df[
            event_df["USER_ID"].isin(user_df.index)
            & event_df["ITEM_ID"].isin(item_df.index)
        ].copy(),
    )
    if len(event_old) > len(event_df):
        print(
            f"filtering events by known USER_ID and ITEM_ID. #events {len(event_old)} -> {len(event_df)}"
        )

    past_event_df = (
        event_df.join(
            user_df.groupby(level=0).first()[["TEST_START_TIME"]], on="USER_ID"
        )
        .query("TIMESTAMP < TEST_START_TIME")
        .drop("TEST_START_TIME", axis=1)
    )

    if len(past_event_df) < len(event_df):
        if clear_future_events is None:
            warnings.warn(
                f"future event detected, rate={1 - len(past_event_df) / len(event_df):.1%}"
            )
        elif clear_future_events:
            print(
                f"removing future events, rate={1 - len(past_event_df) / len(event_df):.1%}"
            )
            event_df = past_event_df

    return event_df


def parse_response(response, step_idx=None):
    response = _sanitize_response(response)
    new_events = response["cand_items"].explode().to_frame("ITEM_ID")
    new_events["USER_ID"] = new_events.index.get_level_values(0)
    new_events["TIMESTAMP"] = new_events.index.get_level_values(-1)
    new_events["VALUE"] = response["multi_label"].explode().values

    if "_group" in response:
        new_events["_group"] = response["_group"].explode().values
    if step_idx is not None:
        new_events["step_idx"] = step_idx  # for visualization only

    return new_events.reset_index(drop=True)
