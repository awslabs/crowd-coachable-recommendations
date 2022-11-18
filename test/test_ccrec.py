import pandas as pd, numpy as np, scipy.sparse as sps, os, pylab as pl
import torch, torch.nn.functional as F
import pytest, dataclasses, rime, ccrec, ccrec.models
from rime.dataset import create_dataset_unbiased
from rime.util import extract_user_item, auto_cast_lazy_score
from ccrec import agent, env, InteractiveExperiment

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_simple_dataset(sign=1):
    event_df = pd.DataFrame(
        {
            "USER_ID": np.repeat(np.arange(11), 2),  # 0, 0, 1, 1, ...
            "ITEM_ID": sign * np.arange(22) % 5 + 1,  # 1, 2, 3, 4, ...
            "TIMESTAMP": np.arange(22) % 2,  # 0, 1, 0, 1, ...
            "VALUE": 1,
        }
    )
    user_df, item_df = extract_user_item(event_df)
    user_df["TEST_START_TIME"] = 1
    D = create_dataset_unbiased(event_df, user_df, item_df, exclude_train=True)
    V = create_dataset_unbiased(
        event_df, pd.concat([user_df for _ in range(5)]), item_df, exclude_train=True
    )
    return (user_df, item_df, event_df), D, V


def create_ml_1m_interactive(**kw):
    """load from data/ml-1m/ratings.dat"""
    D, _, V = rime.dataset.prepare_ml_1m_data(exclude_train=True, **kw)
    user_df = D.user_in_test
    item_df = D.item_in_test
    event_df = D.user_in_test["_hist_items"].explode().to_frame("ITEM_ID").reset_index()
    event_df["TIMESTAMP"] = D.user_in_test["_hist_ts"].explode().values
    event_df["VALUE"] = 1
    return (user_df, item_df, event_df), D, V.reindex(item_df.index, axis=1)


def graph_conv_factory(D, **kw):
    return rime.models.GraphConv(
        D,
        sample_with_prior=True,
        sample_with_posterior=0,
        user_rec=False,
        user_conv_model="plain_average",
        truncated_input_steps=1,
        training_prior_fcn=lambda x: (x + 0.1 / x.shape[1]).clip(0, None).log(),
        **kw,
    )


def empirical_average_factory(D, **kw):
    return ccrec.models.EmpiricalAverageModel(D.user_df.index, D.item_df.index, **kw)


@pytest.mark.parametrize(
    "model_factory",
    [
        graph_conv_factory,
    ],
)
@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_oracle(model_factory):
    (user_df, item_df, event_df), D, V = create_simple_dataset()
    testing_env = env.HoldoutEnv(
        user_df,
        item_df,
        event_df,
        clear_future_events=True,
        oracle=D,
        sample_size=1,
        recording=False,
    )

    oracle_model = model_factory(D).fit(V)
    oracle_agent = agent.GreedyAgent(oracle_model, training=False)
    baseline_model = rime.util.LazyScoreModel(
        user_df.index, item_df.index, tie_breaker=0.01
    )
    baseline_agent = agent.GreedyAgent(baseline_model, training=False)

    reward_by_policy = testing_env.step(oracle_agent, baseline_agent)
    print(reward_by_policy)
    assert (
        reward_by_policy["0"] > reward_by_policy["1"]
    ), f"insufficient test power {reward_by_policy}"
    assert reward_by_policy["0"] > 0.7, f"insufficient accuracy {reward_by_policy}"


@pytest.mark.flaky(max_runs=2, min_passes=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
@pytest.mark.parametrize(
    "working_model_cls",
    [
        empirical_average_factory,  # graph_conv_factory
    ],
)
def test_simplest(
    working_model_cls,
    n_steps=20,
    test_every=1,
    epsilon=0.5,
    unique_titles=["Apple", "Banana", "Orange", "Strawberry", "Grape"],
):

    nunique_titles = len(unique_titles)
    item_df = pd.Series(unique_titles + unique_titles).to_frame("TITLE")
    event_df = pd.DataFrame(
        [(i, i, 0, 1) for i in item_df.index],
        columns=["USER_ID", "ITEM_ID", "TIMESTAMP", "VALUE"],
    )
    user_df = event_df.set_index("USER_ID").assign(TEST_START_TIME=1)[
        ["TEST_START_TIME"]
    ]
    D = create_dataset_unbiased(event_df, user_df, item_df)
    D.print_stats()

    title_one_hot = np.vstack([np.eye(nunique_titles), np.eye(nunique_titles)])
    oracle_model = rime.util.LazyScoreModel(
        user_df.index,
        item_df.index,
        auto_cast_lazy_score(title_one_hot) @ title_one_hot.T * 10,
        tie_breaker=0.01,
    )
    working_model = working_model_cls(D).fit()
    baseline_model = rime.util.LazyScoreModel(
        user_df.index, item_df.index, tie_breaker=0.01
    )

    iexp = InteractiveExperiment(
        user_df,
        item_df,
        event_df,
        {
            "oracle": agent.Agent(oracle_model),
            "prefix": "simplest-train",
            "sample_with_prior": -10,
        },
        {
            "oracle": agent.Agent(oracle_model),
            "prefix": "simplest-test",
        },
        [working_model],
        [oracle_model, baseline_model],
        epsilon=epsilon,
    )
    iexp.run(n_steps, test_every)

    reward_by_policy = (
        pd.DataFrame(iexp.testing_env._reward_by_policy)
        .iloc[n_steps // 2 :]
        .mean(axis=0)
    )
    print(reward_by_policy)
    pl.imshow(
        iexp.training_env.event_df.query("TIMESTAMP > 1")
        .groupby(["USER_ID", "ITEM_ID"])["VALUE"]
        .mean()
        .unstack()
    )
    pl.colorbar()
    assert reward_by_policy["0"] > 0.3, f"insufficient accuracy {reward_by_policy}"


@pytest.mark.parametrize(
    "data_factory, working_model_cls, epsilon",
    [
        (create_simple_dataset, graph_conv_factory, 0.1),
        (create_simple_dataset, graph_conv_factory, "dual"),
        pytest.param(
            create_ml_1m_interactive,
            graph_conv_factory,
            0.1,
            marks=[
                pytest.mark.skipif(
                    not os.path.exists("data/ml-1m/ratings.dat"),
                    reason="data not available",
                ),
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="cuda not available"
                ),
            ],
        ),
    ],
)
def test_exploration(data_factory, working_model_cls, epsilon):
    (user_df, item_df, event_df), D, V = data_factory()
    oracle_model = rime.models.Transformer(item_df).fit(V.auto_regressive)
    oracle_model._truncated_input_steps = 1
    oracle_agent = agent.Agent(oracle_model)

    working_model = working_model_cls(D).fit()
    baseline_model = rime.util.LazyScoreModel(
        user_df.index, item_df.index, tie_breaker=0.01
    )

    iexp = InteractiveExperiment(
        user_df,
        item_df,
        event_df,
        {"oracle": oracle_agent, "sample_with_prior": -10},
        {"oracle": D},
        [working_model],
        [oracle_model, baseline_model],
        epsilon=epsilon,
    )
    iexp.run()
