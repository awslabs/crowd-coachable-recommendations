import pandas as pd, numpy as np
import pytest, rime, ccrec, ccrec.models
from ccrec import InteractiveExperiment, env


@pytest.mark.skip(reason="manually test the validity of online oracles")
@pytest.mark.parametrize("n_steps", [0, 10])
def test_run(
    n_steps,
    unique_titles=["Apple", "Banana", "Orange", "Strawberry", "Grape"],
    test_every=2,
):
    nunique_titles = len(unique_titles)
    item_df = pd.Series(unique_titles + unique_titles).to_frame("TITLE")
    event_df = (
        pd.Series(item_df.index.values)
        .to_frame("ITEM_ID")
        .assign(USER_ID=np.arange(len(item_df)), TIMESTAMP=0)
    )
    user_df = pd.Series(np.ones(len(item_df))).to_frame("TEST_START_TIME")

    D = rime.dataset.Dataset(user_df, item_df, event_df)
    D.print_stats()

    title_one_hot = np.vstack([np.eye(nunique_titles), np.eye(nunique_titles)])
    oracle_model = rime.util.LazyScoreModel(
        user_df.index,
        item_df.index,
        rime.util.LazyDenseMatrix(title_one_hot) @ title_one_hot.T * 10,
        tie_breaker=0.01,
    )
    # working_model = graph_conv_factory(D).fit()
    working_model = ccrec.models.EmpiricalAverageModel(
        D.user_df.index, D.item_df.index
    ).fit()
    baseline_model = rime.util.LazyScoreModel(
        user_df.index, item_df.index, tie_breaker=0.01
    )

    iexp = InteractiveExperiment(
        D.user_df,
        D.item_df,
        D.event_df,
        {"oracle": env.I2IConfig(), "prefix": "simplest-train", "sample_size": 6},
        {"oracle": env.I2IConfig(), "prefix": "simplest-test"},
        [working_model],
        [oracle_model, baseline_model],
        epsilon=0.5,
    )

    iexp.run(n_steps, test_every)
    print(
        pd.DataFrame(iexp.testing_env._reward_by_policy).mean(axis=0)
    )  # expect values=[0, 1, 0]
    print(
        iexp.training_env.event_df.groupby(["USER_ID", "ITEM_ID"])["VALUE"]
        .size()
        .unstack()
    )
    print(
        iexp.training_env.event_df.groupby(["USER_ID", "ITEM_ID"])["VALUE"]
        .mean()
        .round(1)
        .unstack()
    )

    return iexp
