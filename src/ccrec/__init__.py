import dataclasses, functools, warnings, itertools, typing, re
import pandas as pd, numpy as np
from typing import Any, Union
from pytorch_lightning.loggers import TensorBoardLogger
import rime
from .env import I2IConfig, auto_env_select
from .agent import (
    DualAgent,
    EpsAgent,
    GreedyAgent,
    Agent,
    BoltzmannAgent,
    LatentNoiseAgent,
    VAEAgent,
)


def _sanity_check(self):
    assert self.user_df.index.is_unique, "please index user_df with unique USER_ID"
    assert self.item_df.index.is_unique, "please index item_df with unique ITEM_ID"
    if self.event_df is not None:
        assert set(self.event_df.columns) >= {
            "USER_ID",
            "ITEM_ID",
            "TIMESTAMP",
            "VALUE",
        }, "please verify the columns in event_df"
        if len(self.event_df) == 0:
            warnings.warn(
                "we cannot handle cold-start users, please put something as user history events"
            )


@dataclasses.dataclass
class InteractiveExperiment:
    """we usually onboard InteractiveRecommendation in 3 steps:
    1. oracle-train, oracle-test
    2. oracle-train, ground-truth-test (I2IConfig)
    3. ground-truth-train, ground-truth-test (I2IConfig)
    """

    user_df: pd.DataFrame
    item_df: pd.DataFrame
    event_df: pd.DataFrame  # use D_rerank.event_df to allow sample_with_prior to work
    training_env_kw: dict  # provide oracle here; in case of conficts, last value wins
    testing_env_kw: dict  # provide oracle here; in case of conficts, last value wins
    working_models: list  # race multiple models, e.g., bbpr and random with sample_size=(3, 1)
    baseline_models: typing.Union[list, int]
    epsilon: typing.Union[float, str] = 0
    train_user_df: pd.DataFrame = None
    test_user_df: pd.DataFrame = None

    @property
    def working_model(self):
        warnings.warn("please call working_models[0] instead", DeprecationWarning)
        return self.working_models[0]

    def __post_init__(self):
        """set up training/testing env and training/testing/baseline agent"""
        _sanity_check(self)
        if not isinstance(self.working_models, list):
            self.working_models = [self.working_models]
        working_names = (
            f"{self.working_models[0].__class__.__name__}-x{len(self.working_models)}"
        )

        self.training_env = auto_env_select(
            self.train_user_df if self.train_user_df is not None else self.user_df,
            self.item_df,
            self.event_df,
            **{
                "prefix": f"train-{working_names}",
                "sample_size": 4,
                "clear_future_events": True,  # clear state for training exploration
                **self.training_env_kw,
            },
        )
        self.testing_env = auto_env_select(
            self.test_user_df if self.test_user_df is not None else self.user_df,
            self.item_df,
            self.event_df,
            **{
                "prefix": f"test-{working_names}",
                "sample_size": 1,
                "recording": False,  # always test in the same state
                **self.testing_env_kw,
            },
        )

        self.training_agents = [
            self._create_training_agent(m, e)
            for m, e in zip(
                self.working_models,
                self.epsilon
                if np.size(self.epsilon) > 1
                else [self.epsilon] * len(self.working_models),
            )
        ]
        if isinstance(self.baseline_models, list):
            self.testing_agents = [
                GreedyAgent(m, training=False) for m in self.working_models
            ] + [GreedyAgent(m, training=False) for m in self.baseline_models]
        else:  # int
            self.testing_agents = [
                GreedyAgent(m, training=False)
                for m in self.working_models[: self.baseline_models]
            ]

        self._logger = TensorBoardLogger(
            "logs", f"{working_names}-{len(self.user_df)}-{len(self.item_df)}"
        )
        self._logger.log_hyperparams(
            {
                "training_env_log_dir": self.training_env._logger.log_dir,
                "testing_env_log_dir": self.testing_env._logger.log_dir,
                **{
                    repr(m): m._logger.log_dir
                    for m in self.working_models
                    if hasattr(m, "_logger")
                },
            }
        )
        print(f"InteractiveExperiment logs at {self._logger.log_dir}")

    def _create_training_agent(self, model, epsilon):
        if epsilon == "dual":
            alpha_ub = np.sum(self.training_env.sample_size) / len(self.item_df)
            return DualAgent(
                model,
                training=True,
                alpha_ub=alpha_ub,
                beta_lb=alpha_ub / 2,
                beta_ub=alpha_ub * 2,
            )
        elif isinstance(epsilon, str) and epsilon.startswith("boltzmann"):
            target_ppl = float(re.search(r"boltzmann(\d+)", epsilon).group(1))
            return BoltzmannAgent(model, training=True, target_ppl=target_ppl)
        elif isinstance(epsilon, str) and epsilon.startswith("latent"):
            std = float(epsilon[len("latent") :])  # e.g., latent2e-2
            return LatentNoiseAgent(
                model, std=std, training=False
            )  # deprecate training mode
        elif epsilon == "vae":
            return VAEAgent(
                model, item_df=self.item_df, training=False
            )  # deprecate training mode
        else:
            return EpsAgent(model, training=True, epsilon=epsilon)

    def run(self, n_steps=2, test_every=1, test_before_train=True):
        if test_before_train:
            reward_by_policy = self.testing_env.step(*self.testing_agents)
            print(f"test-{self.testing_env._last_step_idx()}", reward_by_policy)

        for i in range(n_steps):
            reward_by_policy = self.training_env.step(*self.training_agents)
            print(f"train-{self.training_env._last_step_idx()}", reward_by_policy)
            if i == 0:
                print(self.training_env.event_df)
            all_data = self.training_env._create_training_dataset()
            if hasattr(self.working_models[0], "fit"):
                self.working_models[0].fit(all_data)
            for m in self.working_models[1:]:
                if hasattr(m, "fit") and m is not self.working_models[0]:
                    m.fit(all_data)

            if test_every is not None and (i + 1) % test_every == 0:
                reward_by_policy = self.testing_env.step(*self.testing_agents)
                print(f"test-{self.testing_env._last_step_idx()}", reward_by_policy)
