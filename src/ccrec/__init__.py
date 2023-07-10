import dataclasses, functools, warnings, itertools, typing, re
import pandas as pd, numpy as np
from typing import Any, Union
from pytorch_lightning.loggers import TensorBoardLogger
import os, traceback, sys


env_defaults = [
    (
        "CCREC_EMBEDDING_TYPE",
        "mean_layer_norm",
        ["cls", "mu", "mean", "mean_pooling", "mean_layer_norm"],
    ),
    ("CCREC_MAX_LENGTH", "256", None),
    ("CCREC_SIM_TYPE", "cos", ["cos", "dot"]),
    ("CCREC_TRAIN_MAIN", "bmt_main", ["bmt_main", "bbpr_main"]),
    ("CCREC_LEGACY_VAE_BUG", "0", ["0", "1"]),
    ("CCREC_DEBUG_SHAP", "0", ["0", "1"]),
    ("CCREC_TRAINING_PRECISION", "32", ["32", "bf16"]),
    ("CCREC_BBPR_INV_TEMPERATURE", "20", None),
    ("CCREC_DISPLAY_LENGTH", "250", None),
    #
    ("CCREC_INIT_ENV_DONE", "1", ["0", "1"]),
    ("CCREC_NON_BLOCKING", "1", ["0", "1"]),
]


def init_env_defaults():
    print("initializing env defaults")
    for name, default, options in env_defaults:
        if name in os.environ:
            prefix = "using  "
        else:
            prefix = "setting"
        val = os.environ.setdefault(name, default)
        if options is not None:
            assert val in options
        print(f"{prefix} {name}={val}; options: {options}")

    if (
        os.environ["CCREC_SIM_TYPE"] == "dot"
        and float(os.environ["CCREC_BBPR_INV_TEMPERATURE"]) >= 20
    ):
        warnings.warn("dot similarity works best with small inv_temperature")

    for k, v in os.environ.items():
        if k.startswith("CCREC") and k not in [x[0] for x in env_defaults]:
            warnings.warn(f"env {k}={v} not recognized by ccrec")


if not int(os.environ.get("CCREC_INIT_ENV_DONE", "0")):
    init_env_defaults()
