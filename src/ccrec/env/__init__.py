import pandas as pd, numpy as np, scipy.sparse as sps, torch, dataclasses, warnings
import torch.nn.functional as F
from rime.util import indices2csr, auto_device
from rime.dataset.base import Dataset
from .base import (
    create_zero_shot,
    create_reranking_dataset,
    _expand_na_class,
    parse_response,
)
