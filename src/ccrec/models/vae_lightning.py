import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from transformers import DefaultDataCollator, DataCollatorForLanguageModeling
import os


class VAEData(LightningDataModule):
    def __init__(
        self,
        item_df,
        tokenizer,
        batch_size=64,
        do_validation=True,
        masked=False,
        truncation=True,
        padding="max_length",
        max_length=int(os.environ.get("CCREC_MAX_LENGTH", 32)),
        **kw,
    ):
        super().__init__()
        self._item_df = item_df
        self._batch_size = batch_size
        self._do_validation = do_validation
        self._tokenizer_fn = lambda x: tokenizer(
            x["TITLE"],
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            **kw,
        )
        self._collate_fn = (
            DataCollatorForLanguageModeling(tokenizer)
            if masked
            else DefaultDataCollator()
        )
        print("masked", masked)
        self._num_batches = len(item_df) / self._batch_size

    def setup(self, stage):
        if stage == "fit":
            _to_dataset = lambda x: Dataset.from_pandas(x.reset_index()[["TITLE"]])
            if self._do_validation and len(self._item_df) >= 5:
                shuffled = self._item_df.sample(frac=1, random_state=1)
                self._ds = DatasetDict(
                    train=_to_dataset(shuffled.iloc[: len(self._item_df) * 4 // 5]),
                    valid=_to_dataset(shuffled.iloc[len(self._item_df) * 4 // 5 :]),
                )
            else:
                self._ds = DatasetDict(train=_to_dataset(self._item_df))
            self._ds = self._ds.map(self._tokenizer_fn, remove_columns=["TITLE"])

    def train_dataloader(self):
        return DataLoader(
            self._ds["train"],
            batch_size=self._batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        if "valid" in self._ds:
            return DataLoader(
                self._ds["valid"],
                batch_size=self._batch_size,
                collate_fn=self._collate_fn,
            )
