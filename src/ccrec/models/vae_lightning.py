import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from pytorch_lightning import LightningDataModule, Trainer
from ccrec.models import vae_models
from transformers import DefaultDataCollator, DataCollatorForLanguageModeling
import rime
from rime.util import _LitValidated
from ccrec.util import _device_mode_context
from rime.models.zero_shot import ItemKNN
from ccrec.env import create_reranking_dataset
from ccrec.models.item_tower import VAEItemTower
from rime.util import empty_cache_on_exit, auto_cast_lazy_score
import os


class LitVAEModel(_LitValidated):
    def __init__(
        self,
        beta=0,
        model_name="distilbert-base-uncased",
        model_cls_name="VAEPretrainedModel",
        tokenizer=None,
        tokenizer_kw={},
    ):
        super().__init__()
        self.save_hyperparameters("beta", "model_name")
        model = getattr(vae_models, model_cls_name).from_pretrained(model_name)
        tokenizer = tokenizer
        if hasattr(model, "set_beta"):
            model.set_beta(beta)
        self.model = VAEItemTower(model, tokenizer=tokenizer, tokenizer_kw=tokenizer_kw)

    def setup(self, stage):
        if stage == "fit":
            print(self.logger.log_dir)

    def training_and_validation_step(self, batch, batch_idx):
        return self.model(**batch, output_step="dict")[0].mean()

    def forward(self, batch):
        return self.model(**batch, output_step="embedding")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5, weight_decay=0.01)

    def to_explainer(self, **kw):
        return self.model.to_explainer(**kw)


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
        max_length=300,
        **kw
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
            **kw
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


class _DataModule:
    def __init__(self, item_df):
        self.item_df = item_df
        item_index = item_df.index
        self.item_to_ptr = {k: ptr for ptr, k in enumerate(item_index)}

    def setup_for_activate_learning(self, rime_dataset):
        self._D = rime_dataset
        self.i_to_ptr = [
            self.item_to_ptr[hist[0]] for hist in self._D.user_in_test["_hist_items"]
        ]
        self.j_to_ptr = [self.item_to_ptr[item] for item in self._D.item_in_test.index]
        self.training_data = {"i_to_ptr": self.i_to_ptr, "j_to_ptr": self.j_to_ptr}


class VAEModel:
    def __init__(
        self,
        item_df,
        batch_size=10,
        model_name="distilbert-base-uncased",
        max_length=300,
        max_epochs=10,
        beta=2e-3,
        model_cls_name="VAEPretrainedModel",
        **kw
    ):
        self.item_df = item_df
        self.item_titles = item_df["TITLE"]
        self.max_length = max_length
        self.batch_size = batch_size * max(1, torch.cuda.device_count())
        self.max_epochs = max_epochs
        self.beta = beta
        self.model_cls_name = model_cls_name
        self.model_name = model_name
        self.dm = _DataModule(item_df)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tower = LitVAEModel(
            self.beta, tokenizer=self.tokenizer, model_cls_name=self.model_cls_name
        )

    def _transform_item_corpus(self, item_tower, output_step):
        with _device_mode_context(item_tower) as item_tower:
            ds = Dataset.from_pandas(self.item_titles.to_frame("text"))
            out = ds.map(
                item_tower.to_map_fn("text", output_step), batched=True, batch_size=64
            )[output_step]
            return np.vstack(out)

    @empty_cache_on_exit
    def fit(self, train_data=None):
        if train_data is None:
            train_data = self.item_df
        train_dm = VAEData(
            train_data,
            self.tokenizer,
            self.batch_size,
            masked=None,
            max_length=self.max_length,
        )
        trainer = Trainer(
            max_epochs=self.max_epochs,
            gpus=torch.cuda.device_count(),
            strategy="dp" if torch.cuda.device_count() else None,
            log_every_n_steps=1,
            precision="bf16",
        )
        trainer.fit(self.tower, datamodule=train_dm)
        self.tower._load_best_checkpoint("best")

        if not os.path.exists(self.tower._checkpoint.dirpath):  # add manual checkpoint
            os.makedirs(self.tower._checkpoint.dirpath)
            torch.save(
                self.tower.state_dict(),
                self.tower._checkpoint.dirpath + "/state-dict.pth",
            )

    @empty_cache_on_exit
    @torch.no_grad()
    def transform(self, D):
        self.dm.setup_for_activate_learning(D)
        # all_emb = self._transform_item_corpus(self.tower, 'embedding')
        all_emb = torch.rand(9, 768)
        user_final = all_emb[self.dm.i_to_ptr]
        item_final = all_emb[self.dm.j_to_ptr]
        return auto_cast_lazy_score(user_final) @ item_final.T

    def to_explainer(self, **kw):
        return self.model.to_explainer(**kw)


def vae_main(
    item_df,
    gnd_response,
    max_epochs=50,
    beta=0,
    train_df=None,
    user_df=None,
    model_cls_name="VAEPretrainedModel",
    masked=None,
):
    """
    item_df = get_item_df()[0]  # indexed by ITEM_ID
    gnd_response = pd.read_json(
        'prime-pantry-i2i-online-baseline4-response.json', lines=True, convert_dates=False
    ).set_index('level_0')  # indexed by USER_ID
    """
    if train_df is None:
        train_df = item_df
    if masked is None:
        masked = model_cls_name != "VAEPretrainedModel"

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tower = LitVAEModel(beta, tokenizer=tokenizer, model_cls_name=model_cls_name)
    train_dm = VAEData(
        train_df, tokenizer, 64 * max(1, torch.cuda.device_count()), masked=masked
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=torch.cuda.device_count(),
        strategy="dp" if torch.cuda.device_count() else None,
        log_every_n_steps=1,
    )
    trainer.fit(tower, datamodule=train_dm)

    ds = Dataset.from_pandas(item_df.rename({"TITLE": "text"}, axis=1))
    with _device_mode_context(tower.model) as model, torch.no_grad():
        ds = ds.map(model.to_map_fn("text", "embedding"), batched=True, batch_size=64)
    item_emb = np.vstack(ds["embedding"])
    varCT = ItemKNN(item_df.assign(embedding=item_emb.tolist(), _hist_len=1))

    # evaluation
    gnd = create_reranking_dataset(user_df, item_df, gnd_response, reranking_prior=1e5)
    reranking_scores = varCT.transform(gnd) + gnd.prior_score
    metrics = rime.metrics.evaluate_item_rec(gnd.target_csr, reranking_scores, 1)

    return metrics, reranking_scores, tower  # tower.model.save_pretrained(save_dir)
