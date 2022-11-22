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
from ccrec.env import create_reranking_dataset, create_zero_shot
from ccrec.models.item_tower import VAEItemTower


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
        max_length=32,
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


def vae_main(
    item_df,
    gnd_response=None,
    max_epochs=50,
    beta=0,
    train_df=None,
    user_df=None,
    model_cls_name="VAEPretrainedModel",
    masked=None,
    topk=1,
    expl_sample=0,
    reranking_prior=1e5,
    exclude_train=True,
    max_length=200,
    ckpt=None,
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
    if ckpt is not None:
        print(f"loading from {ckpt}")
        tower.load_state_dict(torch.load(ckpt)["state_dict"])
    else:
        train_dm = VAEData(
            train_df,
            tokenizer,
            64 * max(1, torch.cuda.device_count()),
            masked=masked,
            max_length=max_length,
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
        tower.model.ae_model.sample = expl_sample
        ds = ds.map(model.to_map_fn("text", "embedding"), batched=True, batch_size=64)
    item_emb = np.vstack(ds["embedding"])
    varCT = ItemKNN(item_df.assign(embedding=item_emb.tolist(), _hist_len=1))

    # evaluation
    gnd = create_reranking_dataset(
        user_df,
        item_df,
        gnd_response,
        reranking_prior=reranking_prior,
        exclude_train=exclude_train,
    )
    reranking_scores = varCT.transform(gnd) + gnd.prior_score
    metrics = rime.metrics.evaluate_item_rec(gnd.target_csr, reranking_scores, topk)

    return metrics, reranking_scores, tower  # tower.model.save_pretrained(save_dir)
    # assignments = rime.metrics._assign_topk(reranking_scores, topk).indices.reshape((-1, topk))