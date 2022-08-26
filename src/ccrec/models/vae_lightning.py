import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from pytorch_lightning import LightningDataModule, Trainer
from irec.models.vae_models import MaskedPretrainedModel, VAEPretrainedModel
from transformers import DefaultDataCollator, DataCollatorForLanguageModeling
import rime
from rime.util import _LitValidated
from rime.models.zero_shot import ItemKNN
from irec.env import create_zero_shot, parse_response


class VAETower(_LitValidated):
    def __init__(self, beta=0, model_name='distilbert-base-uncased'):
        super().__init__()
        self.save_hyperparameters("beta", "model_name")
        self.model = VAEPretrainedModel.from_pretrained(model_name)
        if hasattr(self.model, 'set_beta'):
            self.model.set_beta(beta)

    def setup(self, stage):
        if stage == 'fit':
            print(self.logger.log_dir)

    def training_and_validation_step(self, batch, batch_idx):
        return self.model(**batch, return_dict=True)[0]

    def forward(self, batch):
        return self.model(**batch, return_embedding=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=2e-5, weight_decay=0.01)


class VAEData(LightningDataModule):
    def __init__(self, item_df, tokenizer, batch_size=64, do_validation=True,
                 truncation=True, padding='max_length', max_length=32, **kw):
        super().__init__()
        self._item_df = item_df
        self._batch_size = batch_size
        self._do_validation = do_validation
        self._tokenizer_fn = lambda x: tokenizer(
            x['TITLE'], truncation=truncation, padding=padding, max_length=max_length, **kw)
        self._collate_fn = DefaultDataCollator()
        self._num_batches = len(item_df) / self._batch_size

    def setup(self, stage):
        _to_dataset = lambda x: Dataset.from_pandas(x.reset_index()[['TITLE']])
        if stage == 'fit':
            if self._do_validation and len(self._item_df) >= 5:
                shuffled = self._item_df.sample(frac=1, random_state=1)
                self._ds = DatasetDict(
                    train=_to_dataset(shuffled.iloc[:len(self._item_df) * 4 // 5]),
                    valid=_to_dataset(shuffled.iloc[len(self._item_df) * 4 // 5:]))
            else:
                self._ds = DatasetDict(train=_to_dataset(self._item_df))
        else:  # predict
            self._ds = DatasetDict(predict=_to_dataset(self._item_df))
        self._ds = self._ds.map(self._tokenizer_fn, remove_columns=['TITLE'])

    def _create_dataloader(self, split, shuffle=False):
        if split in self._ds:
            return DataLoader(self._ds[split], batch_size=self._batch_size,
                              collate_fn=self._collate_fn, shuffle=shuffle)

    def train_dataloader(self):
        return self._create_dataloader('train', True)

    def val_dataloader(self):
        return self._create_dataloader('valid')

    def predict_dataloader(self):
        return self._create_dataloader('predict')


def vae_main(item_df, gnd_response, max_epochs=50, beta=0, train_df=None):
    """
    item_df = get_item_df()[0]  # indexed by ITEM_ID
    gnd_response = pd.read_json(
        'prime-pantry-i2i-online-baseline4-response.json', lines=True, convert_dates=False
    ).set_index('level_0')  # indexed by USER_ID
    """
    if train_df is None:
        train_df = item_df

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tower = VAETower(beta)
    train_dm = VAEData(train_df, tokenizer, 64 * max(1, torch.cuda.device_count()))
    trainer = Trainer(max_epochs=max_epochs, gpus=torch.cuda.device_count(),
                      strategy='dp' if torch.cuda.device_count() else None,
                      log_every_n_steps=1)
    trainer.fit(tower, datamodule=train_dm)

    test_dm = VAEData(item_df, tokenizer, 64 * max(1, torch.cuda.device_count()))
    item_emb = trainer.predict(tower, datamodule=test_dm)
    item_emb = torch.cat(item_emb)
    varCT = ItemKNN(item_df.assign(embedding=item_emb.tolist(), _hist_len=1))

    # evaluation
    zero_shot = create_zero_shot(item_df)
    gnd_events = parse_response(gnd_response)
    gnd = rime.dataset.Dataset(
        zero_shot.user_df, item_df, pd.concat([zero_shot.event_df, gnd_events]),
        sample_with_prior=1e5)
    gnd._k1 = 1
    reranking_task = rime.Experiment(gnd)
    reranking_task.run({'varCT': varCT})
    recall = reranking_task.item_rec['varCT']['recall']  # expect 0.363

    return recall, tower  # tower.model.save_pretrained(save_dir)
