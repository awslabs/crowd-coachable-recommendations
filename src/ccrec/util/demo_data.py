import pandas as pd, numpy as np, torch
import pytest, dataclasses, functools
import scipy.sparse as sps
from rime.util import auto_cast_lazy_score
from ccrec.models.vae_lightning import vae_main
from ccrec.models.bert_mt import bmt_main
from attrdict import AttrDict
from ccrec.util import auto_device
from ccrec.env.i2i_env import Image, I2IImageEnv


def _pandas_read(file_name, **kw):
    if 'json' in file_name.split('.')[-2:]:  # json or json.gz
        return pd.read_json(file_name, lines=True, **kw)
    elif 'csv' in file_name.split('.')[-2:]:  # csv or csv.gz
        return pd.read_csv(file_name, **kw)
    else:
        raise NotImplementedError(f'unrecognized file_name {file_name}')


class DemoData:
    """ invoke by DemoData().run_shap()
    to use prime-pantry data:
        gnd_response = pd.read_json('data/amazon_review_prime_pantry/prime_pantry_test_response.json.gz',
                           lines=True, convert_dates=False).set_index('level_0')
        item_df, _ = ccrec.util.amazon_review_prime_pantry.get_item_df('data/amazon_review_prime_pantry')
        DemoData(None, None, item_df, None, gnd_response, max_epochs=50).run_shap(model_main='run_vae_main')
    """
    def __init__(
        self,
        data_root='data/demo',
        user_df='user_df.json',
        item_df='item_df.csv',
        expl_response='expl_response.json',
        gnd_response='test_response.json',
        max_epochs=1,
        convert_time_unit=None,
        cache_embedding=False,
    ):
        for k, v in locals().items():
            if k not in ['self', 'cache_embedding']:
                setattr(self, k, v)
        self.__post_init__()
        if cache_embedding:
            self.create_embedding = functools.lru_cache()(self.create_embedding)

    def __post_init__(self):
        if self.user_df is not None and not isinstance(self.user_df, pd.DataFrame):
            self.user_df = _pandas_read(f'{self.data_root}/{self.user_df}').set_index("USER_ID")

        if not isinstance(self.item_df, pd.DataFrame):
            self.item_df = _pandas_read(f'{self.data_root}/{self.item_df}').set_index("ITEM_ID")

        if not isinstance(self.gnd_response, pd.DataFrame):
            self.gnd_response = _pandas_read(
                f'{self.data_root}/{self.gnd_response}', convert_dates=False,
            ).set_index(['USER_ID'])

        if self.expl_response is not None and not isinstance(self.expl_response, pd.DataFrame):
            self.expl_response = _pandas_read(
                f'{self.data_root}/{self.expl_response}', convert_dates=False
            ).set_index(['USER_ID'])

    @functools.lru_cache()
    def run_vae_main(self):
        return vae_main(self.item_df, self.gnd_response, max_epochs=self.max_epochs, user_df=self.user_df)

    @functools.lru_cache()
    def run_bmt_main(self):
        return bmt_main(self.item_df, self.expl_response, self.gnd_response,
                        max_epochs=self.max_epochs, user_df=self.user_df, convert_time_unit=self.convert_time_unit)

    @torch.no_grad()
    def create_embedding(self, explainer):
        from ccrec.models.vae_lightning import VAEData
        if 'embedding' in self.item_df:
            return np.vstack(self.item_df['embedding'])
        dm = VAEData(self.item_df, explainer.tokenizer)
        dm.setup('predict')
        item_emb = [
            explainer.item_tower(**{k: v.to(explainer.item_tower.device) for k, v in batch.items()}).cpu().numpy()
            for batch in dm.predict_dataloader()
        ]
        return np.vstack(item_emb)

    def retrieve_similar(self, item_id, explainer, prior_score=None, topk=4):
        """ output batch_size * topk from a list of item_ids, a model explainer, and a prior_score matrix """
        item_id = np.ravel(item_id)  # convert to a list if not already
        batch_size = len(item_id)

        item_emb = self.create_embedding(explainer)  # vocab_size * nhid
        query_ptr = self.item_df.index.get_indexer(item_id)
        query_emb = item_emb[query_ptr]  # batch_size * nhid
        item_scores = auto_cast_lazy_score(query_emb) @ item_emb.T  # batch_size * vocab_size

        if prior_score is None:  # exclude query item in the retrieved set
            prior_score = sps.csr_matrix(
                (np.ones(batch_size) * -1e10, query_ptr, np.arange(batch_size + 1)),
                shape=(batch_size, len(self.item_df)))
        item_scores = item_scores + prior_score  # batch_size * vocab_size

        cand_ptr = item_scores.as_tensor(auto_device()).topk(topk).indices.cpu().numpy()
        return self.item_df.index[cand_ptr]  # bach_size * topk

    def run_shap(self, item_id=None, cand_items=None, model_main='run_bmt_main'):
        *_, bmt = getattr(self, model_main)()
        explainer = bmt.to_explainer()

        if item_id is None:
            item_id = self.item_df.index.values[0]
        if cand_items is None:
            cand_items = self.retrieve_similar(item_id, explainer).ravel()  # a list of topk items

        image = I2IImageEnv.image_format(
            self=AttrDict(item_df=self.item_df, explainer=explainer),
            x={'_hist_items': [item_id], 'cand_items': cand_items},
        )
        Image.open(image).show()
        return image, bmt.to_explainer()
