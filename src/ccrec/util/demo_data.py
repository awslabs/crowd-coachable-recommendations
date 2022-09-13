import pandas as pd
import pytest, dataclasses, functools
from ccrec.models.vae_lightning import vae_main
from ccrec.models.bert_mt import bmt_main
from attrdict import AttrDict
from ccrec.env.i2i_env import Image, I2IImageEnv


class DemoData:
    """ invoke by DemoData().run_shap() """
    def __init__(
        self,
        data_root='data/demo',
        user_json='user_df.json',
        item_csv='item_df.csv',
        expl_response_json='expl_response.json',
        gnd_response_json='test_response.json',
        max_epochs=1,
        convert_time_unit=None,
        mock_image=False,
    ):
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.load_data()

    def load_data(self):
        self.user_df = pd.read_json(f'{self.data_root}/{self.user_json}', lines=True).set_index("USER_ID")
        self.item_df = pd.read_csv(f'{self.data_root}/{self.item_csv}').set_index("ITEM_ID")
        if self.mock_image:
            self.item_df = self.item_df.assign(landingImage='missing')

        self.gnd_response = pd.read_json(
            f'{self.data_root}/{self.gnd_response_json}', lines=True, convert_dates=False,
        ).set_index(['USER_ID'])
        self.expl_response = pd.read_json(
            f'{self.data_root}/{self.expl_response_json}', lines=True, convert_dates=False
        ).set_index(['USER_ID'])

    def run_vae_main(self):
        return vae_main(self.item_df, self.gnd_response, max_epochs=self.max_epochs, user_df=self.user_df)

    @functools.lru_cache()
    def run_bmt_main(self):
        return bmt_main(self.item_df, self.expl_response, self.gnd_response,
                        max_epochs=self.max_epochs, user_df=self.user_df, convert_time_unit=self.convert_time_unit)

    def run_shap(self, item_id=None, cand_ids=None):
        *_, bmt = self.run_bmt_main()
        if item_id is None:
            item_id = self.item_df.iloc[0].name
        if cand_ids is None:
            cand_ids = [item_id, item_id]

        Image.open(I2IImageEnv.image_format(
            self=AttrDict(item_df=self.item_df, explainer=bmt.to_explainer()),
            x={'_hist_items': [item_id], 'cand_items': cand_ids},
        )).show()
