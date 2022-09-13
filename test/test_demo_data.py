import pandas as pd
import pytest, dataclasses, functools
from ccrec.models.vae_lightning import vae_main
from ccrec.models.bert_mt import bmt_main
from attrdict import AttrDict
from ccrec.env.i2i_env import Image, I2IImageEnv


@dataclasses.dataclass(unsafe_hash=True)
class TestDemoData:
    """ invoke by TestDemoData().test_shap() """
    data_root = 'data/demo'
    user_json = 'user_df.json'
    item_csv = 'item_df.csv'
    expl_response_json = 'expl_response.json'
    gnd_response_json = 'test_response.json'
    max_epochs = 1
    convert_time_unit = None

    def __post_init__(self):
        self.user_df = pd.read_json(f'{self.data_root}/{self.user_json}', lines=True).set_index("USER_ID")
        self.item_df = pd.read_csv(f'{self.data_root}/{self.item_csv}').set_index("ITEM_ID")
        self.gnd_response = pd.read_json(
            f'{self.data_root}/{self.gnd_response_json}', lines=True, convert_dates=False,
        ).set_index(['USER_ID'])
        self.expl_response = pd.read_json(
            f'{self.data_root}/{self.expl_response_json}', lines=True, convert_dates=False
        ).set_index(['USER_ID'])

    def test_vae_main(self):
        return vae_main(self.item_df, self.gnd_response, max_epochs=self.max_epochs, user_df=self.user_df)

    @functools.lru_cache()
    def test_bmt_main(self):
        return bmt_main(self.item_df, self.expl_response, self.gnd_response,
                        max_epochs=self.max_epochs, user_df=self.user_df, convert_time_unit=self.convert_time_unit)

    def test_shap(self):
        *_, bmt = self.test_bmt_main()
        item_df = self.item_df.assign(landingImage='missing')
        item_id = self.item_df.iloc[0].name

        Image.open(I2IImageEnv.image_format(
            self=AttrDict(item_df=item_df, explainer=bmt.to_explainer()),
            x={'_hist_items': [item_id], 'cand_items': [item_id, item_id]},
        )).show()
