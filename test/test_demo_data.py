import pandas as pd, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest, rime, ccrec, ccrec.models, ccrec.models.bbpr
from ccrec import InteractiveExperiment, env
from ccrec.env import create_zero_shot, Env, parse_response
from ccrec.env.i2i_env import get_notebook_name
from ccrec.models.vae_models import VAEPretrainedModel
from ccrec.util.amazon_review_prime_pantry import get_item_df
from ccrec.models.vae_lightning import vae_main
from ccrec.models.bert_mt import bmt_main


def test_vae_main(
    data_root='data/demo',
    user_json='user_df.json',
    item_csv="item_df.csv",
    gnd_response_json='test_response.json',
    max_epochs=1,
):
    user_df = pd.read_json(f'{data_root}/{user_json}', lines=True).set_index("USER_ID")
    item_df = pd.read_csv(f'{data_root}/{item_csv}').set_index("ITEM_ID")
    gnd_response = pd.read_json(
        f'{data_root}/{gnd_response_json}', lines=True, convert_dates=False,
    ).set_index(['USER_ID'])
    return vae_main(item_df, gnd_response, max_epochs=max_epochs, user_df=user_df)


def test_bmt_main(
    data_root='data/demo',
    user_json='user_df.json',
    item_csv='item_df.csv',
    expl_response_json='expl_response.json',
    max_epochs=1,
    gnd_response_json='test_response.json',
):
    user_df = pd.read_json(f'{data_root}/{user_json}', lines=True).set_index("USER_ID")
    item_df = pd.read_csv(f'{data_root}/{item_csv}').set_index("ITEM_ID")
    expl_response = pd.read_json(
        f'{data_root}/{expl_response_json}', lines=True, convert_dates=False
    ).set_index(['USER_ID'])
    gnd_response = pd.read_json(
        f'{data_root}/{gnd_response_json}', lines=True, convert_dates=False,
    ).set_index(['USER_ID'])
    return bmt_main(item_df, expl_response, gnd_response, max_epochs=max_epochs, user_df=user_df, convert_time_unit=False)
