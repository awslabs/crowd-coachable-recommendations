# adapted from test_prime_pantry
import pandas as pd, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest, rime, ccrec, ccrec.models, ccrec.models.bbpr
from ccrec import InteractiveExperiment, env
from ccrec.env import create_zero_shot, Env, parse_response
from ccrec.env.i2i_env import get_notebook_name
from ccrec.models.vae_models import VAEPretrainedModel


def create_information_retrieval(item_df):
    if isinstance(item_df, str):
        item_df = pd.read_csv(item_df)
    if 'ITEM_ID' in item_df:
        item_df = item_df.set_index('ITEM_ID')

    zero_shot = create_zero_shot(
        item_df,
        create_user_filter=lambda x: x['ITEM_TYPE'] == 'query',
        exclude_train=['ITEM_TYPE'],
    )
    return zero_shot


def test_information_retrieval(item_df='data/demo_information_retrieval/item_df.csv'):
    """ expect answer:
    ITEM_ID            q1            q2   p1   p2   p3   p4   p5
    USER_ID                                                     
    q1      -2.000000e+10 -1.000000e+10  0.0  0.0  0.0  0.0  0.0
    q2      -1.000000e+10 -2.000000e+10  0.0  0.0  0.0  0.0  0.0
    """
    zero_shot = create_information_retrieval(item_df)
    print(pd.DataFrame(
        zero_shot.prior_score.toarray(),
        index=zero_shot.user_df.index,
        columns=zero_shot.item_df.index,
    ))


def test_information_retrieval_ccrec(
    item_df='data/demo_information_retrieval/item_df.csv',
    max_epochs=0,  # choose 0 to skip retraining
    simulation=True,
    pretrained_checkpoint=None,
    train_requests=None,  # subsample training queries
    epsilon=0,  # use 'vae' to turn on candidate sampling
    role_arn=None,
    s3_prefix=None,
    working_model=None,  # VAEPretrainedModel
    multi_label=False,
    n_steps=2,
    exclude_train=['ITEM_TYPE'],
):
    """
    working_model=VAEPretrainedModel.from_pretrained('distilbert-base-uncased')
    working_model.load_state_dict(torch.load('checkpoints/VAE_model_prime_beta_0.002_dict'))
    epsilon='vae'
    """

    zero_shot = create_information_retrieval(item_df)
    user_df = zero_shot.user_df
    item_df = zero_shot.item_df

    if working_model is None:
        working_model = ccrec.models.bbpr.BertBPR(
            item_df, max_epochs=max_epochs, batch_size=10 * max(1, torch.cuda.device_count()),
            sample_with_prior=True, sample_with_posterior=0, elementwise_affine=False,
            replacement=False, n_negatives=5, valid_n_negatives=5,
            training_prior_fcn=lambda x: (x + 1 / x.shape[1]).clip(0, None).log(),
            pretrained_checkpoint=pretrained_checkpoint,
        )
    tfidf_model = rime.models.TF_IDF(item_df)

    if simulation:
        training_env_kw = {
            'oracle': ccrec.agent.Agent(tfidf_model),
            'prefix': 'pp-simu-train',
            'soft_label': False,
            'reserve_score': 0.1,
            'test_requests': train_requests,
            'exclude_train': exclude_train,
        }
    else:
        training_env_kw = {
            'oracle': env.I2IConfig(
                image=True,
                role_arn=role_arn,
                s3_prefix=s3_prefix,
            ),
            'prefix': 'pp-i2i-train',
            'multi_label': multi_label,
            'test_requests': train_requests,
            'exclude_train': exclude_train,
        }

    testing_env_kw = {
        'oracle': 'dummy',
        'prefix': 'pp-simu-test',
        'exclude_train': exclude_train,
    }
    baseline_models = []  # independent test run w/o competition with other models

    iexp = InteractiveExperiment(
        user_df, item_df, zero_shot.event_df,
        training_env_kw, testing_env_kw,
        working_model, baseline_models, epsilon,
    )

    iexp.run(n_steps=n_steps, test_every=None, test_before_train=False)
    print(iexp.training_env.event_df)

    return iexp
