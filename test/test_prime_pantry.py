import pandas as pd, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest, rime, ccrec, ccrec.models, ccrec.models.bbpr
from ccrec import InteractiveExperiment, env
from ccrec.env import create_zero_shot, Env, parse_response
from ccrec.env.i2i_env import get_notebook_name
from ccrec.models.vae_models import VAEPretrainedModel


def _shorten_brand_name_function(x):
    if not isinstance(x, str):
        return x
    for i, part in enumerate(x.split(' ')):
        try:
            int(part)
            continue
        except Exception:
            return ' '.join(x.split(' ')[:i + 1])


def get_item_df(meta_file='data/prime_pantry/pp-items-full-meta.csv',
                landingImageURL_file='data/prime_pantry/landingImageURL-new.csv',
                landingImageURL_folder='data/prime_pantry/landingImage',
                shorten_brand_name=True,
                return_tfidf_csr=True):
    item_df = pd.read_csv(meta_file)
    item_df = item_df.set_index('ITEM_ID').assign(
        TITLE=lambda df: df['DESCRIPTION'].apply(lambda x: ' '.join(x.split(' ')[:30]))
    )[['TITLE', 'BRAND']]

    if shorten_brand_name:
        item_df['BRAND'] = item_df['BRAND'].apply(_shorten_brand_name_function)
    print(f'# items {len(item_df)}, # brands {item_df["BRAND"].nunique()}')

    item_df = item_df.join(
        pd.read_csv(landingImageURL_file, names=['asin', 'landingImage']).set_index('asin')
    )
    item_df = item_df[item_df['landingImage'].notnull()]
    item_df = item_df[item_df['landingImage'].apply(lambda x: x.endswith('.jpg'))]
    item_df['landingImage'] = [f'{landingImageURL_folder}/{x}.jpg' for x in item_df.index.values]

    tfidf_fit = TfidfVectorizer().fit(item_df['TITLE'].tolist())
    tfidf_csr = tfidf_fit.transform(item_df['TITLE'].tolist())

    item_df['tfidf_indices'] = np.split(tfidf_csr.indices, tfidf_csr.indptr[1:-1])
    item_df['tfidf_words'] = np.split(
        np.array(tfidf_fit.get_feature_names())[tfidf_csr.indices],
        tfidf_csr.indptr[1:-1])
    item_df['tfidf_data'] = np.split(tfidf_csr.data, tfidf_csr.indptr[1:-1])

    item_df = item_df.join(
        pd.concat({
            'words': item_df['tfidf_words'].explode(),
            'data': item_df['tfidf_data'].explode(),
        }, axis=1).sort_values('data', ascending=False)
        .groupby('ITEM_ID')['words'].apply(lambda x: x[:5].tolist())
        .to_frame('sorted_words')
    ).drop('tfidf_words', axis=1)

    return (item_df, tfidf_csr) if return_tfidf_csr else item_df


def get_interactions_df(interactions_file='data/prime_pantry/pp-all_interactions.csv'):
    return pd.read_csv(interactions_file).sort_values("TIMESTAMP", kind='mergesort')


def get_collection_event_df(event_file='prime-pantry-i2i-online-baseline4.json'):
    return pd.read_json(event_file, lines=True, convert_dates=False)


def get_collection_response(response_file='prime-pantry-i2i-online-baseline4-response.json'):
    response = pd.read_json(response_file, lines=True, convert_dates=False)
    return response.set_index(response.columns[:2].tolist())


@pytest.mark.skip(reason='requires data')
def test_prime_pantry_ccrec(
    simulation=True,
    pretrained_checkpoint=None,
    train_requests=None,  # subsample training queries
    epsilon=0,
    max_epochs=1,  # choose 0 to skip retraining
    role_arn=None,
    s3_prefix=None,
    working_model=None,  # VAEPretrainedModel
    multi_label=False,
):
    """
    * pretrained_checkpoint
        * tf-idf-coached: lightning_logs/version_29/checkpoints/state-dict.pth
    * VAE collection only
        train_requests = pd.read_csv('user_unc.csv', nrows=1000).assign(
                            TEST_START_TIME=1).set_index(['USER_ID', 'TEST_START_TIME'])[[]]
        working_model=VAEPretrainedModel.from_pretrained('distilbert-base-uncased')
        working_model.load_state_dict(torch.load('checkpoints/VAE_model_prime_beta_0.002_dict'))
        epsilon='vae'
    """

    item_df, tfidf_csr = get_item_df()
    zero_shot = create_zero_shot(item_df)
    user_df = zero_shot.user_df

    gnd_response = pd.read_json(
        'prime-pantry-i2i-online-baseline4-response.json', lines=True, convert_dates=False,
    ).rename({'level_0': 'USER_ID'}, axis=1).set_index(['USER_ID', 'TEST_START_TIME'])
    gnd_events = pd.concat([
        zero_shot.event_df,  # history features
        parse_response(gnd_response),  # labels reformatted as target events with values
    ])

    if working_model is None:
        working_model = ccrec.models.bbpr.BertBPR(
            item_df, max_epochs=max_epochs, batch_size=10 * torch.cuda.device_count(),
            sample_with_prior=True, sample_with_posterior=0, elementwise_affine=False,
            replacement=False, n_negatives=5, valid_n_negatives=5,
            training_prior_fcn=lambda x: (x + 1 / x.shape[1]).clip(0, None).log(),
            pretrained_checkpoint=pretrained_checkpoint,
        )
    tfidf_model = rime.models.zero_shot.ItemKNN(
        item_df.assign(embedding=tfidf_csr.toarray().tolist(), _hist_len=1))

    if simulation:
        training_env_kw = {
            'oracle': ccrec.agent.Agent(tfidf_model),
            'prefix': 'pp-simu-train',
            'soft_label': False,
            'reserve_score': 0.1,
            'test_requests': train_requests,
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
        }

    all_events = gnd_events  # need to create reranking priors at test time
    testing_env_kw = {
        'oracle': rime.Dataset(user_df, item_df, gnd_events),
        'prefix': 'pp-simu-test',
        'sample_with_prior': 1e5,  # reranking
    }
    baseline_models = []  # independent test run w/o competition with other models

    iexp = InteractiveExperiment(
        user_df, item_df, all_events,
        training_env_kw, testing_env_kw,
        working_model, baseline_models, epsilon,
    )

    iexp.run(n_steps=10, test_every=1, test_before_train=True)
    print(pd.DataFrame(iexp.testing_env._reward_by_policy))

    return iexp
