import pandas as pd, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest, rime, ccrec, ccrec.models, ccrec.models.bbpr
from ccrec import InteractiveExperiment, env
from ccrec.env import create_zero_shot, Env, parse_response
from ccrec.env.i2i_env import get_notebook_name
from ccrec.models.vae_models import VAEPretrainedModel
from ccrec.util.amazon_review_prime_pantry import get_item_df


def test_sample_data(data_root="data/amazon_review_prime_pantry"):
    return get_item_df(data_root=data_root)


@pytest.mark.parametrize("nrows", [10])
@pytest.mark.parametrize("max_epochs", [1])
def test_prime_pantry_vae(
    nrows,
    max_epochs,
    data_root="data/amazon_review_prime_pantry",
    gnd_response_json="prime_pantry_test_response.json.gz",
):
    from ccrec.util.demo_data import DemoData

    item_df, tfidf_csr = get_item_df(nrows=nrows, data_root=data_root)
    user_df = create_zero_shot(item_df, copy_item_id=False).user_df
    gnd_response = (
        pd.read_json(
            f"{data_root}/{gnd_response_json}",
            lines=True,
            convert_dates=False,
        )
        .rename({"level_0": "USER_ID"}, axis=1)
        .set_index(["USER_ID", "TEST_START_TIME"])
    )
    demo_data_obj = DemoData(
        data_root=None,
        user_df=user_df,
        item_df=item_df,
        expl_response=None,
        gnd_response=gnd_response,
        max_epochs=max_epochs,
    )
    return demo_data_obj.run_shap(model_main="run_vae_main")


@pytest.mark.parametrize("nrows", [10])
@pytest.mark.parametrize("max_epochs", [0])
def test_prime_pantry_ccrec(
    nrows,  # None
    max_epochs,  # choose 0 to skip retraining
    simulation=True,
    pretrained_checkpoint=None,
    train_requests=None,  # subsample training queries
    epsilon=0,
    role_arn=None,
    s3_prefix=None,
    working_model=None,  # VAEPretrainedModel
    multi_label=False,
    data_root="data/amazon_review_prime_pantry",
    gnd_response_json="prime_pantry_test_response.json.gz",
    n_steps=1,
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

    item_df, tfidf_csr = get_item_df(nrows=nrows, data_root=data_root)
    zero_shot = create_zero_shot(item_df, copy_item_id=False)
    user_df = zero_shot.user_df

    gnd_response = (
        pd.read_json(
            f"{data_root}/{gnd_response_json}",
            lines=True,
            convert_dates=False,
        )
        .rename({"level_0": "USER_ID"}, axis=1)
        .set_index(["USER_ID", "TEST_START_TIME"])
    )
    gnd_events = pd.concat(
        [
            zero_shot.event_df,  # history features
            parse_response(
                gnd_response
            ),  # labels reformatted as target events with values
        ]
    )

    if working_model is None:
        working_model = ccrec.models.bbpr.BertBPR(
            item_df,
            None,
            max_epochs=max_epochs,
            batch_size=10 * max(1, torch.cuda.device_count()),
            sample_with_prior=True,
            sample_with_posterior=0,
            elementwise_affine=False,
            replacement=False,
            n_negatives=5,
            valid_n_negatives=5,
            training_prior_fcn=lambda x: (x + 1 / x.shape[1]).clip(0, None).log(),
            pretrained_checkpoint=pretrained_checkpoint,
        )
    tfidf_model = rime.models.zero_shot.ItemKNN(
        item_df.assign(embedding=tfidf_csr.toarray().tolist(), _hist_len=1)
    )

    if simulation:
        training_env_kw = {
            "oracle": ccrec.agent.Agent(tfidf_model),
            "prefix": "pp-simu-train",
            "soft_label": False,
            "reserve_score": 0.1,
            "test_requests": train_requests,
        }
    else:
        training_env_kw = {
            "oracle": env.I2IConfig(
                image=True,
                role_arn=role_arn,
                s3_prefix=s3_prefix,
            ),
            "prefix": "pp-i2i-train",
            "multi_label": multi_label,
            "test_requests": train_requests,
        }

    all_events = gnd_events  # need to create reranking priors at test time
    testing_env_kw = {
        "oracle": rime.Dataset(user_df, item_df, gnd_events),
        "prefix": "pp-simu-test",
        "sample_with_prior": 1e5,  # reranking
    }
    baseline_models = []  # independent test run w/o competition with other models

    iexp = InteractiveExperiment(
        user_df,
        item_df,
        all_events,
        training_env_kw,
        testing_env_kw,
        working_model,
        baseline_models,
        epsilon,
    )

    iexp.run(n_steps=n_steps, test_every=1, test_before_train=True)
    print(pd.DataFrame(iexp.testing_env._reward_by_policy))

    return iexp


test_prime_pantry_ccrec(nrows=10, max_epochs=0)
