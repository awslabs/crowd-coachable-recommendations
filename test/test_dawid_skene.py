import numpy as np
from ccrec.env.dawid_skene_em import run_dawid_skene
import pytest


@pytest.mark.parametrize("multi_label", [False, True])
def test_dawid_skene_dummy(multi_label):
    I = 3  # number of tasks
    J = 4  # number of workers
    K = 5  # number of labels

    ii = [0, 1, 2, 0, 1, 2]  # tasks
    jj = [0, 0, 1, 2, 3, 3]  # workers

    if not multi_label:
        y = [0, 1, 2, 3, 4, 2]  # labels
    else:
        y = [  # multi-labels. mask is inferred by y > 0
            [5, 0, 0, 0, 0],
            [1, 5, 0, 0, 0],
            [1, 0, 5, 0, 0],
            [1, 0, 0, 5, 0],
            [1, 0, 0, 0, 5],
            [1, 0, 5, 0, 0],
        ]

    vq_net, model, snr, qz, z_hat = run_dawid_skene(
        I, J, K, ii, jj, y, plot_training_curve=False
    )
    print("z_hat =", z_hat, "snr=", snr)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_p(z, snr, K):
    return one_hot(z, K) * snr + np.ones(K) / K * (1 - snr)


@pytest.mark.parametrize("multi_label", [False, True])
def test_dawid_skene_simulation(multi_label, seed=42, plot=False, check_accuracy=True):
    I = 30
    J = 10
    K = 5
    N = I * 8 if multi_label else I * 4  # keep coverage similar

    rng = np.random.RandomState(seed)
    z_true = rng.choice(K, I)
    snr_true = rng.beta(2, 1, J)

    ii = rng.choice(I, N)
    jj = rng.choice(J, N)
    y = [rng.choice(K, p=get_p(z_true[i], snr_true[j], K)) for i, j in zip(ii, jj)]
    if multi_label:
        y_with_mask = (rng.rand(N, K) < 0.5).astype(int)  # mask
        y_with_mask[np.arange(N), y] = y_with_mask[np.arange(N), y] * 5
        y = y_with_mask

    vq_net, model, snr, qz, z_hat = run_dawid_skene(
        I, J, K, ii, jj, y, plot_training_curve=plot
    )
    if plot:
        import matplotlib.pyplot as plt

        plt.scatter(z_true, z_hat, np.bincount(ii, minlength=I) * 5)
        plt.title("true label vs estimated label")
        plt.show()
        plt.scatter(snr_true, snr, np.bincount(jj, minlength=J) * 5)
        plt.title("true snr vs estimated snr")
        plt.show()

    accuracy = np.mean(z_true == z_hat)
    snr_corr = np.corrcoef(snr_true, snr)[0, 1]
    print("simulation accuracy=", accuracy, " snr_corr=", snr_corr)

    if check_accuracy:
        assert accuracy > 0.7, "simulation accuracy should be > 0.7"
        assert snr_corr > 0.4, "simulation snr_corr should be > 0.4"
