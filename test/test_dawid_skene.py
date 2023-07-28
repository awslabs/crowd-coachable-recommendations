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
        I, J, K, ii, jj, y, show_training_curve=False
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
    N = I * 19 if multi_label else 3  # single_label is more efficient

    rng = np.random.RandomState(seed)
    z_true = rng.choice(K, I)
    snr_true = rng.uniform(0, 1, J)

    ii = rng.choice(I, N)
    jj = rng.choice(J, N)
    y = [rng.choice(K, p=get_p(z_true[i], snr_true[j], K)) for i, j in zip(ii, jj)]
    if multi_label:
        mask = (rng.rand(N, K) < 0.5).astype(int)
        mask[np.arange(N), y] = mask[np.arange(N), y] * 5
        y = mask

    vq_net, model, snr, qz, z_hat = run_dawid_skene(
        I, J, K + (1 - multi_label), ii, jj, y, show_training_curve=plot
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
    rms_error = (np.mean((snr_true - snr) ** 2)) ** 0.5
    print("simulation accuracy=", accuracy, " rms_error=", rms_error)

    if check_accuracy:
        assert accuracy > 0.7, "simulation accuracy should be > 0.7"
        assert rms_error < 0.25, "simulation rms should be < 0.25"
