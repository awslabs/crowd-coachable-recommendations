from ccrec.env.dawid_skene_em import run_dawid_skene


def test_dawid_skene_dummy():
    I = 3  # number of tasks
    J = 4  # number of workers
    K = 5  # number of labels

    ii = [0, 1, 2, 0, 1, 2]  # tasks
    jj = [0, 0, 1, 2, 3, 3]  # workers
    y = [0, 1, 2, 3, 4, 2]  # labels

    vq_net, model, snr, qz, z_hat = run_dawid_skene(
        I, J, K, ii, jj, y, show_training_curve=False
    )
    print("For single-label without mask, z_hat =", z_hat, "snr=", snr)

    y_multi = [  # multi-labels. mask is inferred by y_multi > 0
        [5, 0, 0, 0, 0],
        [1, 5, 0, 0, 0],
        [1, 0, 5, 0, 0],
        [1, 0, 0, 5, 0],
        [1, 0, 0, 0, 5],
        [1, 0, 5, 0, 0],
    ]
    vq_net, model, snr, qz, z_hat = run_dawid_skene(
        I, J, K, ii, jj, y_multi, show_training_curve=False
    )
    print("For multi-label with mask, z_hat =", z_hat, "snr=", snr)
    print("posterior class probability =", qz)
