from ccrec.env.dawid_skene_em import run_dawid_skene


def test_dawid_skene_dummy():
    I = 3
    J = 4
    K = 5
    ii = [0, 1, 2, 0, 1, 2]  # tasks
    jj = [0, 0, 1, 2, 3, 3]  # workers
    y = [0, 1, 2, 3, 4, 2]  # labels
    mask = [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0],
    ]
    vq_net, model, snr, qz, z_hat = run_dawid_skene(
        I, J, K, ii, jj, y, show_training_curve=False
    )
    print("without mask, z_hat =", z_hat, "snr=", snr)
    vq_net, model, snr, qz, z_hat = run_dawid_skene(
        I, J, K, ii, jj, y, mask, show_training_curve=False
    )
    print("with mask, z_hat =", z_hat, "snr=", snr)
    print(
        "with mask, worker 0 is not penalized for disagreeing with worker 3 due to different masks"
    )
