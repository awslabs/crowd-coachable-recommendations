from ccrec.env.dawid_skene_em import run_dawid_skene


def test_dawid_skene_dummy():
    I = 3
    J = 4
    K = 5
    ii = [0, 1, 2, 0, 1, 2]
    jj = [0, 0, 1, 2, 3, 3]
    y = [0, 1, 2, 3, 4, 5]
    vq_net, model, snr, qz, z_hat = run_dawid_skene(I, J, K, ii, jj, y)
    print("z_hat is", z_hat)
