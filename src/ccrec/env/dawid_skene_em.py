import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class VqNet(torch.nn.Module):
    """
    This module generates the posterior label distribution as well as the
    marginal likelihood of the worker-label distribution.
    It assumes I tasks, J workers, K classes.
    Typically, we set K=all observable classes during training,
    but then reset K=all classes, including the n/a class for inference.
    """

    def __init__(self, I, J, K):
        super().__init__()
        self.snr_logit = torch.nn.Parameter(torch.empty(J).uniform_(-0.07, 0.07))
        self.I = I
        self.set_K(K)

    def set_K(self, K):
        print(f"setting K={K}")
        self.K = K
        self.register_buffer("signal_const", torch.eye(K).reshape((1, K, K)))
        self.register_buffer(
            "noise_const",
            torch.ones((K, K)).reshape((1, K, K))
            / K,  # new changes after msmarco step-2-em
        )

    def forward(self, ii, jj, y):
        theta = (
            self.snr_logit.sigmoid().reshape((-1, 1, 1)) * self.signal_const
            + (-self.snr_logit).sigmoid().reshape((-1, 1, 1)) * self.noise_const
        ) / 2

        # complete loglike has dimension (I * |z|)
        if y.ndim <= 1 or y.shape[1] <= 1:
            y = y.ravel()
            log_theta = (theta / theta.sum(-1, keepdims=True)).log()  # J, |z|, |y|
            complete_log_lik = (
                F.one_hot(ii, self.I).float().T  # I * batch
                @ log_theta.swapaxes(-2, -1)[jj, y]  # batch * |z|
            )
        else:  # multi-label; for each label, y_norm @ (theta / theta @ mask).log()
            mask = (y > 0).float()
            y_norm = (y - 1).float() * mask
            y_norm = y_norm / torch.where(
                y_norm.any(-1, keepdims=True),
                y_norm.sum(-1, keepdims=True),
                1,
            )  # batch * |y|

            theta_per_label = (
                theta[jj]
                / torch.where(
                    mask.any(-1, keepdims=True),
                    torch.einsum("bzy,by->bz", theta[jj], mask),
                    1,
                )[:, :, None]
            )  # batch * |z| * |y|

            complete_log_lik_per_label = torch.einsum(
                "bzy,by->bz",
                theta_per_label.log(),
                y_norm,
            )

            complete_log_lik = (
                F.one_hot(ii, self.I).float().T  # I * batch
                @ complete_log_lik_per_label  # batch * |z|
            )

        qz = complete_log_lik.softmax(-1).detach()  # EM calls for a detach operation
        Vq = (qz * complete_log_lik).sum(-1) - (qz * qz.log()).sum(-1)

        return qz, Vq


class LitModel(pl.LightningModule):
    """training configuration"""

    def __init__(self, vq_net):
        super().__init__()
        self.vq_net = vq_net
        self._loss_hist = []

    def setup(self, stage):
        if stage == "fit":
            print(self.logger.log_dir)

    def training_step(self, batch, batch_idx):
        ii, jj, y = batch[:, 0], batch[:, 1], batch[:, 2:]
        _, Vq = self.vq_net(ii, jj, y)
        self.log("loss", -Vq.mean())
        self._loss_hist.append(-Vq.mean().item())
        return -Vq.mean()

    def configure_optimizers(self):
        weight_decay = float(os.environ.get("CCREC_DAWID_SKENE_WEIGHT_DECAY", 0.0005))
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=weight_decay)


def train_vq(I, J, K, ii, jj, y, *, plot_training_curve=True):
    """K is the total number of dimensions.
    For single-label y, K includes n/a class.
    For multi-label y, no n/a class is needed. Multi-label masks are inferred by y>0.
    """
    single_label = np.ndim(y) <= 1

    # For single_label, we assume a "none of the above" option as the last class name.
    # This class is positionally biased and excluded during training for worker SNRs.
    vq_net = VqNet(I, J, K - single_label)

    y = np.asarray(y)
    if single_label:
        assert 0 <= y.min() <= y.max() < K, "single label must be between [0, K)"
        data_tuples = np.asarray([ii, jj, y]).T
        unbiased_data = data_tuples[y < K - 1]  # unbiased single label between [0, K-1)

    else:  # multi_label
        assert K == y.shape[1], "multi-label must agree with class dimension K"
        data_tuples = np.hstack([np.asarray([ii, jj]).T, y])
        unbiased_data = data_tuples  # multi-label does not use n/a class

    train_loader = DataLoader(unbiased_data, batch_size=unbiased_data.shape[0])
    trainer = pl.Trainer(
        max_epochs=500,
        gpus=int(torch.cuda.is_available()),
        detect_anomaly=True,
    )
    model = LitModel(vq_net)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )

    if plot_training_curve:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(3, 2))
        plt.plot(model._loss_hist)
        plt.xlabel("step")
        plt.ylabel("-Vq per task")
        plt.grid()
        plt.show()

    with torch.no_grad():
        vq_net.set_K(K)
        snr = vq_net.snr_logit.sigmoid().detach().cpu().numpy()
        qz, _ = vq_net(
            torch.as_tensor(ii),
            torch.as_tensor(jj),
            torch.as_tensor(y),
        )
        qz = qz.detach().cpu().numpy()
    z_hat = qz.argmax(-1)

    return vq_net, model, snr, qz, z_hat


run_dawid_skene = train_vq
