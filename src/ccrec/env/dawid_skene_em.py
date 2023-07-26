import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt


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
        self.register_buffer("signal_const", torch.eye(K).reshape((1, K, K)))
        self.register_buffer(
            "noise_const",
            torch.ones((K, K)).reshape((1, K, K))
            / K,  # new changes after msmarco step-2-em
        )

    def forward(self, ii, jj, y, mask=None):
        theta = (
            self.snr_logit.sigmoid().reshape((-1, 1, 1)) * self.signal_const
            + (-self.snr_logit).sigmoid().reshape((-1, 1, 1)) * self.noise_const
        ) / 2

        # complete loglike has dimension (I * |z|)
        if mask is None or mask.shape[1] == 0:
            log_theta = (theta / theta.sum(-1, keepdims=True)).log()  # J, |z|, |y|
            complete_log_lik = (
                F.one_hot(ii, self.I).float().T  # I * batch
                @ log_theta.swapaxes(-2, -1)[jj, y]  # batch * |z|
            )
        else:
            mask = mask + 1e-10  # avoid nan
            masked_theta_sum = torch.einsum("bzy,by->bz", theta[jj], mask)
            complete_log_lik = (
                F.one_hot(ii, self.I).float().T  # I * batch
                @ (
                    theta.swapaxes(-2, -1)[jj, y] / masked_theta_sum  # batch * |z|
                ).log()
            )

        qz = complete_log_lik.softmax(-1).detach()  # stick to EM; negligible effects
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
        ii, jj, y, mask = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3:]
        _, Vq = self.vq_net(ii, jj, y, mask)
        self.log("loss", -Vq.mean())
        self._loss_hist.append(-Vq.mean().item())
        return -Vq.mean()

    def configure_optimizers(self):
        weight_decay = float(os.environ.get("CCREC_DAWID_SKENE_WEIGHT_DECAY", 0.0005))
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=weight_decay)


def train_vq(I, J, K, ii, jj, y, mask=None):
    """K is the total number of dimensions including n/a.
    Training will exclude n/a first and then add n/a back for inference
    """
    vq_net = VqNet(I, J, K - 1)

    data_tuples = np.asarray([ii, jj, y]).T
    if mask is not None:
        assert K == mask.shape[1], "mask dimension should match K, including n/a"
        data_tuples = np.hstack([data_tuples, mask])
    unbiased_data = data_tuples[data_tuples[:, -1] < K - 1, :-1]

    train_loader = DataLoader(unbiased_data, batch_size=unbiased_data.shape[0])
    trainer = pl.Trainer(
        max_epochs=500,
        gpus=int(torch.cuda.is_available()),
    )
    model = LitModel(vq_net)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )

    plt.figure(figsize=(3, 2))
    plt.plot(model._loss_hist)
    plt.xlabel("step")
    plt.ylabel("-Vq per task")
    plt.grid()
    plt.show()

    with torch.no_grad():
        vq_net.set_K(K)
        snr = vq_net.snr_logit.sigmoid().detach().cpu().numpy()
        qz, _ = vq_net(torch.as_tensor(ii), torch.as_tensor(jj), torch.as_tensor(y))
        qz = qz.detach().cpu().numpy()
    z_hat = qz.argmax(-1)

    return vq_net, model, snr, qz, z_hat


run_dawid_skene = train_vq
