import math
from typing import Dict, List, Optional, Set, Tuple, Union

from transformers.configuration_utils import PretrainedConfig
from transformers import DistilBertPreTrainedModel, DistilBertModel
from transformers.activations import get_activation
from transformers.modeling_outputs import MaskedLMOutput

import torch, os
from torch import nn
from torch.nn import CrossEntropyLoss
import copy


class EmbeddingModel(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, freeze_bert=False):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)

        if freeze_bert:
            self.vocab_transform = torch.nn.Identity()
            self.activation = torch.nn.Identity()
        else:
            self.vocab_transform = nn.Linear(config.dim, config.dim)
            self.activation = get_activation(config.activation)

        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.standard_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)

        self.loss_fct = nn.CrossEntropyLoss(reduction="none")

        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False

    def generate_mean(self, hidden_states):
        raise NotImplementedError("return type: torch.Tensor")

    def generate_std(self, hidden_states):
        raise NotImplementedError("return type: float or torch.Tensor")

    def compute_output_loss(self, mu, std, prediction_logits, input_ids, labels):
        raise NotImplementedError("return type: torch.Tensor")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_cls: Optional[bool] = False,
        return_mean_std: Optional[bool] = False,
        return_embedding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        seq_length = dlbrt_output[0].size(dim=1)
        hidden_states = dlbrt_output[0][:, 0, :]  # (bs, dim)

        if return_cls:
            return hidden_states

        mu = self.generate_mean(hidden_states)
        std = self.generate_std(hidden_states)

        if return_mean_std:
            return mu, std

        eps = torch.randn_like(mu)
        hidden_states = eps * std + mu

        hidden_states = self.vocab_transform(hidden_states)  # (bs, dim)
        hidden_states = self.activation(hidden_states)  # (bs, dim)

        # use standard_layer_norm to avoid using the weights in the trained layer_norm and keep the norm of the embedding as a constant
        if return_embedding:
            return self.standard_layer_norm(hidden_states)

        prediction_logits = self.vocab_layer_norm(hidden_states)  # (bs, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, vocab_size)

        bs = prediction_logits.size(dim=0)
        vocab_size = prediction_logits.size(dim=1)

        prediction_logits = torch.reshape(prediction_logits, (bs, 1, vocab_size))
        prediction_logits = prediction_logits.repeat(1, seq_length, 1)

        output_loss = self.compute_output_loss(
            mu, std, prediction_logits, input_ids, labels
        )

        return MaskedLMOutput(
            loss=output_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )

    def cls_to_embedding(self, hidden_states):
        mu = self.generate_mean(hidden_states)
        std = self.generate_std(hidden_states)

        eps = torch.randn_like(mu)
        hidden_states = eps * std + mu

        hidden_states = self.vocab_transform(hidden_states)  # (bs, dim)
        hidden_states = self.activation(hidden_states)  # (bs, dim)

        return self.standard_layer_norm(hidden_states)


class MaskedPretrainedModel(EmbeddingModel):
    def __init__(self, config: PretrainedConfig, **kw):
        super().__init__(config, **kw)
        self.std = 0.0

    def generate_mean(self, hidden_states):
        return hidden_states

    def generate_std(self, hidden_states):
        return self.std

    def compute_output_loss(self, mu, std, prediction_logits, input_ids, labels):
        if labels is None:
            labels = (
                input_ids
                if int(os.environ.get("CCREC_LEGACY_VAE_BUG", 0))
                else torch.where(input_ids == 0, -100, input_ids)
            )

        label_density = (labels != -100).sum(1).float().mean()
        recon_loss = self.loss_fct(prediction_logits.swapaxes(1, 2), labels)
        return recon_loss.sum(1) / label_density


class FrozenPretrainedModel(MaskedPretrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config, freeze_bert=True)


class VAEPretrainedModel(EmbeddingModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.fc_mu = nn.Linear(config.dim, config.dim)
        self.fc_var = nn.Linear(config.dim, config.dim)

        self.vae_beta = 1e-5

    def VAE_post_init(self):
        dim = self.fc_var.weight.size(1)

        # intialize fc_mu to be identity
        self.fc_mu.bias.data.zero_()
        self.fc_mu.weight.data = torch.eye(dim)

        # initialize fc_var according to prior
        var_init = 0.01
        stdv = var_init / math.sqrt(dim)
        self.fc_var.weight.data.uniform_(-stdv, stdv)
        self.fc_var.bias.data.zero_()

    def set_beta(self, beta):
        self.vae_beta = beta

    def generate_mean(self, hidden_states):
        return self.fc_mu(hidden_states)

    def generate_std(self, hidden_states):
        log_var = self.fc_var(hidden_states)
        return torch.exp(0.5 * log_var)

    def compute_output_loss(self, mu, std, prediction_logits, input_ids, labels):

        if labels is None:
            labels = (
                input_ids
                if int(os.environ.get("CCREC_LEGACY_VAE_BUG", 0))
                else torch.where(input_ids == 0, -100, input_ids)
            )
        label_density = (labels != -100).sum(1).float().mean()
        recon_loss = self.loss_fct(prediction_logits.swapaxes(1, 2), labels)
        recon_loss = recon_loss.sum(1) / label_density

        kld_loss = -0.5 * torch.sum(1 + 2 * torch.log(std) - mu ** 2 - std ** 2, dim=1)

        return recon_loss + self.vae_beta * kld_loss
