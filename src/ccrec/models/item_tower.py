import torch, pandas as pd, numpy as np, functools, os, collections
from datasets import Dataset, DatasetDict
from rime_lite.util import auto_device
from ccrec.util.data_parallel import DataParallel
import warnings


class ItemTowerBase(torch.nn.Module):
    """support text -> inputs -> cls -> embedding / loss;
    tokenizer is required for text_to_inputs, to_map_fn and to_explainer for e2e inference
    """

    def __init__(self, *module_list, tokenizer=None, tokenizer_kw={}):
        super().__init__()
        self.module_list = module_list
        self.tokenizer = tokenizer
        _default_tokenizer_kw = {
            "truncation": True,
            "padding": "max_length",
            "max_length": int(os.environ.get("CCREC_MAX_LENGTH", 200)),
            "return_tensors": "pt",
        }
        self.tokenizer_kw = {**_default_tokenizer_kw, **tokenizer_kw}

    @property
    def device(self):
        return self.module_list[
            -1
        ].device  # workaround for VAEPretrainedModel.device not working

    def text_to_inputs(self, text):
        return self.tokenizer(text, **self.tokenizer_kw)

    def forward(
        self,
        cls=None,
        text=None,
        input_step="inputs",
        output_step="embedding",
        **inputs,
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support {input_step}->{output_step} forward"
        )

    def to_map_fn(self, input_step, output_step, data_parallel=False, sample_param=0):
        """this may change eval mode and device in-place"""
        assert (
            self.tokenizer is not None or input_step != "text"
        ), "map_fn with text input requires tokenizer attribute"

        self.eval()
        if hasattr(self, "set_sample_param"):
            self.set_sample_param(sample_param)

        auto_wrap_dict = (
            lambda x: x
            if isinstance(x, collections.abc.Mapping)
            else {"cls": x}
            if input_step == "cls"
            else {"text": x}
            if input_step == "text"
            else NotImplemented
        )

        if input_step == "text":
            input_step = "inputs"
            tokenizer, tokenizer_kw = self.tokenizer, self.tokenizer_kw
            auto_wrap_text = lambda x: tokenizer(x["text"], **tokenizer_kw)
        else:
            auto_wrap_text = lambda x: x

        if data_parallel:
            self = DataParallel(self.cuda()).cache_replicas()
            auto_wrap_device = lambda x: {k: v.cuda() for k, v in x.items()}
        else:
            auto_wrap_device = lambda x: x

        return torch.no_grad()(
            lambda x: {
                output_step: self(
                    **auto_wrap_device(auto_wrap_text(auto_wrap_dict(x))),
                    input_step=input_step,
                    output_step=output_step,
                )
                .cpu()
                .numpy()
            }
        )

    def to_explainer(self, **kw):
        from ccrec.util.shap_explainer import I2IExplainer

        assert self.tokenizer is not None, "to_explainer requires tokenizer attribute"
        self = self.to(auto_device()).eval()
        return I2IExplainer(self, self.tokenizer, **kw)


class NaiveItemTower(ItemTowerBase):
    """standard_layer_norm on top of cls token"""

    def __init__(self, cls_model, standard_layer_norm, **kw):
        super().__init__(cls_model, standard_layer_norm, **kw)
        self.cls_model = cls_model
        self.standard_layer_norm = standard_layer_norm

    def forward(
        self,
        cls=None,
        text=None,
        input_step="inputs",
        output_step="embedding",
        **inputs,
    ):
        if input_step == "text":
            inputs = self.text_to_inputs(text=text)
            input_step = "inputs"

        if input_step == "inputs":
            # bugfix: weird that self.device does not agree with self.cls_model.device
            inputs = {k: v.to(self.cls_model.device) for k, v in inputs.items()}
            last_hidden_state = self.cls_model(**inputs).last_hidden_state
            cls = last_hidden_state[:, 0]
        else:  # cls
            cls = cls.to(self.device)

        if output_step == "embedding":
            output_step = os.environ["CCREC_EMBEDDING_TYPE"]
            warnings.warn(
                f"{self.__class__} inferring output_step from CCREC_EMBEDDING_TYPE as {output_step}"
            )

        if output_step in ["cls", "mu", "mean"]:
            return cls
        elif output_step == "mean_layer_norm":
            return self.standard_layer_norm(cls)
        elif output_step == "mean_pooling":
            assert input_step != "cls", "cannot create mean pooling from cls"
            mask = inputs["attention_mask"]

            last_hidden_state = last_hidden_state.masked_fill(
                ~mask[..., None].bool(), 0.0
            )
            sentence_embeddings = (
                last_hidden_state.sum(dim=1) / mask.sum(dim=1)[..., None]
            )
            return sentence_embeddings  # unnormalized

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support {input_step}->{output_step} forward"
        )
