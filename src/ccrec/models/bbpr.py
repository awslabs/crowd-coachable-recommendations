from transformers import AutoTokenizer, AutoModel, DefaultDataCollator
from datasets import Dataset
from rime.util import (
    default_random_split,
    empty_cache_on_exit,
    _LitValidated,
    _ReduceLRLoadCkpt,
    auto_cast_lazy_score,
    sps_to_torch,
    auto_device,
    timed,
)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import functools, torch, numpy as np, pandas as pd
import os, itertools, dataclasses, warnings, collections, re, tqdm
from ccrec.util import _device_mode_context
from ccrec.util.shap_explainer import I2IExplainer
from ccrec.models.item_tower import NaiveItemTower
import rime
from ccrec.env import create_reranking_dataset

# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html


def _create_bert(model_name, freeze_bert):
    model = AutoModel.from_pretrained(model_name)
    if freeze_bert > 0:
        for param in model.parameters():
            param.requires_grad = False

    elif freeze_bert < 0:
        for param in model.embeddings.parameters():
            param.requires_grad = False

        for i in range(len(model.encoder.layer) + freeze_bert):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = False

    return model


class _BertBPR(_LitValidated):
    def __init__(
        self,
        all_inputs,
        model_name="distilbert-base-uncased",
        freeze_bert=0,
        n_negatives=10,
        valid_n_negatives=None,
        lr=None,
        weight_decay=None,
        training_prior_fcn=lambda x: x,
        do_validation=True,
        replacement=True,
        sample_with_prior=True,
        sample_with_posterior=0.5,
        elementwise_affine=True,  # set to False to eliminate gradients from f(x)'f(x) in N/A class
        pretrained_checkpoint=None,
        tokenizer=None,
        tokenizer_kw={},
        **bpr_kw,
    ):
        super().__init__()
        if lr is None:
            lr = 0.1 if freeze_bert > 0 else 1e-4
        if weight_decay is None:
            weight_decay = 1e-5 if freeze_bert > 0 else 0
        if valid_n_negatives is None:
            valid_n_negatives = n_negatives
        self.sample_with_prior = sample_with_prior
        self.sample_with_posterior = sample_with_posterior

        self.save_hyperparameters(
            "freeze_bert",
            "n_negatives",
            "valid_n_negatives",
            "lr",
            "weight_decay",
            "replacement",
            "do_validation",
        )
        for name in self.hparams:
            setattr(self, name, getattr(self.hparams, name))
        self.training_prior_fcn = training_prior_fcn

        self.item_tower = NaiveItemTower(
            _create_bert(model_name, freeze_bert),
            torch.nn.LayerNorm(
                768, elementwise_affine=elementwise_affine
            ),  # TODO: other transform layers
            tokenizer=tokenizer,
            tokenizer_kw=tokenizer_kw,
        )
        self.all_inputs = all_inputs

        if pretrained_checkpoint is not None:
            self.load_state_dict(torch.load(pretrained_checkpoint))

        self.objective = "multiple_nrl"

    def set_training_data(
        self, i_to_ptr=None, j_to_ptr=None, prior_score=None, item_freq=None
    ):
        self.register_buffer("i_to_ptr", torch.as_tensor(i_to_ptr), False)
        self.register_buffer("j_to_ptr", torch.as_tensor(j_to_ptr), False)
        if prior_score is not None and self.sample_with_prior:
            self.register_buffer(
                "tr_prior_score", sps_to_torch(prior_score, "cpu"), False
            )
        if item_freq is not None:
            item_proposal = (item_freq + 0.1) ** self.sample_with_posterior
            self.register_buffer(
                "tr_item_proposal", torch.as_tensor(item_proposal), False
            )
        if self.objective == "multiple_nrl":
            self.compute_user_to_negatives()

    def setup(self, stage):  # auto-call in fit loop
        if stage == "fit":
            print(self._checkpoint.dirpath)

    def forward(self, batch):  # tokenized or ptr
        if isinstance(batch, collections.abc.Mapping):  # tokenized
            return self.item_tower(**batch)
        elif hasattr(self, "all_cls"):  # ptr
            return self.item_tower(self.all_cls[batch], input_step="cls")
        else:  # ptr to all_inputs
            return self.item_tower(**{k: v[batch] for k, v in self.all_inputs.items()})

    def _pairwise(self, i, j):  # auto-broadcast on first dimension
        x = self.forward(self.i_to_ptr[i.ravel()]).reshape([*i.shape, -1])
        y = self.forward(self.j_to_ptr[j.ravel()]).reshape([*j.shape, -1])
        return (x * y).sum(-1)

    def training_and_validation_step(self, batch, batch_idx):
        i, j, w = batch.T
        i = i.to(int)
        j = j.to(int)
        if self.objective == "bpr":
            pos_score = self._pairwise(i, j)  # bsz

            n_negatives = self.n_negatives if self.training else self.valid_n_negatives
            n_shape = (n_negatives, len(batch))
            loglik = []

            with torch.no_grad():
                if hasattr(self, "tr_prior_score"):
                    if hasattr(self.tr_prior_score, "as_tensor"):
                        prior_score = self.tr_prior_score[i.tolist()].as_tensor(
                            i.device
                        )
                    else:
                        prior_score = self.tr_prior_score.index_select(0, i).to_dense()
                    nj = torch.multinomial(
                        (
                            self.training_prior_fcn(prior_score)
                            + self.tr_item_proposal.log()
                        ).softmax(1),
                        n_negatives,
                        self.replacement,
                    ).T
                else:
                    nj = torch.multinomial(
                        self.tr_item_proposal, np.prod(n_shape), self.replacement
                    ).reshape(n_shape)
            nj_score = self._pairwise(i, nj)
            loglik.append(F.logsigmoid(pos_score - nj_score))  # nsamp * bsz

            return (-torch.stack(loglik) * w).sum() / (
                len(loglik) * n_negatives * w.sum()
            )

        elif self.objective == "multiple_nrl":
            with torch.no_grad():
                nj = []
                for user in i:
                    neg = self.user_to_negs[user.item()].pop(0)
                    nj.append(neg)
                    self.user_to_negs[user.item()].append(neg)

            qid_emb = self.forward(self.i_to_ptr[i.ravel()]).reshape([*i.shape, -1])
            pos_emb = self.forward(self.j_to_ptr[j.ravel()]).reshape([*j.shape, -1])
            neg_emb = self.forward(self.j_to_ptr[nj]).reshape([*j.shape, -1])

            qid_emb = torch.nn.functional.normalize(qid_emb, p=2, dim=1)
            pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
            neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=1)
            pos_socres = torch.mm(qid_emb, pos_emb.transpose(0, 1))
            neg_scores = torch.mm(qid_emb, neg_emb.transpose(0, 1))
            labels = torch.tensor(
                range(len(pos_socres)), dtype=torch.long, device=pos_socres.device
            )
            scores = torch.cat((pos_socres, neg_scores), dim=1) * 20.0
            loss = torch.nn.CrossEntropyLoss()(scores, labels)

            return loss

    def compute_user_to_negatives(self):
        self.user_to_negs = dict()
        all_indices = self.tr_prior_score._indices()
        all_values = self.tr_prior_score._values()
        for step in range(len(all_values)):
            index = all_indices[:, step]
            user_id = index[0].item()
            neg = index[1].item()
            if user_id not in self.user_to_negs:
                self.user_to_negs[user_id] = []
            if all_values[step] >= 1.0:
                self.user_to_negs[user_id].append(neg)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.do_validation:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay
            )
            lr_scheduler = _ReduceLRLoadCkpt(
                optimizer, model=self, factor=0.25, patience=4, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val_epoch_loss",
                },
            }
        else:
            return torch.optim.Adagrad(
                self.parameters(), eps=1e-3, lr=self.lr, weight_decay=self.weight_decay
            )

    def to_explainer(self, **kw):
        return self.item_tower.to_explainer(**kw)


class _DataModule(LightningDataModule):
    def __init__(
        self,
        rime_dataset,
        item_index=None,
        item_tokenized=None,
        do_validation=None,
        batch_size=None,
        valid_batch_size=None,
    ):
        super().__init__()
        self._D = rime_dataset
        self._item_tokenized = item_tokenized
        self._do_validation = do_validation
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size
        self._num_batches = self._D.target_csr.nnz / self._batch_size

        item_to_ptr = {k: ptr for ptr, k in enumerate(item_index)}
        self.i_to_ptr = [
            item_to_ptr[hist[0]] for hist in self._D.user_in_test["_hist_items"]
        ]
        self.j_to_ptr = [item_to_ptr[item] for item in self._D.item_in_test.index]
        self.i_to_item_id = np.array(item_index)[self.i_to_ptr]
        self.j_to_item_id = np.array(item_index)[self.j_to_ptr]
        self.training_data = {
            "i_to_ptr": self.i_to_ptr,
            "j_to_ptr": self.j_to_ptr,
            "prior_score": self._D.prior_score,
            "item_freq": self._D.item_in_test["_hist_len"].values,
        }

    def setup(self, stage):  # auto-call by trainer
        if stage == "fit":
            target_coo = self._D.target_csr.tocoo()
            dataset = np.transpose([target_coo.row, target_coo.col, target_coo.data])
            self._num_workers = (len(dataset) > 1e4) * 4

            if self._do_validation:
                self._train_set, self._valid_set = default_random_split(dataset)
            else:
                self._train_set, self._valid_set = dataset, None
            print("train_set size", len(self._train_set))

    def train_dataloader(self):
        return DataLoader(
            self._train_set,
            self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        if self._do_validation:
            return DataLoader(
                self._valid_set, self._valid_batch_size, num_workers=self._num_workers
            )


class BertBPR:
    def __init__(
        self,
        item_df,
        freeze_bert=0,
        batch_size=None,
        model_name="bert-base-uncased",
        max_length=128,
        max_epochs=10,
        max_steps=-1,
        do_validation=None,
        strategy=None,
        query_item_position_in_user_history=0,
        **kw,
    ):
        if batch_size is None:
            batch_size = 10000 if freeze_bert > 0 else 10
        if do_validation is None:
            do_validation = max_epochs > 1
        if strategy is None:
            strategy = "dp" if torch.cuda.device_count() > 1 else None

        self.item_titles = item_df["TITLE"]
        self.max_length = max_length
        self.batch_size = batch_size
        self.do_validation = do_validation
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.strategy = strategy

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer_kw = dict(
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        self.all_inputs = self.tokenizer(self.item_titles.tolist(), **self.tokenizer_kw)
        self._model_kw = {
            "model_name": model_name,
            "freeze_bert": freeze_bert,
            "do_validation": do_validation,
            **kw,
            "tokenizer": self.tokenizer,
            "tokenizer_kw": self.tokenizer_kw,
        }

        self.model = _BertBPR(self.all_inputs, **self._model_kw)
        self.valid_batch_size = (
            self.batch_size * self.model.n_negatives * 2 // self.model.valid_n_negatives
        )

        self._ckpt_dirpath = []
        self._logger = TensorBoardLogger("logs", "BertBPR")
        self._logger.log_hyperparams(
            {
                k: v
                for k, v in locals().items()
                if k
                in [
                    "freeze_bert",
                    "batch_size",
                    "max_epochs",
                    "max_steps",
                    "sample_with_prior",
                    "sample_with_posterior",
                ]
            }
        )
        print(f"BertBPR logs at {self._logger.log_dir}")

    def _get_data_module(self, V):
        return _DataModule(
            V,
            self.item_titles.index,
            self.all_inputs,
            self.do_validation,
            self.batch_size,
            self.valid_batch_size,
        )

    def _transform_item_corpus(self, item_tower, output_step):
        with _device_mode_context(item_tower) as item_tower:
            ds = Dataset.from_pandas(self.item_titles.to_frame("text"))
            out = ds.map(
                item_tower.to_map_fn("text", output_step), batched=True, batch_size=64
            )[output_step]
            return np.vstack(out)

    @empty_cache_on_exit
    def fit(self, V=None, _lr_find=False):
        if V is None or not any(
            [param.requires_grad for param in self.model.parameters()]
        ):
            return self
        model = _BertBPR(self.all_inputs, **self._model_kw)
        dm = self._get_data_module(V)
        model.set_training_data(**dm.training_data)
        trainer = Trainer(
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            gpus=torch.cuda.device_count(),
            strategy=self.strategy,
            log_every_n_steps=1,
            callbacks=[model._checkpoint, LearningRateMonitor()],
            precision="bf16" if torch.cuda.is_available() else 32,
        )

        if self._model_kw["freeze_bert"] > 0:  # cache all_cls
            all_cls = self._transform_item_corpus(model.item_tower, "cls")
            model.register_buffer("all_cls", torch.as_tensor(all_cls), False)

        if _lr_find:
            lr_finder = trainer.tuner.lr_find(
                model, datamodule=dm, min_lr=1e-4, early_stop_threshold=None
            )
            fig = lr_finder.plot(suggest=True)
            fig.show()
            return lr_finder, lr_finder.suggestion()

        trainer.fit(model, datamodule=dm)
        model._load_best_checkpoint("best")

        if not os.path.exists(model._checkpoint.dirpath):  # add manual checkpoint
            print("model.load_state_dict(torch.load(...), strict=False)")
            print(f"{model._checkpoint.dirpath}/state-dict.pth")
            os.makedirs(model._checkpoint.dirpath)
            torch.save(
                model.state_dict(), model._checkpoint.dirpath + "/state-dict.pth"
            )

        self._logger.experiment.add_text(
            "ckpt", model._checkpoint.dirpath, len(self._ckpt_dirpath)
        )
        self._ckpt_dirpath.append(model._checkpoint.dirpath)
        self.model = model
        return self

    def get_all_embeddings(self, model, batch_size, output_step="embedding"):
        all_texts = self.item_titles.tolist()
        num = len(all_texts)
        num_batches = int(np.ceil(num / batch_size))
        embeddings_all = torch.zeros(num, 768)
        with torch.no_grad():
            for step in range(num_batches):
                if (step % 100) == 0:
                    print("Processing", step, "|", num_batches)
                text_batch = all_texts[step * batch_size : (step + 1) * batch_size]
                tokens = self.tokenizer(text_batch, **self.tokenizer_kw)
                if output_step == "embedding":
                    embedding_batch = model(**tokens, output_step="embedding")
                elif output_step == "mean":
                    embedding_batch, _ = model(**tokens, output_step="return_mean_std")
                embeddings_all[
                    step * batch_size : (step * batch_size + len(text_batch)), :
                ] = embedding_batch.cpu()
        return embeddings_all

    def cos_sim(self, a: torch.Tensor, b: torch.Tensor):
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    @empty_cache_on_exit
    @torch.no_grad()
    def transform(self, D):
        dm = self._get_data_module(D)
        BATCH_SIZE_ = 256 * max(1, torch.cuda.device_count())
        _gpu_ids = [i for i in range(torch.cuda.device_count())]
        model_ = self.model.item_tower
        model_.eval()
        model_ = torch.nn.DataParallel(model_, device_ids=_gpu_ids)
        if len(_gpu_ids) != 0:
            model_ = model_.cuda(_gpu_ids[0])
        num_of_users = len(dm.i_to_ptr)
        num_of_items = len(dm.j_to_ptr)
        score_matrix = torch.zeros(num_of_users, num_of_items)
        num_user_batches = int(np.ceil(num_of_users / BATCH_SIZE_))
        num_item_batches = int(np.ceil(num_of_items / BATCH_SIZE_))
        if hasattr(self, "oracle_dir"):
            qrels = self.oracle_dir
            item_to_ptr_list = self.item_titles.index.to_list()
            for step, user_index in enumerate(dm.i_to_ptr):
                qid = item_to_ptr_list[user_index].split("_")[-1]
                pids = list(qrels[qid])
                pids = ["p_{}".format(pid) for pid in pids]
                pid_indices = [item_to_ptr_list.index(pid) for pid in pids]
                score_matrix[step, pid_indices] += 1.0

        elif hasattr(self, "random"):
            score_matrix = torch.rand_like(score_matrix)

        elif hasattr(self.model.item_tower, "cls_model") or hasattr(
            self.model.item_tower, "ae_model"
        ):
            if hasattr(self.model.item_tower, "cls_model"):
                all_emb = self.get_all_embeddings(model_, BATCH_SIZE_, "embedding")
            elif hasattr(self.model.item_tower, "ae_model"):
                all_emb = self.get_all_embeddings(model_, BATCH_SIZE_, "mean")
            user_embedding = all_emb[dm.i_to_ptr]
            user_embedding = user_embedding.to(auto_device())
            for step_p in range(num_item_batches):
                item_ids = dm.j_to_ptr[
                    step_p * BATCH_SIZE_ : (step_p + 1) * BATCH_SIZE_
                ]
                item_embeddings_batch = all_emb[item_ids]
                item_embeddings_batch = item_embeddings_batch.to(auto_device())
                num_of_items = item_embeddings_batch.shape[0]
                scores = self.cos_sim(user_embedding, item_embeddings_batch)
                scores = scores.cpu()
                score_matrix[
                    :, step_p * BATCH_SIZE_ : (step_p * BATCH_SIZE_ + num_of_items)
                ] = scores

        else:
            NotImplementedError("NOT IMPLEMENTED!")
        return auto_cast_lazy_score(score_matrix)

    def to_explainer(self, **kw):
        return self.model.to_explainer(**kw)


def bbpr_main(
    item_df,
    expl_response,
    gnd_response,
    max_epochs=50,
    batch_size=10 * max(1, torch.cuda.device_count()),
    alpha=0.05,
    beta=0.0,
    user_df=None,
    train_kw={},
):
    """
    item_df = get_item_df()[0]
    expl_response = pd.read_json(
        'vae-1000-queries-10-steps-response.json', lines=True, convert_dates=False
    ).set_index('level_0')
    gnd_response = pd.read_json(
        'prime-pantry-i2i-online-baseline4-response.json', lines=True, convert_dates=False
    ).set_index('level_0')
    """
    V = create_reranking_dataset(user_df, item_df, expl_response, reranking_prior=1)
    assert V.target_csr.nnz > 0

    bbpr = BertBPR(
        item_df,
        max_epochs=max_epochs,
        batch_size=batch_size,
        sample_with_prior=True,
        sample_with_posterior=0,
        replacement=False,
        n_negatives=5,
        valid_n_negatives=5,
        training_prior_fcn=lambda x: (x + 1 / x.shape[1]).clip(0, None).log(),
        **train_kw,
    )
    bbpr.fit(V)

    gnd = create_reranking_dataset(user_df, item_df, gnd_response, reranking_prior=1e5)
    reranking_scores = bbpr.transform(gnd) + gnd.prior_score
    metrics = rime.metrics.evaluate_item_rec(gnd.target_csr, reranking_scores, 1)

    return metrics, reranking_scores, bbpr
