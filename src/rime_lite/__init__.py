# try:
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(1, 1)); plt.plot(); plt.title("make sure plot shows"); plt.show()
# except ImportError:
#     pass

# import torch, dataclasses, warnings, json
# import pandas as pd, numpy as np
# from typing import Dict, List
# from rime.metrics import (evaluate_item_rec, evaluate_user_rec, evaluate_mtch)
# from rime import dataset
# from rime.dataset import Dataset, create_dataset_unbiased
# from rime.util import _argsort, cached_property, RandScore, plot_rec_results, plot_mtch_results, MissingModel

# from pkg_resources import get_distribution, DistributionNotFound
# try:
#     __version__ = get_distribution("recurrent-intensity-model-experiments").version
#     print("recurrent-intensity-model-experiments (rime)", __version__)
# except DistributionNotFound:
#     warnings.warn("rime version configuration issues in setuptools_scm")


# @dataclasses.dataclass
# class ExperimentResult:
#     dual: bool
#     online: bool
#     _k1: int
#     _c1: int
#     _kmax: int
#     _cmax: int

#     item_ppl_baseline: float = None
#     user_ppl_baseline: float = None

#     item_ppl: float = None  # Deprecated; will be removed
#     user_ppl: float = None  # Deprecated; will be removed

#     item_rec: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)
#     user_rec: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)
#     mtch_: Dict[str, List[Dict[str, float]]] = dataclasses.field(default_factory=dict)

#     def __post_init__(self):
#         if self.item_ppl_baseline is None:
#             warnings.warn("item_ppl -> item_ppl_baseline", DeprecationWarning)
#             self.item_ppl_baseline = self.item_ppl
#         if self.user_ppl_baseline is None:
#             warnings.warn("user_ppl -> user_ppl_baseline", DeprecationWarning)
#             self.user_ppl_baseline = self.user_ppl

#     def print_results(self):
#         print('\nitem_rec')
#         print(pd.DataFrame(self.item_rec).T)
#         print('\nuser_rec')
#         print(pd.DataFrame(self.user_rec).T)

#         mtch1 = self.get_mtch_(self._k1, self._c1)
#         if mtch1 is not None:
#             print('\nmtch_')
#             print(mtch1.T)

#     def save_results(self, fn):
#         with open(fn, 'w') as fp:
#             json.dump(dataclasses.asdict(self), fp)

#     def get_mtch_(self, k=None, c=None, name="mtch_"):
#         y = {}
#         for method, x in getattr(self, name).items():
#             x = pd.DataFrame(x)
#             if k is not None and c is not None:
#                 y[method] = x.set_index(['k', 'c']).loc[(k, c)]
#             elif k is not None:
#                 y[method] = x.set_index(['k', 'c']).loc[k].sort_index().T
#             elif c is not None:
#                 y[method] = x.set_index(['c', 'k']).loc[c].sort_index().T
#             else:
#                 raise ValueError("either k or c must be provided")
#         return pd.concat(y, axis=1) if len(y) else None


# class Experiment:
#     """ Produce item_rec / user_rec metrics;
#     then sweeps through multipliers for relevance-diversity curve,
#     interpreting mult<1 as item min-exposure and mult>=1 as user max-limit
#     """
#     def __init__(
#         self, D, V=None, *V_extra,
#         default_k_items_per_user=None,
#         default_c_users_per_item=None,
#         mult=[],  # [0, 0.1, 0.2, 0.5, 1, 3, 10, 30, 100],
#         models_to_run=None,
#         model_hyps={},
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         dual=False,
#         online=False,
#         tie_break=0,
#         cache=None,
#         results=None,
#         **mtch_kw
#     ):

#         self.D = D
#         self.V = V
#         self.V_extra = V_extra

#         self.mult = mult

#         if models_to_run is None:
#             models_to_run = self.registered.keys()
#         self.models_to_run = models_to_run

#         self.model_hyps = model_hyps
#         self.device = device

#         if online:
#             if not dual:
#                 warnings.warn("online requires dual, resetting dual to True")
#                 dual = True
#             assert V is not None, "online dual is trained with explicit valid_mat"

#         self.tie_break = tie_break
#         if cache is not None:
#             self.update_cache(cache)
#         self.mtch_kw = mtch_kw

#         _k1 = default_k_items_per_user if default_k_items_per_user is not None else \
#               D._k1 if hasattr(D, '_k1') else int(np.ceil(self.D.shape[1] * 0.01))  # 1% of all items
#         _c1 = default_c_users_per_item if default_c_users_per_item is not None else \
#               D._c1 if hasattr(D, '_c1') else int(np.ceil(self.D.shape[0] * 0.01))  # 1% of all users

#         if results is None:
#             results = ExperimentResult(
#                 dual, online,
#                 _k1=_k1,
#                 _c1=_c1,
#                 _kmax=len(self.D.item_in_test),
#                 _cmax=len(self.D.user_in_test),
#                 item_ppl_baseline=self.D.item_ppl_baseline,
#                 user_ppl_baseline=self.D.user_ppl_baseline,
#             )
#         self.results = results

#         # pass-through references
#         self.__dict__.update(self.results.__dict__)
#         self.print_results = self.results.print_results
#         self.get_mtch_ = self.results.get_mtch_

#     def metrics_update(self, name, S, T=None):
#         target_csr = self.D.target_csr
#         score_mat = S

#         if self.online:
#             valid_mat = T
#         elif self.dual:
#             valid_mat = score_mat
#         else:
#             valid_mat = None

#         if self._k1 > 0:
#             self.item_rec[name] = evaluate_item_rec(
#                 target_csr, score_mat, self._k1, device=self.device)
#         else:
#             self.item_rec[name] = None

#         if self._c1 > 0:
#             self.user_rec[name] = evaluate_user_rec(
#                 target_csr, score_mat, self._c1, device=self.device)
#         else:
#             self.user_rec[name] = None

#         print(pd.DataFrame({
#             'item_rec': self.item_rec[name],
#             'user_rec': self.user_rec[name],
#         }).T)

#         if len(self.mult):
#             self.mtch_[name] = self._mtch_update(target_csr, score_mat, valid_mat, name)

#     def _mtch_update(self, target_csr, score_mat, valid_mat, name):
#         """ assign user/item matches and return evaluation results.
#         """
#         confs = []
#         for m in self.mult:
#             if m < 1:
#                 # lower-bound is interpreted as item min-exposure
#                 confs.append((self._k1, self._c1 * m, 'lb'))
#             else:
#                 # upper-bound is interpreted as user max-limit
#                 confs.append((self._k1 * m, self._c1, 'ub'))

#         mtch_kw = self.mtch_kw.copy()
#         if self.dual:
#             mtch_kw['valid_mat'] = valid_mat
#             mtch_kw['prefix'] = f"{name}-{self.online}"
#         else:
#             mtch_kw['argsort_ij'] = _argsort(score_mat, device=self.device)

#         out = []
#         for k, c, constraint_type in confs:
#             res = evaluate_mtch(
#                 target_csr, score_mat, k, c, constraint_type=constraint_type,
#                 dual=self.dual, device=self.device,
#                 item_prior=1 + self.D.item_in_test['_hist_len'].values,
#                 **mtch_kw
#             )
#             res.update({'k': k, 'c': c})
#             out.append(res)

#         return out

#     @cached_property
#     def registered(self):
#         registered = {
#             "Rand": lambda D: Rand().transform(D),
#             "Pop": lambda D: self._pop.transform(D),
#             "EMA": lambda D: EMA(D.horizon).transform(D) * self._pop_item.transform(D),
#             "Hawkes": lambda D: self._hawkes.transform(D) * self._pop_item.transform(D),
#             "HP": lambda D: self._hawkes_poisson.transform(D) * self._pop_item.transform(D),

#             "RNN": lambda D: self._rnn.transform(D),
#             "RNN-Pop": lambda D: self._rnn.transform(D) * Pop(1, 0).transform(D),
#             "RNN-EMA": lambda D: self._rnn.transform(D) * EMA(D.horizon).transform(D),
#             "RNN-Hawkes": lambda D: self._rnn.transform(D) * self._hawkes.transform(D),
#             "RNN-HP": lambda D: self._rnn.transform(D) * self._hawkes_poisson.transform(D),

#             "Transformer": lambda D: self._transformer.transform(D),
#             "Transformer-Pop": lambda D: self._transformer.transform(D) * Pop(1, 0).transform(D),
#             "Transformer-EMA": lambda D: self._transformer.transform(D) * EMA(D.horizon).transform(D),
#             "Transformer-Hawkes": lambda D: self._transformer.transform(D) * self._hawkes.transform(D),
#             "Transformer-HP": lambda D: self._transformer.transform(D) * self._hawkes_poisson.transform(D),

#             "BPR-Item": lambda D: self._bpr_item.transform(D),
#             "BPR-User": lambda D: self._bpr_user.transform(D),
#             "BPR": lambda D: self._bpr.transform(D),

#             "GraphConv-Base": lambda D: self._graph_conv_base.transform(D),
#             "GraphConv-Extra": lambda D: self._graph_conv_extra.transform(D),

#             "LDA": lambda D: self._lda.transform(D),

#             "ALS": lambda D: self._als.transform(D),
#             "LogisticMF": lambda D: self._logistic_mf.transform(D),

#             "BayesLM-0": lambda D: self._bayes_lm_0.transform(D),
#             "BayesLM-1": lambda D: self._bayes_lm_1.transform(D),

#             "ItemKNN-0": lambda D: self._item_knn_0.transform(D),
#             "ItemKNN-1": lambda D: self._item_knn_1.transform(D),
#         }

#         for model in [LDA, ALS, LogisticMF, GraphConv, ItemKNN, BayesLM]:
#             if isinstance(model, MissingModel):
#                 keys = [k for k in registered.keys() if k.startswith(model.name)]
#                 for k in keys:
#                     warnings.warn(f"skipping {k} due to {model.err}")
#                     registered.pop(k, None)

#         # disable models due to missing inputs

#         if not ('TEST_START_TIME' in self.D.user_in_test and '_hist_ts' in self.D.user_in_test
#                 and 0 < self.D.horizon < float("inf")):
#             warnings.warn("disabling temporal models due to missing TEST_START_TIME, _hist_ts or horizon")
#             for model in ['EMA', 'Hawkes', 'HP', 'RNN-EMA', 'RNN-Hawkes', 'RNN-HP',
#                            'Transformer-EMA', 'Transformer-Hawkes', 'Transformer-HP']:
#                 registered.pop(model, None)

#         if self.V is None:
#             warnings.warn("disabling HP and GraphConv due to missing validation set")
#             for model in ['HP', 'RNN-HP', 'Transformer-HP',
#                            'GraphConv-Base', 'GraphConv-Extra']:
#                 registered.pop(model, None)

#         if 'TITLE' not in self.D.item_df:
#             warnings.warn("disabling zero-shot models due to missing item TITLE")
#             for model in ['BayesLM-0', 'BayesLM-1', 'ItemKNN-0', 'ItemKNN-1']:
#                 registered.pop(model, None)

#         return registered

#     def _validate_run_input(self, models_to_run):
#         """ return a dictionary of {model_name: model_str or model_obj} """
#         if models_to_run is None:
#             models_to_run = [m for m in self.models_to_run if m not in
#                              ['BayesLM-0', 'BayesLM-1', 'ItemKNN-0', 'ItemKNN-1']]
#         if isinstance(models_to_run, str):
#             models_to_run = [models_to_run]
#         if isinstance(models_to_run, list):
#             for model in models_to_run:
#                 assert model in self.registered, f"{model} disabled or unregistered"
#             print("models to run", models_to_run)
#             models_to_run = {k: k for k in models_to_run}
#         return models_to_run

#     def run(self, models_to_run=None):
#         """ models_to_exclude is ignored if models_to_run is explicitly provided """

#         models_to_run = self._validate_run_input(models_to_run)

#         for model_name, model_obj in models_to_run.items():
#             print("running", model_name)
#             if isinstance(model_obj, str):
#                 transform_fn = self.registered[model_obj]
#             else:
#                 transform_fn = model_obj.transform

#             S = transform_fn(self.D)
#             if self.D.prior_score is not None:
#                 S = S + self.D.prior_score
#             if self.tie_break:
#                 warnings.warn("Using experimental RandScore class")
#                 S = S + RandScore.create(S.shape) * self.tie_break

#             if self.online:
#                 V = self.V.reindex(self.D.item_in_test.index, axis=1)
#                 T = transform_fn(V)
#                 if V.prior_score is not None:
#                     T = T + V.prior_score
#                 if self.tie_break:
#                     warnings.warn("Using experimental RandScore class")
#                     T = T + RandScore.create(T.shape) * self.tie_break
#             else:
#                 T = None
#             self.metrics_update(model_name, S, T)

#     @cached_property
#     def _pop(self):
#         return Pop().fit(self.D.auto_regressive)

#     @cached_property
#     def _pop_item(self):
#         return Pop(user_rec=False, item_rec=True).fit(self.D.auto_regressive)

#     @cached_property
#     def _rnn(self):
#         return RNN(
#             self.D.item_df, **self.model_hyps.get("RNN", {})
#         ).fit(self.D.auto_regressive)

#     @cached_property
#     def _transformer(self):
#         return Transformer(
#             self.D.item_df, **self.model_hyps.get("Transformer", {})
#         ).fit(self.D.auto_regressive)

#     @cached_property
#     def _hawkes(self):
#         return Hawkes(self.D.horizon).fit(self.D.auto_regressive)

#     @cached_property
#     def _hawkes_poisson(self):
#         assert self.V is not None, "_hawkes_poisson requires self.V"
#         return HawkesPoisson(self._hawkes).fit(self.V)

#     @cached_property
#     def _bpr_item(self):
#         return LightFM_BPR(item_rec=True).fit(self.D.auto_regressive)

#     @cached_property
#     def _bpr_user(self):
#         return LightFM_BPR(user_rec=True).fit(self.D.auto_regressive)

#     @cached_property
#     def _bpr(self):
#         return BPR(**self.model_hyps.get("BPR", {})).fit(self.D.auto_regressive)

#     @cached_property
#     def _graph_conv_base(self):
#         assert self.V is not None, "_graph_conv_base requires self.V"
#         return GraphConv(
#             self.D, **self.model_hyps.get("GraphConv-Base", {})
#         ).fit(self.V)

#     @cached_property
#     def _graph_conv_extra(self):
#         if len(self.V_extra) == 0:
#             warnings.warn("w/o V_extra, GraphConv-Extra will perform the same as GraphConv-Base")
#         return GraphConv(
#             self.D, **self.model_hyps.get("GraphConv-Extra", {})
#         ).fit(self.V, *self.V_extra)

#     @cached_property
#     def _lda(self):
#         return LDA(
#             self.D.auto_regressive, **self.model_hyps.get("LDA", {})
#         ).fit(self.D.auto_regressive)

#     @cached_property
#     def _als(self):
#         return ALS().fit(self.D.auto_regressive)

#     @cached_property
#     def _logistic_mf(self):
#         return LogisticMF().fit(self.D.auto_regressive)

#     @cached_property
#     def _bayes_lm_0(self):
#         return BayesLM(self.D.item_df, item_pop_power=0,
#                         **self.model_hyps.get("BayesLM-0", {}))

#     @cached_property
#     def _bayes_lm_1(self):
#         return BayesLM(self.D.item_df, item_pop_power=1,
#                         **self.model_hyps.get("BayesLM-1", {}))

#     @cached_property
#     def _item_knn_0(self):
#         return ItemKNN(self.D.item_df, item_pop_power=0,
#                         **self.model_hyps.get("ItemKNN-0", {}))

#     @cached_property
#     def _item_knn_1(self):
#         return ItemKNN(self.D.item_df, item_pop_power=1,
#                         **self.model_hyps.get("ItemKNN-1", {}))

#     def update_cache(self, other):
#         for attr in ['registered', '_transformer', '_rnn', '_hawkes', '_hawkes_poisson',
#                      '_bpr_item', '_bpr_user', '_als', '_logistic_mf',
#                      '_bpr', '_graph_conv_base', '_graph_conv_extra', '_lda']:
#             if attr in other.__dict__:
#                 setattr(self, attr, getattr(other, attr))


# def main(name, *args, **kw):
#     prepare_fn = getattr(dataset, name)
#     D, *V = prepare_fn(*args)
#     self = Experiment(D, *V, **kw)
#     self.run()
#     self.results.print_results()
#     return self
