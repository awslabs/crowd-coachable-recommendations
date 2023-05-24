Anonymous submission to 37th Conference on Neural Information Processing Systems (NeurIPS 2023). Do not distribute.

Contents:

* `appendix.pdf` provides the appendix to the submitted paper.
* `al_demo_prime_pantry.ipynb` povides a notebook template to run oracle-labeled active learning experiments on a small-scale dataset.
* `al_demo_nq.ipynb` provides oracle-labeled experiments on the larger-scale natural questions dataset.

The only change between the two notebooks is `DATA_NAME="nq"` in the configuration line. One may also change it to `DATA_NAME="msmarco"` for the larger-scale MS-MARCO oracle-labeled experiments.

Crowd-sourcing experiments are run by calling `scripts/al_0_rank.py`, `scripts/al_1_em.py`, `scripts/al_2_ft.py` sequentially for each batch of the labeling tasks.
Human feedback is provided between al_0_rank and al_1_em by uploading `request_perm.csv` and downloading `human_response.csv` in the same data-step folder.
