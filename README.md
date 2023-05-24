Anonymous submission to 37th Conference on Neural Information Processing Systems (NeurIPS 2023). Do not distribute.

Contents:

* `appendix.pdf` provides the appendix to the submitted paper.
* `al_demo_prime_pantry.ipynb` povides a notebook template to run oracle-labeled active learning experiments.

The notebook can easily be modified with one-line change of `DATA_NAME="nq"` or `DATA_NAME="msmarco"` to run the larger-scale experiments in the paper.

Crowd-sourcing experiments are run by calling `scripts/al_0_rank.py`, `scripts/al_1_em.py`, `scripts/al_2_ft.py` sequentially for each batch of the labeling tasks.
Human feedback is provided between al_0_rank and al_1_em by uploading `request_perm.csv` and downloading `human_response.csv` in the same data-step folder.
