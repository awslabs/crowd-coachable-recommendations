Anonymous submission

Provide a template to run oracle-labeled active learning experiments via `al_prime_pantry-demo.ipynb`.
This demo requires amazon review dataset to be downloaded from:
http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Prime_Pantry.json.gz

Crowd-sourcing experiments are run by calling `scripts/al_0_rank.py`, `scripts/al_1_em.py`, `scripts/al_2_ft.py` sequentially for each batch of labeling tasks.
Human feedback is provided between al_0_rank and al_1_em by uploading `request_perm.csv` and downloading `human_response.csv` in the same data folder.
