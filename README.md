## Crowd Coachable Recommendations / Retrieval (CCR)

Contents:

* `appendix.pdf` provides the appendix to the submitted paper.
* `al_demo_prime_pantry.ipynb` povides a notebook template to run oracle-labeled active learning experiments on a small-scale dataset.
* `al_demo_nq.ipynb` provides oracle-labeled experiments on the larger-scale natural questions dataset.

The only change between the two notebooks is `DATA_NAME="nq"` in the configuration line. One may also change it to `DATA_NAME="msmarco"` for the larger-scale MS-MARCO oracle-labeled experiments.

Crowd-sourcing experiments are run by calling `scripts/al_0_rank.py`, `scripts/al_1_em.py`, `scripts/al_2_ft.py` sequentially for each batch of the labeling tasks.
Human feedback is provided between al_0_rank and al_1_em by uploading `request_perm.csv` and downloading `human_response.csv` in the same data-step folder.

All experiments are conducted using NVidia A10G GPU machines with 4-GPU parallelization. Runtime is usually less than 3 hours per active learning step. Human tasks are usually completed under 45 minutes.

## Contributing

* Write access is managed. Please use pull requests to introduce any changes.
* Automated tests will be run against every pull request / push commit. If any of them fail, please check the log for an ssh location where you may log in and debug.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

