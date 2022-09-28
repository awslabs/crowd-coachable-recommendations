## Crowd Coachable Recommendations (CCRec)

![pytest workflow](https://github.com/awslabs/crowd-coachable-recommendations/actions/workflows/python-app.yml/badge.svg)

Codes for zero-shot recommendations and subsequent online learning and exploration with crowd-sourced preference labels.

## Getting Started

* Run `pip install -e .` for the ccrec package. This should also install part of the `recurrent-intensity-model-experiments` package as a dependency.
* If you encounter a `numba` error, please run this: `pip install --no-cache-dir --ignore-installed -U numba`
* To test: `from ccrec.util.demo_data import DemoData; DemoData().run_shap()`

## Data Preparation
CCRec considers both unsupervised and semi-supervised datasets for training purposes. Unsupervised datasets contain only item dataframes (`item_df`) and semi-supervised datasets contain `user_df` and `response_df` as well. The final dataset can be constructed through `ccrec.env.base.create_zero_shot` and `create_reranking_dataset` functions, respectively.
![ccrec_data.png](figure/ccrec_data.png)

## Contributing

* Write access is managed. Please use pull requests to introduce any changes.
* Automated tests will be run against every pull request / push commit. If any of them fail, please check the log for an ssh location where you may log in and debug.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

