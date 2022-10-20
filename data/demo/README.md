Demo data
---
* `item_df.csv`: The only file required for zero-shot similar-item recommendations. This is to be used with `create_zero_shot` function to create a dataset where the same set of items are treated as seeds for similar-item queries.
* `user_df.json`: A collection of all queries. This file should be indexed by `USER_ID`. Include `user_df` in `create_zero_shot` function if you want to give custom indices to the queries for convenience.
* `test_response.json`: The basic form of data that is returned from crowd-sourcing sessions. The `USER_ID` field must be consistent with `user_df` and the `request_time` field must be larger than the `TEST_START_TIME` column in the user table.
* `expl_response.json`: Crowd-sourced data for active/semi-supervised learning. Same format as `test_response` but used for training by `BertMT` algorithm.

Usage:
* See `test/test_demo_data.py` for a variety of models for unsupervised and semi-supervised learning.
* See `ml_1m.py` where we construct similar demo data from Movielens dataset. The example runs the models in rime (recurrent-intensity-model-experiments) package for sequential recommendation baselines.
