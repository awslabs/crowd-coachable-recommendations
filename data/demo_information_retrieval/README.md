Template for information retrieval where we also specify `ITEM_TYPE` and use `exclude_train=['ITEM_TYPE']` to separate queries and passages.

```
import pandas as pd
from ccrec.env.base import create_zero_shot
item_df = pd.read_csv('item_df.csv').set_index('ITEM_ID')
zero_shot = create_zero_shot(
	item_df, create_users_from=lambda item_df: item_df['ITEM_TYPE'] == 'query', exclude_train=['ITEM_TYPE'])
print(pd.DataFrame(
    zero_shot.prior_score.toarray(),
    index=zero_shot.user_df.index,
    columns=zero_shot.item_df.index,
))
```

See `test/test_information_retrieval.py` for a full example that also includes active learning.
