# construct similar demo data from ml-1m dataset

import os
import pandas as pd, numpy as np
import rime
from ccrec.util.demo_data import DemoData

user_df = pd.read_csv('data/ml-1m/users.dat', sep='::', names=['USER_ID', '+1', '+2', '+3', '+4'])
user_df = user_df.set_index('USER_ID')

item_df = pd.read_csv('data/ml-1m/movies.dat', sep='::', encoding='latin1', names=['ITEM_ID', 'TITLE', '+1'])
item_df = item_df.set_index('ITEM_ID')

event_df = pd.read_csv('data/ml-1m/ratings.dat', sep='::', names=['USER_ID', 'ITEM_ID', 'VALUE', 'TIMESTAMP'])
event_df = event_df.sort_values('TIMESTAMP', kind='mergesort')

user_df['TEST_START_TIME'] = (0.9 * event_df.groupby('USER_ID')['TIMESTAMP'].max() +
                              0.1 * event_df.groupby('USER_ID')['TIMESTAMP'].min())  # implicit join on USER_ID

user_hist = user_df.join(
    event_df.set_index('USER_ID')
).query(
    'TIMESTAMP < TEST_START_TIME'
).groupby(level=0).agg(
    _hist_items=('ITEM_ID', list),
    _hist_ts=('TIMESTAMP', list),
)

user_target = user_df.join(
    event_df.set_index('USER_ID')
).query(
    'TIMESTAMP >= TEST_START_TIME'
).groupby(level=0).agg(
    cand_items=('ITEM_ID', list),
    multi_label=('VALUE', list),
)

demo_data_obj = DemoData(
    user_df=user_df.join(user_hist),
    item_df=item_df,
    gnd_response=user_df.join(user_target).set_index('TEST_START_TIME', append=True),
)

item_rec_metrics = demo_data_obj.run_rime(topk=25)
print(pd.DataFrame(item_rec_metrics).T)
