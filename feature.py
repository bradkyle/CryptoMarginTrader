import pandas as pd
import numpy as np
import random as random
import pyarrow as pa 
import pyarrow.parquet as pq
from fastavro import writer, reader, parse_schema
import collections
import ta
from sklearn.ensemble import RandomForestRegressor
import os
import json

d = "./data/clean/"
files = os.listdir(d)
results = []

for f in files:
    df = pq.read_table(d+f).to_pandas()
    df.sort_index(inplace=True)

    metcol = [
        'interval',
        'quote_asset', 
        'base_asset',
        'exchange'
    ]

    df['all_close_price'] = df['all_close_price'].shift(-1)
    df.drop(columns=metcol, inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    x = df.drop(columns=['all_close_price'])
    y = df[['all_close_price']]

    rf = RandomForestRegressor(n_estimators=250)
    rf.fit(
        x.values,
        y.values
    )

    fp = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x.columns), reverse=True)

    results.append(fp)
    print(pd.DataFrame(fp))

with open("fea.txt", "w") as f:
    json.dump(results, f)

# table = pa.Table.from_pandas(pd.DataFrame(results))
# pq.write_table(table, "./data/results.parquet")