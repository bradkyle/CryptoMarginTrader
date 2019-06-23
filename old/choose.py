import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd 
import numpy as np

d = "./data/"
files = os.listdir(d)

res = []

for f in files:
    df = pq.read_table(d+f).to_pandas()[-30000:]
    numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field])

    std = df[['close']].std(axis=0).values[0]
    volume = df[['quote_asset_volume']].sum().values[0] 
    e = {"file": f, "res": std*volume}
    print(e)
    res.append(e)

s = sorted(res, key=lambda x: x["res"])

print(s)