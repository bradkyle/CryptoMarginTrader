
import pyarrow.parquet as pq
import os

d = './data/stat/'

files = os.listdir(d)

for f in files:
    df = pq.read_table(d+f).to_pandas()
    df.sort_index(inplace=True)

    if set(['close_price', 'high_price', 'volume']).issubset(df.columns):
        print(df.columns)