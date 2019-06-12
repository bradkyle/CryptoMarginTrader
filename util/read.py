import pyarrow as pa 
import pyarrow.parquet as pq 
from util.indicators import add_indicators

def read_parquet_df(path, size=300000):
    df = pq.read_table(path).to_pandas()[:-size]
    print(df.describe())
    print(df.head())
    print("="*90)

    df.drop(['base_asset', 'ignore', 'timestamp_ms'], axis=1, inplace=True)

    numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for f in numeric_fields:
            df[f] = pd.to_numeric(df[f])

    df.rename(columns={'close_timestamp_ms':'timestamp_ms'}, inplace=True)
    df.sort_values(['timestamp_ms'], inplace=True)

    df = add_indicators(df.reset_index(), close_key="close", high_key="high", low_key="low", volume_key="volume")

    print("Table formatted")
    print("="*90)

    return df
