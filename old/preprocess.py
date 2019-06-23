import pandas as pd
import numpy as np
import random as random
import pyarrow as pa 
import pyarrow.parquet as pq
from fastavro import writer, reader, parse_schema
import collections
import ta
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os

d = "./data/aggregated/"
files = os.listdir(d)

clean_dir = "./data/clean/"
stat_dir = "./data/stat/"

def add_candle_indicators(
    df, 
    l, 
    ck, 
    hk, 
    lk, 
    vk
):
    df[l+'rsi'] = ta.rsi(df[ck])
    df[l+'mfi'] = ta.money_flow_index(df[hk], df[lk], df[ck], df[vk])
    df[l+'tsi'] = ta.tsi(df[ck])
    df[l+'uo'] = ta.uo(df[hk], df[lk], df[ck])
    df[l+'ao'] = ta.ao(df[hk], df[lk])
    df[l+'macd_diff'] = ta.macd_diff(df[ck])
    df[l+'vortex_pos'] = ta.vortex_indicator_pos(df[hk], df[lk], df[ck])
    df[l+'vortex_neg'] = ta.vortex_indicator_neg(df[hk], df[lk], df[ck])
    df[l+'vortex_diff'] = abs(df[l+'vortex_pos'] - df[l+'vortex_neg'])
    df[l+'trix'] = ta.trix(df[ck])
    df[l+'mass_index'] = ta.mass_index(df[hk], df[lk])
    df[l+'cci'] = ta.cci(df[hk], df[lk], df[ck])
    df[l+'dpo'] = ta.dpo(df[ck])
    df[l+'kst'] = ta.kst(df[ck])
    df[l+'kst_sig'] = ta.kst_sig(df[ck])
    df[l+'kst_diff'] = (df[l+'kst']-df[l+'kst_sig'])
    df[l+'aroon_up'] = ta.aroon_up(df[ck])
    df[l+'aroon_down'] = ta.aroon_down(df[ck])
    df[l+'aroon_ind'] = (df[l+'aroon_up']-df[l+'aroon_down'])
    df[l+'bbh'] = ta.bollinger_hband(df[ck])
    df[l+'bbl'] = ta.bollinger_lband(df[ck])
    df[l+'bbm'] = ta.bollinger_mavg(df[ck])
    df[l+'bbhi'] = ta.bollinger_hband_indicator(df[ck])
    df[l+'bbli'] = ta.bollinger_lband_indicator(df[ck])
    df[l+'kchi'] = ta.keltner_channel_hband_indicator(df[hk],df[lk],df[ck])
    df[l+'kcli'] = ta.keltner_channel_lband_indicator(df[hk],df[lk],df[ck])
    df[l+'dchi'] = ta.donchian_channel_hband_indicator(df[ck])
    df[l+'dcli'] = ta.donchian_channel_lband_indicator(df[ck])
    df[l+'adi'] = ta.acc_dist_index(df[hk],df[lk],df[ck],df[vk])
    df[l+'obv'] = ta.on_balance_volume(df[ck], df[vk])
    df[l+'cmf'] = ta.chaikin_money_flow(df[hk],df[lk],df[ck],df[vk])
    df[l+'fi'] = ta.force_index(df[ck], df[vk])
    df[l+'em'] = ta.ease_of_movement(df[hk], df[lk], df[ck], df[vk])
    df[l+'vpt'] = ta.volume_price_trend(df[ck], df[vk])
    df[l+'nvi'] = ta.negative_volume_index(df[ck], df[vk])
    df[l+'dr'] = ta.daily_return(df[ck])
    df[l+'dlr'] = ta.daily_log_return(df[ck])
    return df

for f in files:

    name = os.path.splitext(f)[0]

    records = []
    with open(d+f, 'rb') as fo:
        for record in reader(fo):
            records.append(record)

    df = pd.DataFrame(records)

    df.set_index(['window_end'], inplace=True)
    df.sort_index(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    df = add_candle_indicators(
        df,
        l='all_', 
        ck='all_close_price', 
        hk='all_high_price', 
        lk='all_low_price', 
        vk='all_volume'
    )

    df = add_candle_indicators(
        df,
        l='buy_', 
        ck='buy_close_price', 
        hk='buy_high_price', 
        lk='buy_low_price', 
        vk='buy_volume'
    )

    df = add_candle_indicators(
        df,
        l='sell_', 
        ck='sell_close_price', 
        hk='sell_high_price', 
        lk='sell_low_price', 
        vk='sell_volume'
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
        
    metcol = [
        'interval',
        'quote_asset', 
        'base_asset',
        'exchange'
    ]

    indcol = metcol + [
        'all_close_price', 
        'all_volume', 
        'all_high_price'
    ]

    indf = df[indcol]

    indf.rename(
        columns={
            "all_close_price": "close_price", 
            "all_volume":"volume", 
            "all_high_price": "high_price"
        }, 
        inplace=True
    )

    df.drop(
        columns=metcol, 
        inplace=True
    )

    def log_and_difference(df, columns):
        transformed_df = df.copy()
        transformed_df[df.eq(0)] = 1E-10
        for column in columns:
            x = np.log(transformed_df[column])
            y = np.log(transformed_df[column]).shift(1)
            transformed_df[column] = x - y
        transformed_df = transformed_df.fillna(method='bfill').fillna(method='ffill')
        return transformed_df

    sdf = log_and_difference(df, df.columns.values)

    fdf = sdf.merge(
        indf, 
        how='outer', 
        left_index=True, 
        right_index=True
    )

    fdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    fdf.fillna(method='bfill', inplace=True)
    fdf.fillna(method='ffill', inplace=True)

    table = pa.Table.from_pandas(fdf)
    pq.write_table(table, stat_dir+name+".parquet")