import pandas as pd
import numpy as np
import random as random
import pyarrow as pa 
import pyarrow.parquet as pq
from fastavro import writer, reader, parse_schema
import collections
import ta

records = []
with open('okex_spot_ETH_BTC_60.avro', 'rb') as fo:
    for record in reader(fo):
        records.append(record)

df = pd.DataFrame(records)

features = [
 'all_count',
 'all_close_price',
 'all_high_price',
 'all_mean_price',
 'all_low_price',
 'all_std_price',
 'all_close_quantity',
 'all_high_quantity',
 'all_mean_quantity',
 'all_low_quantity',
 'all_std_quantity',
 'buy_count',
 'buy_close_price',
 'buy_high_price',
 'buy_mean_price',
 'buy_low_price',
 'buy_std_price',
 'buy_close_quantity',
 'buy_high_quantity',
 'buy_mean_quantity',
 'buy_low_quantity',
 'buy_std_quantity',
 'sell_count',
 'sell_close_price',
 'sell_high_price',
 'sell_mean_price',
 'sell_low_price',
 'sell_std_price',
 'sell_close_quantity',
 'sell_high_quantity',
 'sell_mean_quantity',
 'sell_low_quantity',
 'sell_std_quantity',
 'ask_0_count',
 'ask_0_close_price',
 'ask_0_high_price',
 'ask_0_mean_price',
 'ask_0_low_price',
 'ask_0_std_price',
 'ask_0_close_quantity',
 'ask_0_high_quantity',
 'ask_0_mean_quantity',
 'ask_0_low_quantity',
 'ask_0_std_quantity',
 'ask_1_count',
 'ask_1_close_price',
 'ask_1_high_price',
 'ask_1_mean_price',
 'ask_1_low_price',
 'ask_1_std_price',
 'ask_1_close_quantity',
 'ask_1_high_quantity',
 'ask_1_mean_quantity',
 'ask_1_low_quantity',
 'ask_1_std_quantity',
 'ask_2_count',
 'ask_2_close_price',
 'ask_2_high_price',
 'ask_2_mean_price',
 'ask_2_low_price',
 'ask_2_std_price',
 'ask_2_close_quantity',
 'ask_2_high_quantity',
 'ask_2_mean_quantity',
 'ask_2_low_quantity',
 'ask_2_std_quantity',
 'bid_0_count',
 'bid_0_close_price',
 'bid_0_high_price',
 'bid_0_mean_price',
 'bid_0_low_price',
 'bid_0_std_price',
 'bid_0_close_quantity',
 'bid_0_high_quantity',
 'bid_0_mean_quantity',
 'bid_0_low_quantity',
 'bid_0_std_quantity',
 'bid_1_count',
 'bid_1_close_price',
 'bid_1_high_price',
 'bid_1_mean_price',
 'bid_1_low_price',
 'bid_1_std_price',
 'bid_1_close_quantity',
 'bid_1_high_quantity',
 'bid_1_mean_quantity',
 'bid_1_low_quantity',
 'bid_1_std_quantity',
 'bid_2_count',
 'bid_2_close_price',
 'bid_2_high_price',
 'bid_2_mean_price',
 'bid_2_low_price',
 'bid_2_std_price',
 'bid_2_close_quantity',
 'bid_2_high_quantity',
 'bid_2_mean_quantity',
 'bid_2_low_quantity',
 'bid_2_std_quantity',
 'all_volume',
 'all_vwap',
 'buy_volume',
 'buy_vwap',
 'sell_volume',
 'sell_vwap'
]

df=df[features+['window_end']]
df['close_price'] = df['all_close_price']

df.sort_values(by ='window_end', inplace=True)
df.fillna(method='bfill', inplace=True)
df.set_index(['window_end'], inplace=True)

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

df.fillna(method='bfill', inplace=True)

def log_and_difference(df, columns):
    transformed_df = df.copy()
    for column in columns:
        transformed_df.loc[df[column] == 0] = 1E-10
        transformed_df[column] = np.log(transformed_df[column]) - np.log(transformed_df[column]).shift(1)
    transformed_df = transformed_df.fillna(method='bfill')
    return transformed_df

sdf = log_and_difference(df, features)
table = pa.Table.from_pandas(sdf)
pq.write_table(table, './data/t.parquet')