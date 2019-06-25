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
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pylab as pl

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
    df[l+'ma50'] = df[ck].rolling(window=50).mean()
    df[l+'ma100'] = df[ck].rolling(window=100).mean()    
    df[l+'26ema'] = df[[ck]].ewm(span=26).mean()
    df[l+'12ema'] = df[[ck]].ewm(span=12).mean()
    df[l+'macd'] = (df[l+'12ema']-df[l+'26ema'])
    df[l+'100sd'] = df[[ck]].rolling(100).std()
    df[l+'upper_band'] = df[l+'ma100'] + (df[l+'100sd']*2)
    df[l+'lower_band'] = df[l+'ma100'] - (df[l+'100sd']*2)
    df[l+'ema'] = df[ck].ewm(com=0.5).mean()
    df[l+'momentum'] = df[ck]-1
    return df

def log_and_difference(df, columns, cutoff=1500):
    transformed_df = df.copy()
    transformed_df[df.eq(0)] = 1E-10
    for column in columns:
        x = np.log(transformed_df[column])
        y = np.log(transformed_df[column]).shift(1)
        transformed_df[column] = x - y
    transformed_df = transformed_df.fillna(method='bfill').fillna(method='ffill')
    return transformed_df[cutoff:]

def apply_fastfracdiff(df, d=0.4):
    def fast_fracdiff(x, d):
        T = len(x)
        np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
        k = np.arange(1, T)
        b = (1,) + tuple(np.cumprod((k - d - 1) / k))
        z = (0,) * (np2 - T)
        z1 = b + z
        z2 = tuple(x) + z
        dx = pl.ifft(pl.fft(z1) * pl.fft(z2))
        return np.real(dx[0:T])

    for col in df.columns:
        try:
            df[col] = fast_fracdiff(
                np.array(df[col], dtype=np.float), 
                d=d
            )
        except Exception as e:
            print(df[[col]].head())
            raise e

def apply_fracdiff(df, d=0.4, thres=1e-5, cutoff=1500):
    def get_weight_ffd(d, thres, lim):
        w, k = [1.], 1
        ctr = 0
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
            ctr += 1
            if ctr == lim - 1:
                break
        w = np.array(w[::-1]).reshape(-1, 1)
        return w

    def frac_diff_ffd(x, d, thres=1e-5):
        w = get_weight_ffd(d, thres, len(x))
        width = len(w) - 1
        output = []
        output.extend([0] * width)
        for i in range(width, len(x)):
            output.append(np.dot(w.T, x[i - width:i + 1])[0])
        return np.array(output)
    
    for col in df.columns:
        try:
            df[col] = frac_diff_ffd(
                np.array(df[col], dtype=np.float), 
                d=d,
                thres=thres
            )
        except Exception as e:
            print(df[[col]].head())
            raise e

    return df[cutoff:]

def clean(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

def main_simple():

    df = pq.read_table("./data/raw/Binance_5m_ETHBTC.parquet").to_pandas()    
    clean(df)

    df.drop(columns=[
        'close_timestamp_ms', 
        'quote_asset_volume',
        'trades',
        'taker_buy_quote_asset_volume',
        'taker_buy_base_asset_volume',
        'open',
        'ignore'
    ], inplace=True)

    numeric_fields = ['high', 'low', 'close', 'volume']
    for f in numeric_fields:
            df[f] = pd.to_numeric(df[f])

    indf = df[['timestamp_ms', 'close','high','volume']]

    indf.rename(
        columns={
            "close": "close_price", 
            "volume":"all_volume", 
            "high": "high_price"
        }, 
        inplace=True
    )

    df.sort_values(['timestamp_ms'], inplace=True)

    df = add_candle_indicators(
        df,
        l='all_', 
        ck='close', 
        hk='high', 
        lk='low', 
        vk='volume'
    )

    df.set_index(['timestamp_ms'], inplace=True)
    df.sort_index(inplace=True)

    clean(df)
    df = apply_fracdiff(df)

    indf.set_index(['timestamp_ms'], inplace=True)
    indf.sort_index(inplace=True)

    df = df.merge(
        indf, 
        how='outer', 
        left_index=True, 
        right_index=True
    )

    clean(df)

    print(df.index)
    df.reset_index(inplace=True)
    print(df.columns)
    print(df.head())

    table = pa.Table.from_pandas(df)
    pq.write_table(table, "./data/clean/Binance_5m_ETHBTC_clean.parquet")


if __name__ == "__main__":
    main_simple()