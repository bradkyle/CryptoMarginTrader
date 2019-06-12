import ta


def add_indicators(df, close_key="Close", high_key="High", low_key="Low", volume_key="Volume BTC"):
    df['RSI'] = ta.rsi(df[close_key])
    df['MFI'] = ta.money_flow_index(
        df[high_key], df[low_key], df[close_key], df[volume_key])
    df['TSI'] = ta.tsi(df[close_key])
    df['UO'] = ta.uo(df[high_key], df[low_key], df[close_key])
    df['AO'] = ta.ao(df[high_key], df[low_key])

    df['MACD_diff'] = ta.macd_diff(df[close_key])
    df['Vortex_pos'] = ta.vortex_indicator_pos(
        df[high_key], df[low_key], df[close_key])
    df['Vortex_neg'] = ta.vortex_indicator_neg(
        df[high_key], df[low_key], df[close_key])
    df['Vortex_diff'] = abs(
        df['Vortex_pos'] -
        df['Vortex_neg'])
    df['Trix'] = ta.trix(df[close_key])
    df['Mass_index'] = ta.mass_index(df[high_key], df[low_key])
    df['CCI'] = ta.cci(df[high_key], df[low_key], df[close_key])
    df['DPO'] = ta.dpo(df[close_key])
    df['KST'] = ta.kst(df[close_key])
    df['KST_sig'] = ta.kst_sig(df[close_key])
    df['KST_diff'] = (
        df['KST'] -
        df['KST_sig'])
    df['Aroon_up'] = ta.aroon_up(df[close_key])
    df['Aroon_down'] = ta.aroon_down(df[close_key])
    df['Aroon_ind'] = (
        df['Aroon_up'] -
        df['Aroon_down']
    )

    df['BBH'] = ta.bollinger_hband(df[close_key])
    df['BBL'] = ta.bollinger_lband(df[close_key])
    df['BBM'] = ta.bollinger_mavg(df[close_key])
    df['BBHI'] = ta.bollinger_hband_indicator(
        df[close_key])
    df['BBLI'] = ta.bollinger_lband_indicator(
        df[close_key])
    df['KCHI'] = ta.keltner_channel_hband_indicator(df[high_key],
                                                    df[low_key],
                                                    df[close_key])
    df['KCLI'] = ta.keltner_channel_lband_indicator(df[high_key],
                                                    df[low_key],
                                                    df[close_key])
    df['DCHI'] = ta.donchian_channel_hband_indicator(df[close_key])
    df['DCLI'] = ta.donchian_channel_lband_indicator(df[close_key])

    df['ADI'] = ta.acc_dist_index(df[high_key],
                                  df[low_key],
                                  df[close_key],
                                  df[volume_key])
    df['OBV'] = ta.on_balance_volume(df[close_key],
                                     df[volume_key])
    df['CMF'] = ta.chaikin_money_flow(df[high_key],
                                      df[low_key],
                                      df[close_key],
                                      df[volume_key])
    df['FI'] = ta.force_index(df[close_key],
                              df[volume_key])
    df['EM'] = ta.ease_of_movement(df[high_key],
                                   df[low_key],
                                   df[close_key],
                                   df[volume_key])
    df['VPT'] = ta.volume_price_trend(df[close_key],
                                      df[volume_key])
    df['NVI'] = ta.negative_volume_index(df[close_key],
                                         df[volume_key])

    df['DR'] = ta.daily_return(df[close_key])
    df['DLR'] = ta.daily_log_return(df[close_key])

    df.fillna(method='bfill', inplace=True)

    return df
