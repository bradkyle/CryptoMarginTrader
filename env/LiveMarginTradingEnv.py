from lomond.persist import persist
from lomond import WebSocket
import logging
import json
import zlib
import dateutil.parser as dp
import requests
import hmac
import base64
import zlib
import pandas as pd
from sklearn import preprocessing

import okex.account_api as account
import okex.ett_api as ett
import okex.futures_api as future
import okex.lever_api as lever
import okex.spot_api as spot
import okex.swap_api as swap

class FifoObsBuffer():
    def __init__(
        self,
        window_size,
        feature_num
    ):
        self.window_size = window_size
        self.df = pd.DataFrame()

    def update(self):
        pass

    def get(self):
        pass

class LiveMarginTradingEnv(gym.Env):

    def __init__(
        self,
        model,
        quote_asset,
        base_asset,
        api_key,
        api_secret,
        passphrase,
        annualization=365*24*60,
        max_leverage=4,
        **kwargs
    ):
        super(LiveMarginTradingEnv, self).__init__()
        
        self.model = model
        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self.instrument_id = "-".join(self.base_asset, self.quote_asset)
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.spotAPI = spot.SpotAPI(api_key, api_secret, passphrase, True)
        self.leverAPI = lever.LeverAPI(api_key, api_secret, passphrase, True)

        self.window_size = window_size
        self.max_leverage = max_leverage
        self.reward_func = reward_func
        self.annualization = annualization


    def run(self):
        pass

    def process(self, obs):
        pass
    
    def _next_observation(self):
        pass
    
    def _get_account(self):
        result = self.leverAPI.get_specific_account(self.instrument_id)
        return result

    def _get_config_info(self):
        result = self.leverAPI.get_specific_config_info(self.instrument_id)
        return result

    def _borrow(self, currency, amount):
        result = self.leverAPI.borrow_coin(
            instrument_id=self.instrument_id, 
            currency=currency, 
            amount=amount
        )
        return result

    def _repay(self, currency, amount):
        result = self.leverAPI.repayment_coin(
            instrument_id=self.instrument_id, 
            currency=currency, 
            amount=amount
        )
        return result

    def _multiple_orders(self, orders):
        params = [
            {"client_oid":"20180728","instrument_id":"btc-usdt","side":"sell","type":"market"," size ":"0.001"," notional ":"10001","margin_trading ":"1"},
            {"client_oid":"20180728","instrument_id":"btc-usdt","side":"sell","type":"limit"," size ":"0.001","notional":"10002","margin_trading ":"1"}
        ]
        result = self.leverAPI.take_orders(params)
        return result

    def _revoke_orders(self, orders):
        params = [
        {"instrument_id":"btc-usdt","order_ids":[23464,23465]},
        {"instrument_id":"ltc-usdt","order_ids":[243464,234465]}
        ]
        result = self.leverAPI.revoke_orders(params)
        return result

    def _get_pending_orders(self, froms, to, limit):
        result = self.leverAPI.get_order_pending(self.instrument_id, froms, to, limit)
        return result

    def _get_order_info(self, oid):
        result = self.leverAPI.get_order_info(oid, self.instrument_id)
        return result