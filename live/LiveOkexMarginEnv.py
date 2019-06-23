
import okex.spot_api as spot
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


class FifoObsBuffer():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

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

        self.df = pd.DataFrame()

        self.forecast_len = kwargs.get('forecast_len', 200)
        self.confidence_interval = kwargs.get('confidence_interval', 0.95)

        self.obs_shape = (1, 8 + len(self.stationary_df.columns) + (self.forecast_len * 3))

        # Actions of the format -1=Short 1=Long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)

    def process(self, obs):
        self.df.append(obs, ignore_index=True)

        obs = self._next_observation()
        action = self.model.predict(obs)

        # get current balance
        # get current price
        current_price = 0

        dist = round(action[0], 1)

        b_dist, q_dist, nex_b, nex_q, lev_q, lev_b = self._balance_dist(
            dist, 
            total_quote,
            max_leverage=self.max_leverage,
            threshold=0.5
        )

        b = round(b_dist/current_price, 6)
        nex_b = round(nex_b/current_price, 6)
        lev_b = round(lev_b/current_price, 6)
        q = q_dist

        price = 0
        cost = 0
        sales = 0
        base_sold = 0
        base_bought = 0
        q_delta = 0
        b_delta = 0
        t_delta = 0

        lev_q_delta = (lev_q - self.quote_debt)
        lev_b_delta = (lev_b - self.base_debt)*current_price

        if self.base_held >= b and self.quote_held <= q:
            self.side = "sell"
            q_delta = (q-self.quote_held)
            b_delta = (b - self.base_held)*current_price

            if abs(q_delta) > abs(b_delta):
                t_delta = abs(q_delta)
            else:
                t_delta = abs(b_delta)

            self.trades.append({
                'step': self.current_step,
                'quantity': q_delta/current_price, 
                'price': current_price,
                'type': 'sell'
            })

            exec_prob = 0
            cost =  t_delta * self.commission

            self.base_held = round(b, 6) 
            self.quote_held = round((q - cost), 6)

        # Buy / Long
        elif self.base_held <= b and self.quote_held >= q:
            self.side="buy"
            q_delta = (q-self.quote_held)/current_price
            b_delta = (b - self.base_held)
            t_delta = b_delta - q_delta

            if abs(q_delta) > abs(b_delta):
                t_delta = abs(q_delta)
            else:
                t_delta = abs(b_delta)
            
            self.trades.append({
                'step': self.current_step,
                'quantity': b_delta, 
                'price': current_price,
                'type': 'buy'
            })

            exec_prob = 0
            cost =  t_delta * self.commission

            self.base_held = round((b - cost), 6) 
            self.quote_held = round(q, 6)
        else:
            logging.error()

        self.net_worths.append(self.total_value_minus_debt)

        self.account_history = np.append(self.account_history, [
            [self.quote_held],
            [self.base_held],
            [self.base_debt],
            [self.quote_debt],
            [cost],
            [sales],
            [base_sold],
            [base_bought]
        ], axis=1)

        self.prev_price = current_price
    
    def run(self):
        pass
    

    def _next_observation(self):
        scaler = preprocessing.MinMaxScaler()

        features = self.stationary_df[self.stationary_df.columns.difference(['index', self.date_key])]

        scaled = features[:self.current_step + self.forecast_len + 1].values
        scaled[abs(scaled) == inf] = 0
        scaled = scaler.fit_transform(scaled.astype('float32'))
        scaled = pd.DataFrame(scaled, columns=features.columns)

        obs = scaled.values[-1]

        past_df = self.stationary_df['all_close_price'][:self.current_step + self.forecast_len + 1]
        forecast_model = SARIMAX(past_df.values, enforce_stationarity=False, simple_differencing=True)
        model_fit = forecast_model.fit(method='bfgs', disp=False)
        forecast = model_fit.get_forecast(
            steps=self.forecast_len, 
            alpha=(1 - self.confidence_interval)
        )

        obs = np.insert(obs, len(obs), forecast.predicted_mean, axis=0)
        obs = np.insert(obs, len(obs), forecast.conf_int().flatten(), axis=0)

        scaled_history = scaler.fit_transform(self.account_history.astype('float32'))

        obs = np.insert(obs, len(obs), scaled_history[:, -1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obs_shape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs
