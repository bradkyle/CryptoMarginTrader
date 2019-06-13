import gym
import pandas as pd
import numpy as np
from numpy import inf
from gym import spaces
from sklearn import preprocessing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, calmar_ratio, omega_ratio

from render.MarginTradingGraph import MarginTradingGraph
from util.stationarization import log_and_difference
from util.benchmarks import buy_and_hodl, rsi_divergence, sma_crossover
from util.indicators import add_indicators


# Delete this if debugging
np.warnings.filterwarnings('ignore')


class MarginTradingEnv(gym.Env):
    '''A Bitcoin trading environment for OpenAI gym'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(
        self, 
        df, 
        initial_balance=1, 
        commission=0.0025, 
        reward_func='sortino', 
        close_key='close',
        date_key='timestamp_ms',
        features=['open', 'high', 'low', 'close', 'base_volume', 'quote_volume'],
        annualization=365*24*60,
        max_leverage=1,
        **kwargs
    ):
        super(MarginTradingEnv, self).__init__()

        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_func = reward_func
        self.annualization = annualization
        self.max_leverage = max_leverage
        self.close_key = close_key
        self.date_key = date_key
        self.position=0.1

        self.df = df.fillna(method='bfill').reset_index()
        self.stationary_df = log_and_difference(
            self.df, features
        )

        benchmarks = kwargs.get('benchmarks', [])
        self.benchmarks = [
            {
                'label': 'Buy and HODL',
                'values': buy_and_hodl(self.df[self.close_key], initial_balance, commission)
            },
            {
                'label': 'RSI Divergence',
                'values': rsi_divergence(self.df[self.close_key], initial_balance, commission)
            },
            {
                'label': 'SMA Crossover',
                'values': sma_crossover(self.df[self.close_key], initial_balance, commission)
            },
            *benchmarks,
        ]

        self.forecast_len = kwargs.get('forecast_len', 10)
        self.confidence_interval = kwargs.get('confidence_interval', 0.95)
        self.obs_shape = (1, 5 + len(self.df.columns) - 2 + (self.forecast_len * 3))

        # Actions of the format -1=Short 1=Long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.float16
        )

    def _next_observation(self):
        scaler = preprocessing.MinMaxScaler()

        features = self.stationary_df[self.stationary_df.columns.difference(['index', self.date_key])]

        scaled = features[:self.current_step + self.forecast_len + 1].values
        scaled[abs(scaled) == inf] = 0
        scaled = scaler.fit_transform(scaled.astype('float32'))
        scaled = pd.DataFrame(scaled, columns=features.columns)

        obs = scaled.values[-1]

        past_df = self.stationary_df[self.close_key][:self.current_step + self.forecast_len + 1]
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

    # TODO
    def _current_price(self):
        return self.df[self.close_key].values[self.current_step + self.forecast_len] #WTF

    def _take_action(self, action):

        dist = action[0]
        current_price = self._current_price()
        threshold = 0.5

        # Long Base: Quote + Leverage > Base (All value stored in base)
        # Value should be moved from the quote asset into the base
        # asset
        # TODO commission + decay
        # self.position_type = "long"

        total_quote = (self.quote_held+(self.base_held*current_price)) 
        total_base = total_quote/current_price

        max_quote = total_quote * self.max_leverage
        max_base = max_quote/current_price

        # dist = 0.6

        next_base_debt = 0
        next_quote_debt = 0
        
        print("total value base: "+str(total_base))
        print("total value quote: "+str(total_quote))
        print("next base held: "+str(next_base))
        print("next quote held: "+str(next_quote))
        print("next base debt: "+str(next_base_debt))
        print("next quote debt: "+str(next_quote_debt))
        
        self.base_held = next_base
        self.quote_held = next_quote
        self.quote_debt = next_quote_debt
        self.base_debt = next_base_debt
        self.total_value_base = total_base
        self.total_value_quote = total_quote

        self.net_worths.append(total_quote)

        base_sold = 0
        quote_sold = 0
        base_loaned = 0
        quote_loaned = 0
        cost = 0

        #TODO
        # self.account_history = np.append(self.account_history, [
        #     [self.base_held],
        #     [self.quote_held],
        #     [self.base_debt],
        #     [self.quote_debt],
        #     [base_sold],
        #     [quote_sold],
        #     [base_loaned],
        #     [quote_loaned],
        #     [cost]
        # ], axis=1)

    # todo change to margin
    def _reward(self):
        length = min(self.current_step, self.forecast_len)
        returns = np.diff(self.net_worths[-length:])

        if np.count_nonzero(returns) < 1:
            return 0

        if self.reward_func == 'sortino':
            reward = sortino_ratio(
                returns, annualization=self.annualization
            )
        elif self.reward_func == 'calmar':
            reward = calmar_ratio(
                returns, annualization=self.annualization
            )
        elif self.reward_func == 'omega':
            reward = omega_ratio(
                returns, annualization=self.annualization
            )
        else:
            reward = returns[-1]

        return reward if np.isfinite(reward) else 0

    # TODO
    def _done(self):
        # TODO
        # return self.net_worths[-1] < self.initial_balance / 10 or self.current_step == len(self.df) - self.forecast_len - 1
        return False

    def reset(self):
        self.quote_held = self.initial_balance
        self.base_held = 0
        self.net_worths = [self.initial_balance]
        self.current_step = 0
        self.base_debt = 0
        self.quote_debt = 0
        self.position_type="short"

        self.account_history = np.array([
            [self.initial_balance],
            [0],
            [0],
            [0],
            [0]
        ])
        self.trades = []
        self.loans = []
        self.repayments = []

        return self._next_observation()

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        obs = self._next_observation()
        reward = self._reward()
        done = self._done()

        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'system':
            print('Price: ' + str(self._current_price()))
            print('Bought: ' + str(self.account_history[2][self.current_step]))
            print('Sold: ' + str(self.account_history[4][self.current_step]))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = MarginTradingGraph(self.df)

            self.viewer.render(
                self.current_step, 
                self.net_worths, 
                self.benchmarks, 
                self.trades,
                self.loans,
                self.repayments,
                self.position,
                self.position_type
            )

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
