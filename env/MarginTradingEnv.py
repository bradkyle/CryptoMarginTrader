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
        close_key='close_price',
        date_key='timestamp_ms',
        annualization=365*24*60,
        max_leverage=4,
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

        self.df = df[['close_price']].reset_index()

        self.stationary_df = df.drop('close_price', axis=1).reset_index()

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

        self.forecast_len = kwargs.get('forecast_len', 200)
        self.confidence_interval = kwargs.get('confidence_interval', 0.95)
        self.obs_shape = (1, 8 + len(self.stationary_df.columns) + (self.forecast_len * 3))

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

    # TODO
    def _current_price(self):
        return self.df[self.close_key].values[self.current_step + self.forecast_len] #WTF

    def _take_action(self, action):

        dist = action[0]
        current_price = self._current_price()

        # Get the total representative value in quote asset
        total_quote = (self.quote_held+(self.base_held*current_price))

        # Find the distribution of value in denominations 
        # representative of the quote asset
        b_dist, q_dist, nex_b, nex_q, lev_q, lev_b = self._balance_dist(
            dist, 
            total_quote,
            max_leverage=self.max_leverage,
            threshold=0.5
        )

        b=b_dist/current_price
        q=q_dist
        sales = 0
        cost = 0
        base_sold = 0
        base_bought = 0

        # TODO simulate loss 
        # TODO simulate gain

        if self.base_held > b:
            base_sold = self.base_held - b

            self.trades.append({
                'step': self.current_step,
                'quantity': base_sold, 
                'price': current_price,
                'type': 'sell'
            })

            price = current_price * (1 - self.commission)
            sales =  base_sold * price

            self.base_held -= base_sold
            self.quote_held += sales
            
        else:
            base_bought = b - self.base_held

            self.trades.append({
                'step': self.current_step,
                'quantity': base_bought, 
                'price': current_price,
                'type': 'buy'
            })

            price = current_price * (1 + self.commission)
            cost =  base_bought * price

            self.base_held += base_bought
            self.quote_held -= cost

        self.base_debt = lev_b
        self.quote_debt = lev_q

        net_worth = (self.quote_held + self.base_held*current_price) - (self.quote_debt+self.base_debt*current_price)

        self.net_worths.append(net_worth)

        print("="*80)
        print("step: "+str(self.current_step))
        print("net worth: "+str(net_worth))
        print("current_price: "+str(current_price))
        print("action: "+str(action))
        print("quote held: "+str(self.quote_held))
        print("base held: "+str(self.base_held))
        print("quote debt: "+str(self.quote_debt))
        print("base debt: "+str(self.base_debt))
        print("base sold: "+str(base_sold))
        print("base bought: "+str(base_bought))
        print("cost: "+str(cost))
        print("sales: "+str(sales))
        print("nex_b: "+ str(nex_b))
        print("nex_q: "+ str(nex_q))
        print("lev_b: "+ str(lev_b))
        print("lev_q: "+ str(lev_q))
        print("b: "+ str(b))
        print("q: "+ str(q))
        print("done: "+str(self._done()))
        print("reward: " +(str(self._reward())))
        print("="*80)

        if net_worth < 0:
            raise ValueError("Net worth cant be less than 0.5")

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

    def _balance_dist(
        self,
        action,
        total_q,
        max_leverage=2,
        threshold=-0.5
    ):
        max_allowed = (total_q*max_leverage)

        next_q = total_q*(threshold-action)
        next_b = total_q*(threshold+action)
        
        next_q = clip(total_q, next_q)
        next_b = clip(total_q, next_b)
        
        lev_b = ((action-threshold)*2) * max_allowed
        lev_q =  -((action+threshold)*2) * max_allowed
        
        lev_b = clip(max_allowed, lev_b)
        lev_q = clip(max_allowed, lev_q)

        q = lev_q+next_q
        b = lev_b+next_b

        return (
            b,
            q,
            next_b,
            next_q,
            lev_b,
            lev_q
        )

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
        return self.net_worths[-1] < self.initial_balance / 10 or self.current_step == len(self.df) - self.forecast_len - 1

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

def clip(bal, scal):
    if scal < 0:
        scal = 0 
    elif scal > bal:
        scal = bal
    return scal