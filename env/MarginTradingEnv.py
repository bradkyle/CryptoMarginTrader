import gym
import pandas as pd
import numpy as np
from numpy import inf
from gym import spaces
from sklearn import preprocessing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, roll_max_drawdown, alpha_beta, max_drawdown
from render.MarginTradingGraph import MarginTradingGraph
from util.benchmarks import buy_and_hodl, rsi_divergence, sma_crossover
from termcolor import colored
import csv 
from sklearn.preprocessing import scale

# Delete this if debugging
np.warnings.filterwarnings('ignore')
np.set_printoptions(edgeitems=300)
np.core.arrayprint._line_width = 800

class MarginTradingEnv(gym.Env):
    '''A Bitcoin trading environment for OpenAI gym'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(
        self, 
        df, 
        initial_balance=1, 
        commission=0.002, 
        close_key='close_price',
        date_key='timestamp_ms',
        annualization=365*24*12,
        max_leverage=4,
        training=True,
        max_steps=None,
        is_cnn=True,
        verbose=False,
        **kwargs
    ):
        super(MarginTradingEnv, self).__init__()

        self.initial_balance = initial_balance
        self.commission = commission
        self.annualization = annualization
        self.max_leverage = max_leverage
        self.close_key = close_key
        self.date_key = date_key
        self.position=0.1
        self.training=training
        self.max_steps = max_steps
        self.is_cnn = is_cnn
        self.verbose = verbose
       
        self.reward_func = kwargs.get('reward_func', 'sortino')
        self.window_size = kwargs.get('window_size', 20)
        self.account_history_size = kwargs.get('account_history_size', 3)
        self.action_type = kwargs.get('action_type', 'discrete_nl_3')
        self.obs_type = kwargs.get('obs_type', 2)


        # Actions of the format -1=Short 1=Long
        if self.action_type=="continuous":
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        elif self.action_type=="discrete_20":
            self.action_space = spaces.Discrete(20)
        elif self.action_type =="discrete_5":
            self.action_space = spaces.Discrete(5)
        elif self.action_type=="discrete_3":
            self.action_space = spaces.Discrete(3)
        elif self.action_type=="discrete_nl_3":
             self.action_space = spaces.Discrete(3)

        try:
            self.df = df[['close_price', 'high_price', 'all_volume']].reset_index()
            self.stationary_df = df.drop(columns=['close_price', 'high_price', 'all_volume'], axis=1).reset_index()
        except Exception as e:
            print(df.columns)
            print(e)

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

        self.obs_shape = (1, (8*self.account_history_size)+(len(self.stationary_df.columns)*self.window_size))

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(low=-1, high=1, shape=self.obs_shape, dtype=np.float16)
        self.scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

        
    def _next_observation(self):

        features = self.stationary_df[self.current_step: self.current_step + self.window_size].values
        account = np.transpose(self.account_history[:,-self.account_history_size:])

        # print(features)
        self.scaler.fit(features)
        features = self.scaler.transform(features)

        # print(account)
        self.scaler.fit(account)
        account = self.scaler.transform(account)

        obs = np.append(features, account)

        obs[obs==-np.inf] = 0
        obs[obs==np.nan] = 0
        obs[obs==np.inf] = 0
        
        obs = np.reshape(obs.astype('float16'), self.obs_shape)

        return obs     

    # TODO
    def _current_price(self):
        return self.df[self.close_key].values[self.current_step + self.window_size] #WTF
        
    def get_value(self):
        current_price = self._current_price()
        total_debt = (self.quote_debt+self.base_debt*current_price)
        total_value = (self.quote_held + self.base_held*current_price)
        total_value_minus_debt = round(total_value - total_debt, 6)
        return total_value, total_value_minus_debt, total_debt

    def derive_action(self, action):
        if self.action_type=="continuous":
            return round(action[0], 1)
        elif self.action_type=="discrete_20":
            return (action-10)/10
        elif self.action_type == "discrete_5":
            if action == 4:
                return 1
            elif action == 3:
                return 0.5
            elif action == 2:
                return 0
            elif action == 1:
                return -0.5
            elif action == 0:
                return 0
        elif self.action_type=="discrete_3":
            if action == 2:
                return 1
            elif action == 1:
                return 0
            elif action == 0:
                return -1
        elif self.action_type=="discrete_nl_3":
            if action == 2:
                return 0.5
            elif action == 1:
                return 0.0
            elif action == 0:
                return -0.5

    def _take_action(self, action):

        dist = self.derive_action(action)

        current_price = self._current_price()

        if self.quote_held <=0:
            self.base_held -= self.quote_held/current_price
            self.quote_held = 0
        elif self.base_held <0:
            self.quote_held -= self.base_held*current_price
            self.base_held = 0

        price = 0
        cost = 0
        sales = 0
        base_sold = 0
        base_bought = 0
        q_delta = 0
        b_delta = 0
        t_delta = 0
        lev_b_delta = 0
        lev_q_delta = 0
        
        if dist != self.prev_action:

            # Get the total representative value in quote asset
            total_quote = (self.quote_held+(self.base_held*current_price))-(self.quote_debt+(self.base_debt*current_price))

            # Find the distribution of value in denominations 
            # representative of the quote asset
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

            lev_q_delta = (lev_q - self.quote_debt)
            lev_b_delta = (lev_b - self.base_debt)*current_price
            
            # Sell / Short
            # if an action is taken then trades will occur
            if self.base_held >= b and self.quote_held <= q:
                self.side = "sell"
                q_delta = (q-self.quote_held)
                b_delta = (b - self.base_held)*current_price

                if abs(q_delta) > abs(b_delta):
                    t_delta = abs(q_delta)
                else:
                    t_delta = abs(b_delta)

                if not self.training:
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
                
                if not self.training:
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
                raise ValueError("Not Buy or Sell")
        
            self.base_debt = round(lev_b, 6)
            self.quote_debt = round(lev_q, 6)

        self.total_debt = (self.quote_debt+self.base_debt*current_price)
        self.total_value = (self.quote_held+self.base_held*current_price)
        self.total_value_minus_debt = round(self.total_value - self.total_debt, 6)

        self.net_worths.append(self.total_value_minus_debt)

        if self.verbose:
            print("="*80)
            print("step: "+str(self.current_step))
            reward = self._reward()
            if reward > 0:
                print(colored("net worth: "+str(self.total_value_minus_debt), 'green'))
            else:
                print(colored("net worth: "+str(self.total_value_minus_debt), 'red'))
            print("side: "+ self.side)
            print("current_price: "+str(current_price))
            print("price change: "+str(abs(round((current_price-self.prev_price)/self.prev_price, 6))))
            print("price: "+str(price))
            print("actual action"+str(action))
            print("action: "+str(dist))
            print("quote held: "+str(self.quote_held))
            print("base held: "+str(self.base_held))
            print("lev_b_delta: "+str(lev_b_delta))
            print("lev_q_delta: "+str(lev_q_delta))
            print("total debt: "+str(self.total_debt))
            print("quote debt: "+str(self.quote_debt))
            print("base debt: "+str(self.base_debt))
            print("base sold: "+str(base_sold))
            print("base bought: "+str(base_bought))
            print("cost: "+str(cost))
            print("sales: "+str(sales))
            print("q_delta: "+ str(q_delta))
            print("b_delta: "+ str(b_delta))
            print("t_delta: "+ str(t_delta))
            print("done: "+str(self._done()))
            print("reward: " +(str(reward)))
            print("="*80)
        else:
            print("-"*80)
            print("step: "+str(self.current_step))
            reward = self._reward()
            if reward > 0:
                print(colored("net worth: "+str(self.total_value_minus_debt), 'green'))
            else:
                print(colored("net worth: "+str(self.total_value_minus_debt), 'red'))
            print("reward: " +(str(reward)))
            print("-"*80)

        self.account_history = np.append(self.account_history, [
            [self.quote_held],
            [self.base_held*current_price],
            [self.base_debt*current_price],
            [self.quote_debt],
            [cost],
            [sales],
            [base_sold],
            [base_bought]
        ], axis=1)

        self.prev_price = current_price
        self.prev_action = dist

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
        length = min(self.current_step, self.window_size)
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
        elif self.reward_func == "logret":
            reward = np.log(returns[-1])
        else:
            reward = returns[-1]

        return reward if np.isfinite(reward) else 0

    # TODO ((total asset- debt )/debt≤10%)
    def _done(self):
        if self.max_steps is not None and self.current_step >= self.max_steps:
            return True
        return self.net_worths[-1] < self.initial_balance / 10 or self.current_step == len(self.df) - self.window_size - 1

    def reset(self):
        self.quote_held = self.initial_balance
        self.total_value_minus_debt=self.initial_balance
        self.base_held = 0
        self.net_worths = [self.initial_balance]
        self.current_step = 0
        self.base_debt = 0
        self.quote_debt = 0
        self.position_type="short"
        self.prev_price = self._current_price()
        self.prev_action = 0

        self.account_history = np.repeat([
            [self.initial_balance],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]
        ], self.window_size+1, axis=1)

        self.trades = []
        self.loans = []
        self.repayments = []
        self.side = "sell"

        return self._next_observation()

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        obs = self._next_observation()

        reward = self._reward()
        done = self._done()

        # info = {
        #     "net_worth": self.net_worths[-1],
        #     "net_worth_std": np.std(self.net_worths),
        #     "net_worth_mean": np.mean(self.net_worths),
        #     "num_losses": 0,
        #     "num_wins": 0,
        #     "exposure": 0,
        #     "annual_return": 0,
        #     "risk_adj_ret": 0,
        #     "average_cost": 0,
        #     "average_bars_held": 0,
        #     "max_consec_win": 0,
        #     "max_consec_loss": 0,
        #     "max_drawdown": 0,
        #     "recovery_factor":0,
        #     "carmaxdd": 0,
        #     "rarmaxdd": 0,
        #     "profitfac": 0,
        #     "ulcer": 0,
        #     "sharpe": 0,
        #     "kratio": 0,
        #     "commission": self.commission
        # }

        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'system':
            print('Price: ' + str(self._current_price()))
            print('Bought: ' + str(self.account_history[2][self.current_step]))
            print('Sold: ' + str(self.account_history[4][self.current_step]))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = MarginTradingGraph(self.df, self.account_history)

            self.viewer.render(
                self.current_step, 
                self.net_worths, 
                self.benchmarks, 
                self.trades,
                self.account_history
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

