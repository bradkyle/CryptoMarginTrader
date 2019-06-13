import pandas as pd
import numpy as np

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG

from env.MarginTradingEnv import MarginTradingEnv
from env.MarginTradingEnv import MarginTradingEnv
from util.read import read_parquet_df

curr_idx = -1
reward_strategy = 'sortino'
input_data_file = './data/Binance_1m_ETHBTC.parquet'
df = read_parquet_df(input_data_file, size=1000)

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

train_df = df[:train_len]
test_df = df[train_len:]

train_env = DummyVecEnv([lambda: MarginTradingEnv(
    train_df
)])

model = DDPG(
    LnMlpPolicy, 
    train_env, 
    verbose=1, 
    tensorboard_log="./tensorboard"
)

for idx in range(curr_idx + 1, 10):
    print('[', idx, '] Training for: ', train_len, ' time steps')

    # model.learn(total_timesteps=train_len)

    obs = train_env.reset()
    done, reward_sum = False, 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = train_env.step(action)
        reward_sum += reward

        # train_env.render(mode="human")

    print('[', idx, '] Total reward: ', reward_sum, ' (' + reward_strategy + ')')
    model.save('./agents/ppo2_' + reward_strategy + '_' + str(idx) + '.pkl')
