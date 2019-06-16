import pandas as pd
import numpy as np

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG

from env.MarginTradingEnv import MarginTradingEnv
from env.MarginTradingEnv import MarginTradingEnv
from util.read import read_parquet_df
import pyarrow.parquet as pq
import time

curr_idx = -1
reward_strategy = 'sortino'
df = pq.read_table('./data/clean.parquet').to_pandas()
df.sort_values(['window_end'], inplace=True)

train_env_1 = DummyVecEnv([lambda: MarginTradingEnv(
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

    step = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = train_env.step(action)
        reward_sum += reward

        # train_env.render(mode="human")
        # step += 1
        # if step>=100:
        #     time.sleep(60*100)

    print('[', idx, '] Total reward: ', reward_sum, ' (' + reward_strategy + ')')
    model.save('./agents/ppo2_' + reward_strategy + '_' + str(idx) + '.pkl')
