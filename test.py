import gym
import optuna
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import random
from stable_baselines.common.policies import MlpLnLstmPolicy, CnnPolicy, MlpPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2, DDPG, ACER
from stable_baselines.ddpg.policies import LnMlpPolicy, LnCnnPolicy
from env.MarginTradingEnv import MarginTradingEnv
import random
from random import shuffle
import csv 

# Config
# ========================================================>

m_type = "ppo2"
curr_idx = -1
reward_strategy = "sortino"
action_type = 'discrete'
params_db_file = 'sqlite:///params.db'
train_len = 10000
continuous=False
initial_balance=1000

study_name = "ppo2"+'_'+action_type+'_'+"sortino"
study = optuna.load_study(
    study_name=study_name, 
    storage=params_db_file
)
params = study.best_trial.params

print("Training PPO2 agent with params:", params)
print("Best trial reward:", -1 * study.best_trial.value)

# Create Episodes
# ========================================================>

def load(files, l):
    dfs = []
    for f in files:
        df = pq.read_table(f).to_pandas()
        df.rename(
            columns={
                "timestamp_ms": "window_end"
            }, 
            inplace=True
        )
        df.set_index(['window_end'], inplace=True)
        df.sort_index(inplace=True)
        for x in range(int(len(df)/l)):
            dfs.append(df[int(x*l):int(x*l+l)])
    return dfs

dfs = load(['./data/clean/Binance_5m_ETHBTC.parquet'], l=train_len)


# Environment setup 
# ========================================================>

env_params = {

}

test_env = DummyVecEnv(
    [lambda: MarginTradingEnv(
        dfs[-1], 
        verbose=True,
        **env_params
    )]
)

# Load Model
# ========================================================>

model = PPO2.load('./agents/'+m_type+'_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=test_env)

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    test_env.render(mode="human")