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

shuffle(dfs,random.random)

# Environment setup 
# ========================================================>

env_params = {

}

def create_envs(env_params, idx):
    df = dfs[idx]

    train_df = df[:int(len(df)*0.8)]
    test_df = df[int(len(train_df)):]

    train_env = DummyVecEnv(
        [lambda: MarginTradingEnv(
            train_df,
            **env_params
        )]
    )
    test_env = DummyVecEnv(
        [lambda: MarginTradingEnv(
            test_df, 
            verbose=True,
            **env_params
        )]
    )
    return test_env, train_env

test_env, train_env = create_envs(env_params, 0) 

# Model Setup
# ========================================================> 

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate':params["learning_rate"],
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam'],
}

if curr_idx == -1:
    model = PPO2(
        MlpLnLstmPolicy, 
        train_env, 
        verbose=0, 
        nminibatches=1,
        tensorboard_log="./tensorboard"
    )
else:
    model = PPO2.load('./agents/'+m_type+'_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=train_env)

# Environment setup 
# ========================================================>

episode = 0
num = len(dfs)
upreg_interval = 2
upregs = 5
frac_comm = 0
max_comm = 0.003

for idx in range(curr_idx + 1, len(dfs)):
    print('[', idx, '] Training for: ', train_len, ' time steps')

    model.learn(total_timesteps=train_len)
    model.save('./agents/'+m_type+'_' +action_type+'_'+ reward_strategy + '_' + str(idx) + '.pkl')

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = test_env.step(action)

    idx+=1
    test_env, train_env = create_envs(env_params, idx)
    model.set_env(train_env)



