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

dfs = load([
    './data/raw/check.parquet'
], l=train_len)

shuffle(dfs,random.random)

train_dfs = dfs[:-1]
test_dfs = dfs[-1:]

# Environment setup 
# ========================================================>

train_env = DummyVecEnv([lambda: MarginTradingEnv(
    train_dfs[0], 
    reward_func=reward_strategy, 
    window_size=int(params['window_size']), 
    action_type=params['action_type'],
    initial_balance=initial_balance,
    commission=0.0
)])

# Model Setup
# ========================================================> 

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate':1e-4,
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam'],
}

if m_type == "ppo2":
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

elif m_type == "acktr":
    model = ACKTR(MlpLnLstmPolicy, train_env, verbose=1)

elif m_type == "acer":
    model = ACER(MlpLnLstmPolicy, train_env, verbose=1)

elif m_type == "ddpg":
    if curr_idx == -1:
        model = DDPG(
            CnnPolicy, 
            train_env, 
            verbose=1, 
            tensorboard_log="./tensorboard"
        )
    else:
        model = DDPG.load('./agents/'+m_type+'_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=train_env)

# Environment setup 
# ========================================================>

episode = 0
num = len(dfs)
upreg_interval = 2
upregs = 5
frac_comm = 0
max_comm = 0.003

for idx in range(curr_idx + 1, len(dfs)):
    episode += 1
    commission = round(min(frac_comm*(max_comm/upregs), max_comm), 4)
    print("Commission:"+str(commission))

    if episode%upreg_interval == 0:
        frac_comm +=1

    print('[', idx, '] Training for: ', train_len, ' time steps')

    model.learn(total_timesteps=train_len)
    model.save('./agents/'+m_type+'_' +action_type+'_'+ reward_strategy + '_' + str(idx) + '.pkl')

    if episode%2 == 0:
        done=False
        test_env = DummyVecEnv([lambda: MarginTradingEnv(
            test_dfs[0], 
            reward_func=reward_strategy, 
            window_size=int(params['window_size']), 
            action_type=params['action_type'],
            initial_balance=initial_balance,
            commission=0.0015,
            training=False,
            max_steps=300
        )])

        obs, done = test_env.reset(), False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            # test_env.render(mode="human")

    new_env = DummyVecEnv([lambda: MarginTradingEnv(
        train_dfs[idx], 
        reward_func=reward_strategy, 
        window_size=int(params['window_size']), 
        action_type=params['action_type'],
        initial_balance=initial_balance,
        commission=commission
    )])

    model.set_env(new_env)

