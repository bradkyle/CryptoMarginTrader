import gym
import optuna
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import random
from random import shuffle

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2, DDPG
from stable_baselines.ddpg.policies import LnMlpPolicy, LnCnnPolicy, CnnPolicy

from env.MarginTradingEnv import MarginTradingEnv

# Config
# =============================================================>

l = 600
m_type = "ppo"
curr_idx = -1
reward_strategy = 'sortino'
params_db_file = 'sqlite:///params.db'
continuous_actions = False
commission=0.005

# Load Data
# =============================================================>

df = pq.read_table('./data/clean.parquet').to_pandas()
df.sort_values(['window_end'], inplace=True)

dfs = [df[int(x*l):int(x*l+l)] for x in range(int(len(df)/l))]

train_dfs = dfs[:int(len(dfs)*0.8)]
test_dfs = dfs[len(train_dfs):]

# Shuffle training data
shuffle(train_dfs,random.random)


# Load Optuna Study
# =============================================================>

study_name = 'ppo2_discrete_' + reward_strategy
study = optuna.load_study(
    study_name=study_name, 
    storage=params_db_file
)
params = study.best_trial.params

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate': params['learning_rate']/100,
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam'],
}

print("Training PPO2 agent with params:", params)
print("Best trial reward:", -1 * study.best_trial.value)

# Train Agent
# =============================================================>

print(train_dfs[0].head())

initial_env = DummyVecEnv([lambda: MarginTradingEnv(
    train_dfs[-1], 
    reward_func=reward_strategy, 
    forecast_len=int(100), 
    confidence_interval=params['confidence_interval'],
    continuous=continuous_actions,
    commission=commission
),
])

if m_type == "ppo":
    if curr_idx == -1:
        model = PPO2(
            MlpLnLstmPolicy, 
            initial_env, 
            verbose=0, 
            nminibatches=1,
            tensorboard_log="./tensorboard", 
            **model_params
        )
    else:
        model = PPO2.load('./agents/'+m_type+'_' + reward_strategy + '_' + str(curr_idx) + '.0.pkl', env=initial_env)
else:
    model = DDPG(
        LnMlpPolicy, 
        initial_env, 
        verbose=1, 
        tensorboard_log="./tensorboard"
    )

for i, df in enumerate(train_dfs):

    model.learn(total_timesteps=l)

    if i%10==0:
        model.save('./agents/'+m_type+'_' + reward_strategy + '_' + str(i/10) + '.pkl')

    model.set_env(DummyVecEnv([lambda: MarginTradingEnv(
        df, 
        reward_func=reward_strategy, 
        forecast_len=int(100), 
        confidence_interval=params['confidence_interval'],
        continuous=continuous_actions,
        commission=commission
    )
    ]))

# Train Agent
# =============================================================>
print("Commencing Test")


test_env = DummyVecEnv([lambda: MarginTradingEnv(
    test_dfs[0], 
    reward_func=reward_strategy, 
    forecast_len=int(100), 
    confidence_interval=params['confidence_interval'],
    commission=0.00,
    continuous=continuous_actions
)
])

obs, done = test_env.reset(), False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = test_env.step(action)

    test_env.render(mode="human")