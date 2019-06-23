import gym
import optuna
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import random

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2, DDPG
from stable_baselines.ddpg.policies import LnMlpPolicy, LnCnnPolicy, CnnPolicy

from env.MarginTradingEnv import MarginTradingEnv
from util.read import read_parquet_df

div = 200
m_type = "ppo"
curr_idx = -1
reward_strategy = 'sortino'
params_db_file = 'sqlite:///params.db'

study_name = 'ppo2_' + reward_strategy
study = optuna.load_study(
    study_name=study_name, 
    storage=params_db_file
)
params = study.best_trial.params

print("Training PPO2 agent with params:", params)
print("Best trial reward:", -1 * study.best_trial.value)


def create_environment(df, index, div=30):
    p_len = int(len(df)/div)

    start = int(p_len*index)
    end = int(start+p_len)

    next_df = df[start:end]

    print(start)
    print(end)

    train_env = DummyVecEnv([lambda: MarginTradingEnv(
        next_df, 
        reward_func=reward_strategy, 
        forecast_len=int(60), 
        confidence_interval=params['confidence_interval'])
    ])

    return train_env, p_len

trained = []
trainable = np.arange(0, 100, 1)

def pick(trainable):
    idx = random.choice(trainable)
    trainable = np.delete(trainable, np.argwhere(trainable == idx))
    print(idx)
    return trainable, idx

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate': params['learning_rate']/100,
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam'],
}


train_env, train_len = create_environment(train_df, i, div)

if m_type == "ppo":
    if curr_idx == -1:
        model = PPO2(
            MlpLnLstmPolicy, 
            train_env, 
            verbose=0, 
            nminibatches=1,
            tensorboard_log="./tensorboard", 
            **model_params
        )
    else:
        model = PPO2.load('./agents/'+m_type+'_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=train_env)

else:
    if curr_idx == -1:
        model = DDPG(
            CnnPolicy, 
            train_env, 
            verbose=1, 
            tensorboard_log="./tensorboard"
        )

    else:
        model = DDPG.load('./agents/'+m_type+'_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=train_env)

for idx in range(curr_idx + 1, div):

    print('[', idx, '] Training for: ', train_len, ' time steps')

    model.learn(total_timesteps=train_len)
    model.save('./agents/'+m_type+'_' + reward_strategy + '_' + str(idx) + '.pkl')

    trainable, i = pick(trainable)

    train_env, train_len = create_environment(train_df, i, div)

    model.set_env(train_env)

test_env, test_len = create_environment(test_df, 0, 2)

obs, done = test_env.reset(), False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = test_env.step(action)

    test_env.render(mode="human")