import gym
import optuna
import pandas as pd
import numpy as np

from stable_baselines.common.policies import LnMlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2, DDPG
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from env.MarginEnv import MarginEnv
from util.read import read_parquet_df

curr_idx = -1
reward_strategy = 'sortino'
input_data_file = './data/Binance_1m_ETHBTC.parquet'
params_db_file = 'sqlite:///params.db'

study_name = 'ddpg_' + reward_strategy
study = optuna.load_study(
    study_name=study_name, 
    storage=params_db_file
)
params = study.best_trial.params

print("Training PPO2 agent with params:", params)
print("Best trial reward:", -1 * study.best_trial.value)

df = read_parquet_df(input_data_file, size=400000)

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

train_df = df[:train_len]
test_df = df[train_len:]

train_env = DummyVecEnv([lambda: MarginEnv(
    train_df, 
    reward_func=reward_strategy, 
    forecast_len=int(params['forecast_len']), 
    confidence_interval=params['confidence_interval'])
])

test_env = DummyVecEnv([lambda: MarginEnv(
    test_df, 
    reward_func=reward_strategy, 
    forecast_len=int(params['forecast_len']), 
    confidence_interval=params['confidence_interval']
)])

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate': params['learning_rate'],
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam'],
}

if curr_idx == -1:

    n_actions = train_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=float(0.5) * np.ones(n_actions)
    )

    model = DDPG(
        LnMlpPolicy, 
        train_env,
        action_noise=action_noise,
        verbose=0, 
        nminibatches=1,
        tensorboard_log="./tensorboard", 
        **model_params
    )
else:
    model = DDPG.load('./agents/ddpg_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=train_env)

for idx in range(curr_idx + 1, 10):
    print('[', idx, '] Training for: ', train_len, ' time steps')

    model.learn(total_timesteps=train_len)

    obs = test_env.reset()
    done, reward_sum = False, 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        reward_sum += reward

    print('[', idx, '] Total reward: ', reward_sum, ' (' + reward_strategy + ')')
    model.save('./agents/ppo2_' + reward_strategy + '_' + str(idx) + '.pkl')
