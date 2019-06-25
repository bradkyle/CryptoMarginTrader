'''

A large part of the code in this file was sourced from the rl-baselines-zoo library on GitHub.
In particular, the library provides a great parameter optimization set for the PPO2 algorithm,
as well as a great example implementation using optuna.

Source: https://github.com/araffin/rl-baselines-zoo/blob/master/utils/hyperparams_opt.py

'''

import optuna

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import random
from random import shuffle
from nets.CustomCnnPolicy import CustomCnnPolicy
from nets.CustomLstmPolicy import CustomLstmPolicy

from env.MarginTradingEnv import MarginTradingEnv

reward_strategy = 'sortino'

params_db_file = 'sqlite:///params.db'

# number of parallel jobs
n_jobs = 12
# maximum number of trials for finding the best hyperparams
n_trials = 1000
# number of evaluations for pruning per trial
n_evaluations = 6

# Config
# =============================================================>

train_len = 40000
m_type = "ppo"
curr_idx = -1
reward_strategy = 'sortino'
params_db_file = 'sqlite:///params.db'
continuous_actions = False

# Load Data
# =============================================================>

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

# Shuffle training data
shuffle(dfs,random.random)

def optimize_envs(trial):
    return {
        'initial_balance':1000, 
        'action_type': trial.suggest_categorical('action_type', ['continuous', 'discrete_20', 'discrete_5', 'discrete_3', 'discrete_nl_3']),
        'window_size': trial.suggest_categorical('window_size', [5, 10, 20, 40, 80])
    }

def optimize_policy(trial):
    return {
        'policy': trial.suggest_categorical('policy', ['MLPLNLSTM', 'CUSTOMCNN', 'CUSTOMLSTM'])
    }

def optimize_ppo2(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
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
            **env_params
        )]
    )
    return test_env, train_env

def optimize_agent(trial):
    env_params = optimize_envs(trial)
    test_env, train_env = create_envs(env_params, 0) 

    # policy_params = optimize_policy(trial)

    # pp = policy_params["policy"]

    # if pp == "MLPLNLSTM":
    #     policy = MlpLnLstmPolicy
    # elif pp == "CUSTOMCNN":
    #     policy = CustomCnnPolicy
    # elif pp == "CUSTOMLSTM":
    #     policy = CustomLstmPolicy
    policy = MlpLnLstmPolicy

    model_params = optimize_ppo2(trial)
    model = PPO2(
        policy, 
        train_env, 
        verbose=0, 
        nminibatches=1,
        tensorboard_log="./tensorboard", 
        **model_params
    )

    print("="*90)
    print(str(model_params))
    print(str(env_params))
    print("="*90)

    for eval_idx in range(n_evaluations-1):
        try:
            model.learn(total_timesteps=train_len)
        except AssertionError:
            raise

        reward_sum = 0.0
        done = False
        obs = test_env.reset()
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            reward_sum += reward

        trial.report(-1 * reward_sum, eval_idx)

        if trial.should_prune(eval_idx):
            raise optuna.structs.TrialPruned()

        eval_idx+=1
        test_env, train_env = create_envs(env_params, eval_idx)
        model.set_env(train_env)

    return -1 * reward_sum


def optimize():
    study_name = 'ppo2_' + reward_strategy
    study = optuna.create_study(
        study_name=study_name, storage=params_db_file, load_if_exists=True)

    try:
        study.optimize(optimize_agent, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


if __name__ == '__main__':
    optimize()
