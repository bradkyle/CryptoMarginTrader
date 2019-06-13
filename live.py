


params_db_file = 'sqlite:///params.db'
reward_strategy = 'sortino'

study_name = 'ppo2_' + reward_strategy
study = optuna.load_study(study_name=study_name, storage=params_db_file)
params = study.best_trial.params

print("Testing PPO2 agent with params:", params)
print("Best trial:", -1 * study.best_trial.value)

live_env = DummyVecEnv([lambda: LiveMarginEnv(
    reward_func=reward_strategy,
    api_key="",
    api_secret="",
    forecast_len=int(params['forecast_len']), 
    confidence_interval=params['confidence_interval'])
])

model_params = {
    'n_steps': int(params['n_steps']),
    'gamma': params['gamma'],
    'learning_rate': params['learning_rate'],
    'ent_coef': params['ent_coef'],
    'cliprange': params['cliprange'],
    'noptepochs': int(params['noptepochs']),
    'lam': params['lam'],
}

model = PPO2.load(
    './agents/ppo2_' + reward_strategy + '_' + str(curr_idx) + '.pkl', 
    env=live_env
)

def main():
    
    # Subscribe to source
    action = model.predict()
    live_env.execute(action)

    pass

if __name__ == '__main__':
    main()