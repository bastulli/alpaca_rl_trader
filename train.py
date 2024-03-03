import time
import os
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from torch.utils.tensorboard import SummaryWriter

# Custom imports
from environment.trading_game_env import TradingGameEnv
from data.data_loader import load_data, SplitOption
from util.neural_network import CustomNetwork
from util.custom_eval_callback import EvalCallback

# Configuration parameters and hyperparameters
FRAMESTACK = 5  # Number of stacked frames (history)s
BATCH_SIZE = 64
DENSE_LAYER_SIZE = 64
LEARNING_RATE = 0.0003
TOTAL_TIMESTEPS = 1000000

features = [
    'f_month_high', 'f_month_low',
    'f_half_month_high', 'f_half_month_low',
    'f_week_high', 'f_week_low',
    'f_day_high', 'f_day_low',
    'f_hour_high', 'f_hour_low',
    'f_price_ema_1', 'f_price_ema_2', 'f_price_ema_3'
]

crypto_symbols = ['BAT', 'BCH', 'BTC', 'CRV',
                  'DOT', 'ETH', 'GRT', 'LINK', 'LTC', 'MKR', 'UNI', 'XTZ']

# Using list comprehension to modify each element in the list
symbols = ['X:' + symbol + 'USD' for symbol in crypto_symbols]

table_name = 'crypto_data_hourly'

# Add unrealized_pl_array, holdings_array
NUM_FEATURES = len(features) + 3
NUM_STOCK_SYMBOLS = len(symbols)

# Define the observation space
observation_space = spaces.Dict({
    "stacked_obs": spaces.Box(low=-1, high=1, shape=(NUM_STOCK_SYMBOLS, NUM_FEATURES, FRAMESTACK), dtype=np.float32),
    "holdings_array": spaces.Box(low=-1, high=1, shape=(NUM_STOCK_SYMBOLS,), dtype=np.float32),
    "unrealized_pl_array": spaces.Box(low=-1, high=1, shape=(NUM_STOCK_SYMBOLS,), dtype=np.float32)
})


# Creating directories for models and logs
model_type = "PPO"
timestamp = int(time.time())
models_dir = f"models/{model_type}-{timestamp}"
logdir = f"logs/{model_type}-{timestamp}"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Neural Network and TensorBoard setup
model_temp = CustomNetwork(
    observation_space=observation_space, features_dim=DENSE_LAYER_SIZE)

# Creating a mock observation that matches the expected dictionary structure
mock_observation = {
    "stacked_obs": torch.rand(BATCH_SIZE, NUM_STOCK_SYMBOLS, NUM_FEATURES, FRAMESTACK),
    "holdings_array": torch.rand(BATCH_SIZE, NUM_STOCK_SYMBOLS),
    "unrealized_pl_array": torch.rand(BATCH_SIZE, NUM_STOCK_SYMBOLS)
}

# Model information and TensorBoard logging
writer = SummaryWriter(logdir)
writer.add_graph(model_temp, mock_observation)
writer.close()

print(f"Model output shape: {model_temp(mock_observation).shape}")
print(f"Model architecture:\n{model_temp}")

# Loading data
train_data = load_data(
    ratio=0.8, split_option=SplitOption.TRAIN_SPLIT, symbols=symbols, table_name=table_name)

test_data = load_data(
    ratio=0.8, split_option=SplitOption.TEST_SPLIT, symbols=symbols, table_name=table_name)

# Environment setup
env = TradingGameEnv(data=train_data, features=features,
                     framestack=FRAMESTACK, random_reset=True)

test_env = TradingGameEnv(
    data=test_data, features=features, framestack=FRAMESTACK, random_reset=False)

check_env(env)  # Optional: Check if the environment follows the Gym interface

# PPO model setup
policy_kwargs = {
    "features_extractor_class": CustomNetwork,
    "features_extractor_kwargs": {"features_dim": DENSE_LAYER_SIZE}
}

model = PPO('MultiInputPolicy', env, verbose=1, learning_rate=LEARNING_RATE, n_steps=2048,
            batch_size=BATCH_SIZE, n_epochs=10, gae_lambda=0.95, clip_range=0.2,
            normalize_advantage=True, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            use_sde=True, sde_sample_freq=21, tensorboard_log=logdir,
            policy_kwargs=policy_kwargs, gamma=0.999)


# Callbacks for model saving and evaluation
checkpoint_callback = CheckpointCallback(
    save_freq=10000, save_path=models_dir, name_prefix=model_type)
best_models_dir = os.path.join(models_dir, "best_models")
os.makedirs(best_models_dir, exist_ok=True)

eval_log_path = os.path.join(logdir, "eval_log.txt")
eval_callback = EvalCallback(
    test_env, best_models_dir, eval_log_path, eval_freq=10000)

# Training
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[
            checkpoint_callback, eval_callback])

# Save the final model
model.save(os.path.join(models_dir, "final_trading_model"))
