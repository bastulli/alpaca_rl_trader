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
DENSE_LAYER_SIZE = 32
LEARNING_RATE = 0.0001
TOTAL_TIMESTEPS = 5000000

features = ['f_percentage_change_zscore', 'f_dollar_volume_zscore', 'f_fractional_difference_price', 'f_vmar',
            'f_cumulative_return', 'f_log_pct_change', 'f_rsi', 'f_bbands', 'f_macd', 'f_obv', 'f_historical_volatility']

# symbols = ['AAPL',  # Apple
#            'MSFT',  # Microsoft
#            'INTC',  # Intel
#            'NVDA',  # Nvidia
#            'JPM',   # JPMorgan Chase
#            'GS',    # Goldman Sachs
#            'BAC',   # Bank of America
#            'KO',    # Coca-Cola
#            'PEP',   # PepsiCo
#            'PG',    # Procter & Gamble
#            'WMT',   # Walmart
#            'PFE',   # Pfizer
#            'JNJ',   # Johnson & Johnson
#            'MRK',   # Merck
#            'XOM',   # ExxonMobil
#            'CVX',   # Chevron
#            'BP',    # BP
#            'AMZN',  # Amazon
#            'BABA',  # Alibaba
#            'T',     # AT&T
#            'VZ',    # Verizon
#            'DUK',   # Duke Energy
#            'SO',    # Southern Company
#            'TSLA',  # Tesla
#            'GM',    # General Motors
#            'F',     # Ford
#            'BA']    # Boeing

crypto_symbols = ['BAT', 'BCH', 'BTC', 'CRV',
                  'DOT', 'ETH', 'GRT', 'LINK', 'LTC', 'MKR', 'UNI', 'XTZ']

# Using list comprehension to modify each element in the list
symbols = ['X:' + symbol + 'USD' for symbol in crypto_symbols]

table_name = 'crypto_data_hourly'

# Add unrealized_pl_array, holdings_array, max_drawdown_array
NUM_FEATURES = len(features) + 3
NUM_STOCK_SYMBOLS = len(symbols)

# Define the observation space
observation_space = spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(NUM_STOCK_SYMBOLS, NUM_FEATURES, FRAMESTACK),
    dtype=np.float32
)

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
mock_observation = torch.rand(
    BATCH_SIZE, NUM_STOCK_SYMBOLS, NUM_FEATURES, FRAMESTACK)

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
env = TradingGameEnv(data=train_data, features=features, framestack=FRAMESTACK)
test_env = TradingGameEnv(
    data=test_data, features=features, framestack=FRAMESTACK)
test_env.reset(random_reset=False)
check_env(env)  # Optional: Check if the environment follows the Gym interface

# PPO model setup
policy_kwargs = {
    "features_extractor_class": CustomNetwork,
    "features_extractor_kwargs": {"features_dim": DENSE_LAYER_SIZE}
}

model = PPO('CnnPolicy', env, verbose=1, learning_rate=LEARNING_RATE, n_steps=2048,
            tensorboard_log=logdir, batch_size=BATCH_SIZE, policy_kwargs=policy_kwargs)

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
