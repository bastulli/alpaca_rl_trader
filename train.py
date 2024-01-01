import time
import os
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
from trading_game_env import TradingGameEnv
from data.data_loader import load_data, SplitOption
from torch.utils.tensorboard import SummaryWriter
from neural_network import CustomNetwork
from util.custom_eval_callback import EvalCallback

# Hyperparameters for example model
num_stock_symbols = 28
learning_rate = 0.0001
BATCH_SIZE = 64

# Hyperparameters for custom model
denseLayerSize = 32

# Define observation space
# Assuming these values are defined somewhere in your environment
NUM_STOCK_SYMBOLS = 28  # Replace with your actual number of stock symbols
NUM_FEATURES = 6        # Replace with your actual number of features per stock symbol
FRAMESTACK = 5          # Number of stacked frames

# Define the observation space
observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(NUM_STOCK_SYMBOLS, NUM_FEATURES, FRAMESTACK),
    dtype=np.float32
)

# Create the Neural Network
modelTemp = CustomNetwork(
    observation_space=observation_space, features_dim=denseLayerSize)

# Create a mock observation
mock_observation = torch.rand(
    BATCH_SIZE, NUM_STOCK_SYMBOLS, NUM_FEATURES, FRAMESTACK)

# Forward pass through the model with mock observation
output = modelTemp(mock_observation)

# Print shapes and the model
print("\nOutput shape:")
print(output.shape)
print("\nModel output:")
print(output)
print("\nModel architecture:")
print(modelTemp)

model_type = "PPO"
models_dir = f"models/{model_type}-{int(time.time())}"
logdir = f"logs/{model_type}-{int(time.time())}"

# Make directories
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

writer = SummaryWriter(logdir)
writer.add_graph(modelTemp, mock_observation)
writer.close()

# Get data
train_data = load_data(ratio=0.8, split_option=SplitOption.TRAIN_SPLIT)
# Get test data
test_data = load_data(ratio=0.8, split_option=SplitOption.TEST_SPLIT)

# Create the environment
env = TradingGameEnv(train_data)
# Create the test environment
test_env = TradingGameEnv(test_data)

check_env(env)  # Optional: check if the environment follows the Gym interface
print(f"env.observation_space {env.observation_space}")
print(f'env.observation_space.shape {env.observation_space.shape}')

# # Custom nn architecture
policy_kwargs = dict(
    features_extractor_class=CustomNetwork,
    features_extractor_kwargs=dict(
        features_dim=denseLayerSize)  # Adjust as needed
)

# Instantiate the agent with your custom policy
model = PPO('CnnPolicy', env, verbose=1,
            learning_rate=learning_rate, tensorboard_log=logdir, batch_size=BATCH_SIZE, policy_kwargs=policy_kwargs)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000, save_path=models_dir, name_prefix=model_type)

# Train the agent
total_timesteps = 5000000
best_models_dir = os.path.join(models_dir, "best_models")
if not os.path.exists(best_models_dir):
    os.makedirs(best_models_dir)

eval_log_path = os.path.join(logdir, "eval_log.txt")

eval_callback = EvalCallback(
    test_env, best_models_dir, eval_log_path, eval_freq=10000)

model.learn(total_timesteps=total_timesteps, callback=[
            checkpoint_callback, eval_callback])

# Save the final model
final_model_path = os.path.join(models_dir, "final_trading_model")
model.save(final_model_path)
