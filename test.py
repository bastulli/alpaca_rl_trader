import os
from stable_baselines3 import PPO
from environment.trading_game_env import TradingGameEnv
from data.data_loader import load_data, SplitOption
import matplotlib.pyplot as plt

# Configuration parameters
FRAMESTACK = 7
MODEL_TYPE = "PPO"
MODEL_TIMESTAMP = "1704225259"  # Replace with the specific model timestamp
# Replace with the specific best model number
BEST_MODEL_NUMBER = "27_930107027757913"

# Paths setup
models_dir = f"models/{MODEL_TYPE}-{MODEL_TIMESTAMP}"
best_model_path = os.path.join(
    models_dir, f"best_models/best_model_{BEST_MODEL_NUMBER}.zip")

# Features and symbols setup
features = ['f_vmar_10', 'f_volitility_10', 'f_sma_diff_10',
            'f_percentage_change_zscore', 'f_fractional_difference_price']
symbols = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON',
           'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WMT', 'DOW', 'WBA', 'NVDA']

# Load test data
test_data = load_data(
    ratio=0.8, split_option=SplitOption.TEST_SPLIT, symbols=symbols)

# Create the environment with test data
env = TradingGameEnv(data=test_data, features=features, framestack=FRAMESTACK)

# Load the model
model = PPO.load(best_model_path, env=env)

# Testing the model
env.rendering = False
obs, _ = env.reset()
done = False

try:
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, _, info = env.step(action)
        # Uncomment to see step and reward info
        # print(f"Step: {env.current_step}, Reward: {rewards}")

        # Uncomment to render the environment
        # env.render()

except KeyboardInterrupt:
    print("Rendering stopped by user.")

finally:
    env.close()  # Clean up the environment
    plt.ioff()  # Turn off interactive mode
