import os
import numpy as np
from stable_baselines3 import PPO
from environment.trading_game_env import TradingGameEnv
from data.data_loader import load_data, SplitOption
import matplotlib.pyplot as plt

# Configuration parameters
FRAMESTACK = 5  # Number of stacked frames (history)
MODEL_TYPE = "PPO"
MODEL_TIMESTAMP = "1704751113"  # Replace with the specific model timestamp
# Replace with the specific best model number
BEST_MODEL_NUMBER = "79_5"

# Paths setup
models_dir = f"models/{MODEL_TYPE}-{MODEL_TIMESTAMP}"
best_model_path = os.path.join(
    models_dir, f"PPO_4240000_steps.zip")

# Features and symbols setup
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

# Load test data
test_data = load_data(
    ratio=0.8, split_option=SplitOption.TEST_SPLIT, symbols=symbols, table_name=table_name)

# Create the environment with test data
env = TradingGameEnv(data=test_data, features=features,
                     framestack=FRAMESTACK, rendering=True)

# Load the model
model = PPO.load(best_model_path, env=env)

# Testing the model
obs, _ = env.reset(random_reset=False)
done = False

try:
    while 1:
        if not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, _, info = env.step(action)
            # Uncomment to see step and reward info
            # print(f"Step: {env.current_step}, Reward: {rewards}")

        # Uncomment to render the environment
        env.render()

except KeyboardInterrupt:
    print("Rendering stopped by user.")

finally:
    env.close()  # Clean up the environment
    plt.ioff()  # Turn off interactive mode
