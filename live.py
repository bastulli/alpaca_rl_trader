import os
import time
from stable_baselines3 import PPO
from dotenv import load_dotenv
import schedule

# Custom imports
from environment.trading_game_env import TradingGameEnv
from util.alpaca_trade import LiveTrader
from data.data_loader import load_data, SplitOption
from data.historical_data import fetch_data_for_ticker

# Load environment variables from .env file
load_dotenv('keys.env')

# Configuration parameters
MODEL_TYPE = "PPO"
MODEL_TIMESTAMP = "1703620609"  # Replace with your model's timestamp
# Replace with your model's filename
BEST_MODEL_FILENAME = "best_model_30.29.zip"

# Paths setup
model_path = f"models/{MODEL_TYPE}-{MODEL_TIMESTAMP}"
final_model_path = os.path.join(
    model_path, f"best_models/{BEST_MODEL_FILENAME}")

# API keys
api_key = os.environ.get("APCA_API_KEY_ID")
secret_key = os.environ.get("APCA_API_SECRET_KEY")
polygon_key = os.environ.get("POLYGON_API_KEY")

crypto_symbols = ['BAT', 'BCH', 'BTC', 'CRV',
                  'DOT', 'ETH', 'GRT', 'LINK', 'LTC', 'MKR', 'UNI', 'XTZ']
# Using list comprehension to modify each element in the list
symbols = ['X:' + symbol + 'USD' for symbol in crypto_symbols]

table_name = 'crypto_data_hourly'

# Load and create the trading environment
test_data = load_data(ratio=0.8, split_option=SplitOption.TEST_SPLIT,
                      symbols=symbols, table_name=table_name)

env = TradingGameEnv(test_data)
model = PPO.load(final_model_path, env=env)

# Initialize LiveTrader
live_trader = LiveTrader(api_key, secret_key, polygon_key, symbols)

# Load historical data and reset environment for live trading
data = load_data(ratio=0.8, split_option=SplitOption.NO_SPLIT,
                 symbols=symbols, trim_data=True)

env = TradingGameEnv(
    data_dict=data, live_trader=live_trader, symbols=symbols)
env.reset()
env.verbose = True


def run_bot():
    """
    Main function to run the trading bot.
    """
    try:
        # Check for new data and update database file
        fetch_data_for_ticker(symbols)
        # Load new data into environment from database file
        new_data = load_data(split_option=SplitOption.NO_SPLIT,
                             symbols=symbols, trim_data=True)

        # Update environment with new live data
        env.sync_with_live_trader(live_trader)  # sync cash and positions

        # Update environment with new data
        env.update_data(new_data)

        # Get observation (stacked data features, prices and history)
        obs = env.next_observation()

        # Predict action and get new observation
        action, _states = model.predict(obs, deterministic=True)
        _, rewards, _, _, info = env.step(action)

        # Uncomment to see step and reward info
        # print(f"Step: {env.current_step}, Reward: {rewards}")

        # Uncomment to render the environment
        # env.render()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        env.close()  # Clean up the environment


# Schedule the bot to run at the beginning of every hour (modify as needed)
schedule.every().hour.do(run_bot)

# Main loop
try:
    while True:
        schedule.run_pending()
        time.sleep(1)

except KeyboardInterrupt:
    print("Scheduled trading stopped by user.")
