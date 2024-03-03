import os
import time
from stable_baselines3 import PPO
from dotenv import load_dotenv
import schedule

# Custom imports
from environment.trading_game_env import TradingGameEnv
from util.alpaca_trade import LiveTrader
from data.data_loader import load_data, SplitOption
from data.historical_data import TimeFrequency, fetch_all_data

# Load environment variables from .env file
load_dotenv('keys.env')

# Configuration parameters
FRAMESTACK = 5  # Number of stacked frames (history)
MODEL_TYPE = "PPO"
MODEL_TIMESTAMP = "1705435660"  # Replace with the specific model timestamp
# Replace with the specific best model number
BEST_MODEL_NUMBER = "788_01"

# Paths setup
models_dir = f"models/{MODEL_TYPE}-{MODEL_TIMESTAMP}"
best_model_path = os.path.join(
    models_dir, f"best_models/best_model_{BEST_MODEL_NUMBER}.zip")

# Features and symbols setup
features = ['f_idvwpm', 'f_dvwpm', 'f_return', 'f_price', 'f_dollar_volume']

crypto_symbols = ['BAT', 'BCH', 'BTC', 'CRV',
                  'DOT', 'ETH', 'GRT', 'LINK', 'LTC', 'MKR', 'UNI', 'XTZ']
# API keys
api_key = os.environ.get("APCA_API_KEY_ID")
secret_key = os.environ.get("APCA_API_SECRET_KEY")
polygon_key = os.environ.get("POLYGON_API_KEY")

# Using list comprehension to modify each element in the list
symbols = ['X:' + symbol + 'USD' for symbol in crypto_symbols]

table_name = 'crypto_data_hourly'

# Load and create the trading environment
data = load_data(
    ratio=0.8, split_option=SplitOption.LIVE_MODE, symbols=symbols, table_name=table_name)

# Initialize LiveTrader
live_trader = LiveTrader(api_key, secret_key, polygon_key)

# Create the environment with test data
env = TradingGameEnv(data=data, features=features, live_trader=live_trader,
                     framestack=FRAMESTACK, rendering=True)
# Load the model
model = PPO.load(best_model_path, env=env)


env.reset()
env.verbose = True


def run_bot():
    """
    Main function to run the trading bot.
    """
    try:
        print('getting new data...')
        # Check for new data and update database file
        fetch_all_data(symbols, days=1, table_name=table_name,
                       frequency=TimeFrequency.HOUR)

        print('loading new data...')
        # Load new data into environment from database file
        new_data = load_data(split_option=SplitOption.LIVE_MODE,
                             symbols=symbols, trim_data=True, table_name=table_name)

        print('syncing with live trader...')
        # Update environment with new live data
        env.sync_with_live_trader()  # sync cash and positions

        print('updating data...')
        # Update environment with new data
        env.update_data(new_data)

        print('getting observation...')
        # Get observation (stacked data features, prices and history)
        obs = env.next_observation()

        print('predicting action...')
        # Predict action and get new observation
        action, _states = model.predict(obs, deterministic=True)
        _, rewards, _, _, info = env.step(action)

        # Uncomment to see step and reward info
        print(f"Step: {env.current_step}, Reward: {rewards}")

        # Uncomment to render the environment
        env.render()

    except Exception as e:
        print(f"An error occurred: {e}")

    # finally:
        # env.close()  # Clean up the environment


# Schedule the bot to run at the beginning of every hour (modify as needed)
schedule.every().hour.at("00:05").do(run_bot)  # "00:05"

# Run the bot once immediately at startup
run_bot()

# Main loop
try:
    while True:
        schedule.run_pending()
        env.render()
        # time.sleep(1)

except KeyboardInterrupt:
    print("Scheduled trading stopped by user.")
