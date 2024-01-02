import os
from stable_baselines3 import PPO
from environment.trading_game_env import TradingGameEnv
from util.alpaca_trade import LiveTrader
from data.data_loader import load_data, SplitOption
from dotenv import load_dotenv
# Load the environment file
load_dotenv('keys.env')

model = "PPO"
time = 1703620609
model_path = f"models/{model}-{time}"
# Change the file name if needed
final_model_path = os.path.join(model_path, "best_models/best_model_30.29")

# Get test data
test_data = load_data(ratio=0.8, split_option=SplitOption.TEST_SPLIT)

# Create the environment with test data
env = TradingGameEnv(test_data)
model = PPO.load(final_model_path, env=env)

# Run the model on the environment
obs, _ = env.reset()
done = False
# env.initialize_pygame()
env.rendering = False

api_key = os.environ.get("APCA_API_KEY_ID")
secret_key = os.environ.get("APCA_API_SECRET_KEY")
polygon_key = os.environ.get("POLYGON_API_KEY")

symbols = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON',
           'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WMT']

live_trader = LiveTrader(api_key, secret_key, polygon_key, symbols)
# Load historical data
data = load_data(ratio=0.8, split_option=SplitOption.NO_SPLIT)
env = TradingGameEnv(data_dict=data, live_trader=live_trader)
env.reset()
env.verbose = True

try:
    # Main loop
    while True:
        # Update environment with new data
        env.sync_with_live_trader(live_trader)
        new_data = live_trader.fetch_all_data(symbols)
        env.update_data(new_data)

        obs = env.next_observation()
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, _, info = env.step(action)
        # print(f"Step: {env.current_step}, Reward: {rewards}")
        # Wait for next update cycle
        break
        # Render the environment
        # env.render()

except KeyboardInterrupt:
    print("Rendering stopped by user.")

finally:
    env.close()  # Call the close method for cleanup
