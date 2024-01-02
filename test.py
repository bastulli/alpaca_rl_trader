import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.trading_game_env import TradingGameEnv
from data.data_loader import load_data, SplitOption
import matplotlib.pyplot as plt

model = "PPO"
time = 1704149964
model_path = f"models/{model}-{time}"
# Change the file name if needed
final_model_path = os.path.join(
    model_path, "best_models/best_model_25_166099219117314.zip")


symbols = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON',
           'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WMT', 'DOW', 'WBA', 'NVDA']

# Get test data
test_data = load_data(
    ratio=0.8, split_option=SplitOption.TEST_SPLIT, symbols=symbols)

# Create the environment with test data
env = TradingGameEnv(test_data)
model = PPO.load(final_model_path, env=env)

# Run the model on the environment
obs, _ = env.reset()
done = False
# env.initialize_pygame()
env.rendering = False

# evaluate_policy(model, env, n_eval_episodes=5, render=True)

try:
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, _, info = env.step(action)
        # print(f"Step: {env.current_step}, Reward: {rewards}")

        # Render the environment
        # env.render()

except KeyboardInterrupt:
    print("Rendering stopped by user.")

finally:
    env.close()  # Call the close method for cleanup
    plt.ioff()  # Turn off interactive mode, if it's still on
