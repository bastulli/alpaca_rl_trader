from collections import deque
import pickle
import numpy as np
import math
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque, OrderedDict
import sqlite3

FRAMESTACK = 7
NUM_FEATURES = 6


class TradingGameEnv(gym.Env):
    def __init__(self, data_dict, initial_balance=100000.0, transaction_cost=0.0001, live_trader=None):
        super(TradingGameEnv, self).__init__()
        self.verbose = False
        self.rendering = False
        self.min_dollar_value = 100.0

        # Sort data_dict by its keys
        self.data_dict = OrderedDict(sorted(data_dict.items()))

        # Initialize
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Add live trader instance
        self.live_trader = live_trader

        # Get the first ticker key from the sorted dictionary
        first_ticker = next(iter(self.data_dict))
        self.length_of_data = len(self.data_dict[first_ticker]['price'])
        self.number_of_symbols = len(self.data_dict.keys())
        self.observations = deque(maxlen=FRAMESTACK)
        print(f"Number of symbols: {self.number_of_symbols}")
        print(f" keys {self.data_dict.keys()}")

        # Action space is a tuple of (allocation amount)
        self.action_space = spaces.Box(low=0, high=1, shape=(
            self.number_of_symbols,), dtype=np.float32)

        # Define the observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.number_of_symbols, NUM_FEATURES, FRAMESTACK),
            dtype=np.float32

        )

    def step(self, action):

        if self.live_trader is None:
            # Increment step
            self.current_step += 1

        # Take action
        self._take_action(action)

        if self.rendering:
            self.action_space_history.append(action)

        # Update the maximum portfolio value
        self.max_portfolio_value = max(
            self.max_portfolio_value, self.last_portfolio_value)

        # Calculate current drawdown
        current_drawdown = (self.max_portfolio_value -
                            self.last_portfolio_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Calculate reward
        reward = self._calculate_reward()
        self.cumulative_reward += reward

        # Get the next observation
        obs = self.next_observation(self.current_step)

        # Check if we are done
        done = self.current_step >= self.length_of_data - 2

        # If we are low on cash, we are done
        if self.last_portfolio_value < (self.initial_balance * 0.1):
            done = True

        if done:
            # Calculate total portfolio value
            total_portfolio_value = self._calculate_total_portfolio_value()

            # Calculate percentage gain/loss
            percentage_change = (
                (total_portfolio_value - self.initial_balance) / self.initial_balance) * 100

            # Step 1: Calculate cumulative log returns for each symbol
            cumulative_log_return = np.array([self.data_dict[symbol]['log_pct_change'].cumsum()
                                              for symbol in self.data_dict.keys()], dtype=np.float32)

            # Step 2: Calculate the final cumulative log returns for each symbol
            final_cumulative_log_returns = cumulative_log_return[:, -1]

            # Step 3: Compute the mean of these final values
            average_performance = final_cumulative_log_returns.mean()

            # Optionally, convert this to a percentage
            average_performance_percentage = (
                np.exp(average_performance) - 1) * 100

            # Print the performance
            print(f"Total Reward: {self.cumulative_reward:.2f}")
            print(f"Total Portfolio Value: {total_portfolio_value:.2f}")
            print(f"Percentage Gain/Loss: {percentage_change:.2f}%")
            print(f"Benchmark: {average_performance_percentage:.2f}%")
            print(
                f"Beat Benchmark: {(percentage_change - average_performance_percentage):.2f}%")
            print(f"Maximum Drawdown: {self.max_drawdown*100:.2f}%")

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.live_trader is None:
            # Reset the current step
            self.current_step = 1
        else:
            # set step to be last datapoint every time
            self.current_step = -1

        # Reset the initial balance
        init_balance = self.initial_balance
        self.balance = init_balance
        self.last_portfolio_value = init_balance
        self.max_portfolio_value = init_balance

        # Reset the order history
        self.order_history = {symbol: {
            'qty': 0, 'avg_entry_price': 0} for symbol in self.data_dict.keys()}

        # Reset the cumulative reward and max drawdown
        self.cumulative_reward = 0
        self.max_drawdown = 0
        self.last_drawdown = 0

        # Reset tracking for visualization (rendering)
        self.portfolio_value_history = []
        self.holdings_ratio_history = []
        self.action_space_history = []

        # Reset the observations deque
        self.observations = deque(maxlen=FRAMESTACK)

        # warmup up framestack
        for i in range(FRAMESTACK):
            # appends the observation to the deque
            stacked_obs = self.next_observation(i-FRAMESTACK)

        return stacked_obs, {}

    def update_data(self, new_data):
        # Update self.data_dict with new_data
        for symbol in new_data:
            if symbol in self.data_dict:
                for feature in new_data[symbol]:
                    self.data_dict[symbol][feature] = np.append(
                        self.data_dict[symbol][feature], new_data[symbol][feature])[-self.length_of_data:]

    def sync_with_live_trader(self):
        account_info = self.live_trader.get_account()
        self.initial_balance = float(account_info.cash)
        self.balance = float(account_info.cash)

        # Update order history based on live trader information
        # You might need to adjust this based on how your live trader tracks orders
        positions = self.live_trader.get_all_positions()
        for position in positions:
            symbol = position.symbol
            self.order_history[symbol]['qty'] = float(position.qty)
            self.order_history[symbol]['avg_entry_price'] = float(
                position.avg_entry_price)

    def next_observation(self, index=None):
        if index is None:
            index = self.current_step

        # Vectorized extraction of features
        f_vmar_10 = np.array([self.data_dict[symbol]['f_vmar_10'][index]
                              for symbol in self.data_dict.keys()], dtype=np.float32)

        f_sma_diff_10 = np.array([self.data_dict[symbol]['f_sma_diff_10'][index]
                                  for symbol in self.data_dict.keys()], dtype=np.float32)

        f_fractional_difference_price = np.array([self.data_dict[symbol]['f_fractional_difference_price'][index]
                                                  for symbol in self.data_dict.keys()], dtype=np.float32)

        f_volitilit_10 = np.array([self.data_dict[symbol]['f_volitility_10'][index]
                                   for symbol in self.data_dict.keys()], dtype=np.float32)

        # Perform the division with safe check
        holdings_array = self._get_holdings_ratio()

        # Ensure the result is in float32 format
        holdings_array = holdings_array.astype(np.float32)

        # Calculate unrealized P/L
        unrealized_pl_array = self._calculate_unrealized_pl()

        # Ensure the result is in float32 format
        unrealized_pl_array = unrealized_pl_array.astype(np.float32)

        # Construct the observation array for the current step
        current_observation = np.stack([
            unrealized_pl_array,
            holdings_array,
            f_vmar_10,
            f_sma_diff_10,
            f_fractional_difference_price,
            f_volitilit_10
        ], axis=0)  # Shape will be [num_features, num_symbols]

        # Transpose the observation to make it [num_symbols, num_features]
        current_observation = np.transpose(
            np.clip(current_observation, -3, 3), (1, 0))

        # Add the current observation to the deque
        self.observations.append(current_observation)

        # Ensure the deque is filled up to FRAMESTACK
        while len(self.observations) < FRAMESTACK:
            self.observations.append(np.zeros_like(current_observation))

        # Stack the observations to create the final array
        # This will stack the last FRAMESTACK observations along a new dimension
        # Shape will be [num_symbols, num_features, FRAMESTACK]
        stacked_obs = np.stack(self.observations, axis=-1)

        # Store the holdings ratio for visualization
        if self.rendering:
            self.holdings_ratio_history.append(holdings_array)

        if self.live_trader is not None:
            # Save observation to database
            self._save_observation_to_db(stacked_obs)

        return stacked_obs

    def _get_holdings_ratio(self):

        # Vectorized extraction of current prices
        current_prices = np.array([self.data_dict[symbol]['price'][self.current_step]
                                   for symbol in self.data_dict.keys()], dtype=np.float32)

        # Vectorized calculations of quantities and current investments
        quantities = np.array([self.order_history[symbol]['qty']
                               for symbol in self.data_dict.keys()], dtype=np.float32)

        current_investments = quantities * current_prices

        # Calculate total portfolio value
        total_portfolio_value = self.balance + np.sum(current_investments)

        # Calculate the investment ratio for each symbol in the portfolio
        portfolio_ratios = np.divide(current_investments, total_portfolio_value,
                                     out=np.zeros_like(
                                         current_investments, dtype=np.float32),
                                     where=(total_portfolio_value > 0))

        return portfolio_ratios

    def _normalize_action_vector(self, action_vector):
        # Find the maximum value in the action vector
        max_value = np.max(action_vector)

        # Calculate the sum of the non-zero elements
        non_zero_sum = np.sum(action_vector)

        # Avoid division by zero
        if non_zero_sum == 0:
            return action_vector

        # Normalize non-zero elements while maintaining their ratio and adjusting to the max value
        normalized_action_vector = action_vector / non_zero_sum * max_value

        return normalized_action_vector

    def _bin_action_values(self, action_vector, skip_first=False):
        # Define the bin edges
        # max is never 1 to dissallow 100% utilization of the portfolio
        bins = np.arange(0, 1.0, 0.01)

        # Check if the first element needs to be skipped
        if skip_first and len(action_vector) > 0:
            # Skip binning the first element
            first_element = action_vector[0]
            rest_of_vector = action_vector[1:]

            # Reshape the rest of the vector for broadcasting
            rest_of_vector_reshaped = rest_of_vector.reshape(-1, 1)

            # Calculate the absolute differences for the rest of the vector
            abs_diff = np.abs(rest_of_vector_reshaped - bins)

            # Find the indices of the closest bin edge for each remaining action value
            closest_bin_indices = np.argmin(abs_diff, axis=1)

            # Assign the closest bin value
            binned_rest_of_vector = bins[closest_bin_indices]

            # Combine the first element with the binned rest of the vector
            binned_action_vector = np.concatenate(
                [[first_element], binned_rest_of_vector])
        else:
            # Bin the entire vector as before
            action_vector_reshaped = action_vector.reshape(-1, 1)
            abs_diff = np.abs(action_vector_reshaped - bins)
            closest_bin_indices = np.argmin(abs_diff, axis=1)
            binned_action_vector = bins[closest_bin_indices]

        return binned_action_vector

    def _take_action(self, action):

        # Normalize and bin the action values as before
        action = self._normalize_action_vector(action)

        # Bin the action values
        action = self._bin_action_values(action)

        # get the current holdings ratio
        holdings_ratio = self._get_holdings_ratio()

        # Bin the holding values
        # we skip the binning first element because we should sell
        # if action is 0 while the current holding is not 0
        holdings_ratio = self._bin_action_values(
            holdings_ratio, skip_first=True)

        # Calculate total portfolio value (buying and selling might change the portfolio value)
        # might need to update this for each symbol after buying and selling
        total_portfolio_value = self._calculate_total_portfolio_value()

        if self.verbose:
            for i, symbol in enumerate(self.data_dict.keys()):
                print(
                    f"{symbol} - Action: {action[i]:.2f}, Holdings: {holdings_ratio[i]:.2f}, dollar value: {holdings_ratio[i] * total_portfolio_value:.2f}")
            print(f'Total portfolio value: {total_portfolio_value:.2f}')

        # First, process all sell actions
        for symbol_idx in range(self.number_of_symbols):
            symbol = list(self.data_dict.keys())[symbol_idx]
            desired_percentage = action[symbol_idx]
            current_percentage = holdings_ratio[symbol_idx]

            # Execute sell action if desired percentage less than current percentage
            if desired_percentage < current_percentage:
                self._sell(symbol, desired_percentage, total_portfolio_value)

        # Next, process all buy actions
        for symbol_idx in range(self.number_of_symbols):
            symbol = list(self.data_dict.keys())[symbol_idx]
            desired_percentage = action[symbol_idx]
            current_percentage = holdings_ratio[symbol_idx]

            # Execute buy action if current holdings less than desired percentage
            if current_percentage < desired_percentage:
                self._buy(symbol, desired_percentage, total_portfolio_value)

    def _sell(self, symbol, desired_percentage, total_portfolio_value):
        # Clip the desired percentage to make sure it's between 0 and 1
        desired_percentage = np.clip(desired_percentage, 0, 1)

        # get the current price and quantity for symbol
        current_price = self.data_dict[symbol]["price"][self.current_step]
        current_quantity = self.order_history[symbol]['qty']
        # Current dollar amount invested in this symbol
        current_dollar_amount = current_quantity * current_price

        # Desired dollar amount allocation for this symbol
        desired_dollar_amount = total_portfolio_value * desired_percentage

        # Calculate the quantity to sell based on the dollar amount difference
        sell_dollar_amount = current_dollar_amount - desired_dollar_amount

        # Calculate the quantity to sell based on the dollar amount
        sell_quantity = sell_dollar_amount / current_price

        # Ensure we don't sell more than we have
        sell_quantity = min(sell_quantity, current_quantity)

        # Check if we are selling less than the minimum dollar value and it's not the entire holding
        if sell_dollar_amount < self.min_dollar_value and sell_quantity != current_quantity:
            if self.verbose:
                print(
                    f'Skipping sell for {symbol} because order ${sell_dollar_amount:.2f} is less than min ${self.min_dollar_value:.2f}')
            return  # skip if we are selling less than min dollar value

        # Use LiveTrader to execute the sell
        if self.live_trader and sell_quantity > 0:
            self.live_trader.sell(symbol, sell_quantity)

        # Update the quantity in the dictionary
        self.order_history[symbol]['qty'] -= sell_quantity

        # Update the balance, considering the transaction cost
        self.balance += sell_dollar_amount - \
            (sell_dollar_amount * self.transaction_cost)

        # Calculate realized profit or loss
        average_entry_price = self.order_history[symbol]['avg_entry_price']

        # Update the average entry price if all shares are sold
        if self.order_history[symbol]['qty'] <= 0:
            self.order_history[symbol]['avg_entry_price'] = 0

        realized_profit = (current_price - average_entry_price) * \
            sell_quantity - (sell_dollar_amount * self.transaction_cost)

        if self.verbose:
            print(f"Sold {sell_quantity} shares of {symbol} at ${current_price:.2f} for a gain/loss of {realized_profit:.2f}, dollar amount: ${sell_dollar_amount:.2f}")

    def _buy(self, symbol, desired_percentage, total_portfolio_value):
        # Clip the desired percentage to make sure it's between 0 and 1
        desired_percentage = np.clip(desired_percentage, 0, 1)

        # Current price for this symbol and current quantity
        current_price = self.data_dict[symbol]["price"][self.current_step]
        current_quantity = self.order_history[symbol]['qty']

        # Current dollar amount invested in this symbol
        current_dollar_amount = current_quantity * current_price

        # Desired dollar amount allocation for this symbol
        desired_dollar_amount = total_portfolio_value * desired_percentage

        # Calculate how much more needs to be bought to reach the desired allocation
        additional_dollar_amount_needed = desired_dollar_amount - current_dollar_amount

        # Adjust for transaction cost
        fee = additional_dollar_amount_needed * self.transaction_cost
        total_cost = additional_dollar_amount_needed + fee
        purchased_quantity = additional_dollar_amount_needed / current_price

        if additional_dollar_amount_needed < self.min_dollar_value:
            if self.verbose:
                print(
                    f'Skipping buy for {symbol} because order amount ${additional_dollar_amount_needed:.2f} is less than min ${self.min_dollar_value:.2f}')
            return  # skip if we are selling less than min dollar value

        if total_cost <= self.balance and purchased_quantity > 0:

            # Use LiveTrader to execute the buy
            if self.live_trader:
                self.live_trader.buy(symbol, purchased_quantity)

            self.balance -= total_cost

            # Update the average entry price and quantity directly in the dictionary
            total_quantity = current_quantity + purchased_quantity
            self.order_history[symbol]['avg_entry_price'] = (
                (self.order_history[symbol]['avg_entry_price'] * current_quantity) + additional_dollar_amount_needed) / total_quantity

            self.order_history[symbol]['qty'] = total_quantity

            if self.verbose:
                print(
                    f"Bought {purchased_quantity} shares of {symbol} at ${current_price:.2f} for a cost of ${total_cost:.2f}")

    def _calculate_unrealized_pl(self):
        # Extract quantities, current prices, and average entry prices for all symbols
        quantities = np.array([self.order_history[symbol]['qty']
                               for symbol in self.data_dict.keys()])

        current_prices = np.array(
            [self.data_dict[symbol]['price'][self.current_step] for symbol in self.data_dict.keys()])

        average_entry_prices = np.array(
            [self.order_history[symbol]['avg_entry_price'] for symbol in self.data_dict.keys()])

        # Calculate market values and total P/L
        market_values = quantities * current_prices
        total_pl = market_values - (quantities * average_entry_prices)

        # Safe division with np.divide, handling division by zero
        denominator = quantities * average_entry_prices
        unrealized_pl = np.divide(total_pl, denominator, out=np.zeros_like(
            denominator, dtype=np.float64), where=denominator != 0)  # * 100

        return unrealized_pl

    def _calculate_total_portfolio_value(self):
        # Extract quantities and current prices for all symbols
        quantities = np.array([self.order_history[symbol]['qty']
                              for symbol in self.data_dict.keys()])

        current_prices = np.array(
            [self.data_dict[symbol]['price'][self.current_step] for symbol in self.data_dict.keys()])

        # Calculate total portfolio value
        portfolio_value = self.balance + np.sum(quantities * current_prices)

        # return dollar value
        return portfolio_value

    def _calculate_reward(self):
        current_portfolio_value = self._calculate_total_portfolio_value()
        if self.last_portfolio_value > 0:
            # Calculate the logarithmic return
            log_return = math.log(
                current_portfolio_value / self.last_portfolio_value)

            if math.isnan(log_return) or math.isinf(log_return):
                log_return = 0

            reward = log_return
        else:
            reward = 0

        self.last_portfolio_value = current_portfolio_value

        if self.rendering:
            self.portfolio_value_history.append(current_portfolio_value)

        return reward * 100

    def initialize_pygame(self):
        self.window_size = 800
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def _draw_action_space(self):
        bar_chart_width = self.window_size
        bar_chart_height = self.window_size // 8  # 1/8th of the height
        bar_chart_x = 0
        bar_chart_y = self.window_size - (bar_chart_height * 2)  # 7th eighth

        # Background for the bar chart area (optional, for better visibility)
        pygame.draw.rect(self.screen, (220, 220, 220), (bar_chart_x,
                                                        bar_chart_y, bar_chart_width, bar_chart_height))

        # Number of symbols and width for each bar
        num_symbols = len(self.action_space_history[-1])
        bar_width = bar_chart_width / num_symbols

        for i, ratio in enumerate(self.action_space_history[-1]):
            # Calculate the height of the bar based on the holdings ratio
            bar_height = ratio * bar_chart_height

            # Calculate the position of the bar
            x = bar_chart_x + i * bar_width
            y = bar_chart_height - bar_height + bar_chart_y  # Adjust y-position

            # Draw the bar
            pygame.draw.rect(self.screen, (0, 128, 0),
                             (x, y, bar_width, bar_height))  # Green bars

            # # Optional: Draw labels for each bar
            # font = pygame.font.SysFont(None, 24)
            # label = font.render(f'Symbol {i+1}', True, (0, 0, 0))
            # # Adjust label position as needed
            # self.screen.blit(label, (x + 5, bar_chart_height - 20))

    def _draw_holdings_ratio_chart(self):
        # Define the size and position of the bar chart area
        bar_chart_width = self.window_size  # Span the entire width of the window
        bar_chart_height = self.window_size // 8  # Maintain reduced height
        bar_chart_x = 0  # Start from the left edge of the window
        bar_chart_y = self.window_size - bar_chart_height  # Position at the bottom half

        # Background for the bar chart area (optional, for better visibility)
        pygame.draw.rect(self.screen, (220, 220, 220), (bar_chart_x,
                                                        bar_chart_y, bar_chart_width, bar_chart_height))

        # Number of symbols and width for each bar
        num_symbols = len(self.holdings_ratio_history[-1])
        bar_width = bar_chart_width / num_symbols

        for i, ratio in enumerate(self.holdings_ratio_history[-1]):
            # Calculate the height of the bar based on the holdings ratio
            bar_height = ratio * bar_chart_height

            # Calculate the position of the bar
            x = bar_chart_x + i * bar_width
            y = bar_chart_height - bar_height + bar_chart_y  # Adjust y-position

            # Draw the bar
            pygame.draw.rect(self.screen, (0, 128, 0),
                             (x, y, bar_width, bar_height))  # Green bars

            # # Optional: Draw labels for each bar
            # font = pygame.font.SysFont(None, 24)
            # label = font.render(f'Symbol {i+1}', True, (0, 0, 0))
            # # Adjust label position as needed
            # self.screen.blit(label, (x + 5, bar_chart_height - 20))

    def _draw_portfolio_value_chart(self):
        # Assuming self.portfolio_value_history is a list of portfolio values
        max_value = max(self.portfolio_value_history)
        min_value = min(self.portfolio_value_history)
        normalized_values = [(value - min_value) / (max_value - min_value)
                             for value in self.portfolio_value_history]

        # Adjust the chart area to occupy the top 6/8th of the window
        chart_height = self.window_size * (6 / 8)

        for i in range(len(normalized_values) - 1):
            start_pos = (i * (self.window_size / len(self.portfolio_value_history)),
                         chart_height * (1 - normalized_values[i]))
            end_pos = ((i + 1) * (self.window_size / len(self.portfolio_value_history)),
                       chart_height * (1 - normalized_values[i + 1]))
            pygame.draw.line(self.screen, (0, 0, 255),
                             start_pos, end_pos, 2)

    def render(self, done=False, mode='human'):
        if mode == 'human':
            self.screen.fill((255, 255, 255))  # Clear screen (fill with white)

            self._draw_action_space()
            self._draw_portfolio_value_chart()
            self._draw_holdings_ratio_chart()

            pygame.display.flip()  # Update the full display Surface to the screen
            self.clock.tick(60)  # Limit frames per second

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            if done:
                pygame.quit()

        else:
            raise NotImplementedError

    def close(self):
        pygame.quit()
