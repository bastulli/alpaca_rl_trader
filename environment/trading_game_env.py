from collections import deque
import numpy as np
import math
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque, OrderedDict
import scipy
from scipy.stats import norm


class TradingGameEnv(gym.Env):
    def __init__(self, data, initial_balance=100000.0, transaction_cost=0.0025, framestack=10, features=[], live_trader=None, verbose=False, rendering=False, random_reset=False):
        super(TradingGameEnv, self).__init__()
        self.verbose = verbose
        self.rendering = rendering
        self.frame_stack = framestack
        self.features = features
        self.random_reset = random_reset
        # Add unrealized_pl_array, holdings_array, actions
        self.num_features = len(features) + 3
        self.min_dollar_value = 100.0

        # Sort data_dict by its keys
        self.time_series_data = OrderedDict(sorted(data.items()))
        self.symbol_names = list(self.time_series_data.keys())

        # Initialize
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Add live trader instance
        self.live_trader = live_trader
        self.init_sync = True

        # Get the first ticker key from the sorted dictionary
        first_ticker = next(iter(self.time_series_data))
        self.length_of_data = len(self.time_series_data[first_ticker]['close'])
        self.number_of_symbols = len(self.time_series_data.keys())
        self.observations = deque(maxlen=self.frame_stack)

        print(f"Number of symbols: {self.number_of_symbols}")
        print(f" keys {self.time_series_data.keys()}")

        # Action space is a tuple of (allocation amount)
        self.action_space = spaces.Box(low=-1, high=1, shape=(
            self.number_of_symbols,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "stacked_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.number_of_symbols, self.num_features, self.frame_stack), dtype=np.float32),
        })

        if self.rendering:
            self.initialize_pygame()

    def step(self, action):

        if self.live_trader is None:
            # Increment step
            self.current_step += 1

        # Take action
        self._take_action(action)

        # Calculate total portfolio value
        current_portfolio_value = self._calculate_total_portfolio_value()

        # Calculate logarithmic return of the strategy
        strategy_log_return = self._calculate_log_return(
            current_portfolio_value)

        # Compute average performance of an evenly split portfolio
        benchmark_log_return = self._calculate_benchmark_log_return()

        self.strategy_returns_list.append(strategy_log_return)
        self.benchmark_returns_list.append(benchmark_log_return)

        # Calculate current drawdown
        drawdown = self._calculate_drawdown(current_portfolio_value)

        # Update max drawdown
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.drawdown_with_decay += drawdown
        self.drawdown_with_decay *= 0.5

        # Update running totals and history for strategy and benchmark
        self._update_running_totals(strategy_log_return, benchmark_log_return)

        # Calculate reward
        reward = self._calculate_reward()
        # reward = strategy_log_return * 100

        self.cumulative_reward += reward

        # make sure its after reward calculation
        self.last_portfolio_value = current_portfolio_value

        self._update_rendering_history(current_portfolio_value)

        # Update history for rendering if enabled
        if self.rendering:
            self.action_space_history.append(action)
            self.strategy_returns.append(self.running_strategy)
            self.benchmark_returns.append(self.running_benchmark)

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
            cumulative_log_return = np.array([self.time_series_data[symbol]['log_pct_change'].cumsum()
                                              for symbol in self.time_series_data.keys()], dtype=np.float32)

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
            print(f"Strategy: {percentage_change:.2f}%")
            print(f"Benchmark: {average_performance_percentage:.2f}%")
            print(
                f"Beat Benchmark: {(percentage_change - average_performance_percentage):.2f}%")
            print(f"Maximum Drawdown: {self.max_drawdown*100:.2f}%")

        info = {'portfolio': current_portfolio_value,
                'step': self.current_step,
                'strategy_return': self.running_strategy,
                'benchmark_return': self.running_benchmark,
                'reward': reward,
                'cumulative_reward': self.cumulative_reward,
                'max_drawdown': self.max_drawdown,
                'action': action,
                'holdings_ratio': np.array([self.order_history[symbol]['holdings_ratio'] for symbol in self.time_series_data.keys()]).flatten(),
                'unrealized_pl': np.array([self.order_history[symbol]['unrealized_pl'] for symbol in self.time_series_data.keys()]).flatten(),
                'obs': obs,
                'date': self.time_series_data[self.symbol_names[-1]]['date'][self.current_step],
                'symbols': self.symbol_names,
                'trades': self.trades,
                'order_history': self.order_history,
                }

        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.live_trader is None:
            # Reset the current step
            if self.random_reset:
                self.current_step = np.random.randint(
                    self.frame_stack, self.length_of_data - 2)
                print(f"Random reset to step {self.current_step}")
            else:
                self.current_step = self.frame_stack
        else:
            # set step to be last datapoint every time
            self.current_step = -1

        # Reset the initial balance
        init_balance = self.initial_balance
        self.balance = init_balance
        self.last_portfolio_value = init_balance
        self.max_portfolio_value = init_balance

        # Reset the order history
        self.order_history = {
            symbol: {
                'qty': 0,
                'avg_entry_price': 0,
                'unrealized_pl': 0,
                'holdings_ratio': 0,
                'prev_unrealized_pl': 0,
                'percentage_change': 0,
                'max_drawdown': 0,
                'max_unrealized': 0,
                'total_invested': 0,

            } for symbol in self.time_series_data.keys()
        }
        self.trades = []

        # Reset the cumulative reward and max drawdown
        self.cumulative_reward = 0
        self.max_drawdown = 0
        self.highest_portfolio_value = 0
        self.running_benchmark = 0
        self.running_strategy = 0
        self.last_action = -1 * np.ones(self.number_of_symbols)
        self.symbols_log_return = np.zeros(self.number_of_symbols)

        # Reset tracking for visualization (rendering)
        self.portfolio_value_history = []  # deque(maxlen=250)
        self.benchmark_history = []  # deque(maxlen=250)
        self.strategy_history = []  # deque(maxlen=250)
        self.holdings_ratio_history = []  # deque(maxlen=250)
        self.action_space_history = []  # deque(maxlen=250)
        self.benchmark_returns = []  # deque(maxlen=250)
        self.strategy_returns = []  # deque(maxlen=250)
        # Reset the observations deque
        self.observations = deque(maxlen=self.frame_stack)

        # Reset reward mean
        self.sparse_reward = 0
        self.std_dev_strategy_returns = 0
        self.risk_free_rate = 0
        self.strategy_returns_list = []
        self.benchmark_returns_list = []
        self.drawdown_with_decay = 0
        self.reward_mean_list = deque(maxlen=24)

        # warmup up self.frame_stack
        for i in range(self.frame_stack):
            # appends the observation to the deque
            stacked_obs = self.next_observation(i-self.frame_stack)

        return stacked_obs, {}

    def update_data(self, new_data):
        # Sort data_dict by its keys
        # replace self.time_series_data with new_data
        self.time_series_data = OrderedDict(sorted(new_data.items()))

    def convert_cyrpto_ticker_symbol(self, symbol):
        if 'USD' in symbol and not 'X:' in symbol:
            return 'X:' + symbol
        else:
            raise ValueError("Invalid symbol format - 'USD' not found")

    def sync_with_live_trader(self):
        account_info = self.live_trader.get_account()

        self.balance = float(account_info.cash)
        if self.init_sync:  # only update initial balance on first sync
            self.initial_balance = float(account_info.cash)
            self.last_portfolio_value = float(account_info.portfolio_value)
            self.max_portfolio_value = self.last_portfolio_value
            self.init_sync = False

        # Update order history based on live trader information
        # You might need to adjust this based on how your live trader tracks orders
        positions = self.live_trader.get_all_positions()
        for position in positions:
            symbol = self.convert_cyrpto_ticker_symbol(position.symbol)
            print(f"Syncing {symbol} with live trader...")
            self.order_history[symbol]['qty'] = float(position.qty)
            self.order_history[symbol]['avg_entry_price'] = float(
                position.avg_entry_price)

    def next_observation(self, index=None):

        if index is None:
            index = self.current_step
        feature_arrays = []
        # Dynamic extraction of features
        for feature_name in self.features:
            feature_array = np.array([self.time_series_data[symbol][feature_name][index]
                                      for symbol in self.time_series_data.keys()], dtype=np.float32)
            feature_arrays.append(feature_array)

        # get holdings ratio
        holdings_array = np.array([self.order_history[symbol]['holdings_ratio']
                                   for symbol in self.time_series_data.keys()])

        holdings_array = self._bin_action_values(
            holdings_array).astype(np.float32)

        # get unrealized P/L
        unrealized_pl_array = np.array([self.order_history[symbol]['unrealized_pl']
                                        for symbol in self.time_series_data.keys()]).astype(np.float32)

        # Construct the observation array for the current step
        # Shape will be [num_features, num_symbols]
        current_observation = np.clip(np.stack(
            feature_arrays + [unrealized_pl_array, holdings_array, self.last_action], axis=0).astype(np.float32), -1, 1)

        # Transpose the observation to make it [num_symbols, num_features]
        current_observation = np.transpose(
            current_observation, (1, 0))

        # Add the current observation to the deque
        self.observations.append(current_observation)

        # Ensure the deque is filled up to FRAMESTACK
        while len(self.observations) < self.frame_stack:
            self.observations.append(np.zeros_like(current_observation))

        # Stack the observations to create the final array
        # This will stack the last FRAMESTACK observations along a new dimension
        # Shape will be [num_symbols, num_features, FRAMESTACK]
        stacked_obs = np.stack(self.observations, axis=-1)

        # Store the holdings ratio for visualization
        if self.rendering:
            self.holdings_ratio_history.append(holdings_array)

        return {
            "stacked_obs": stacked_obs,
        }

    def _get_holdings_ratio_for_symbol(self, symbol):
        # Calculate holdings ratio for a specific symbol
        total_portfolio_value = self._calculate_total_portfolio_value()
        current_price = self.time_series_data[symbol]['close'][self.current_step]
        current_investment = self.order_history[symbol]['qty'] * current_price

        holdings_ratio = current_investment / \
            total_portfolio_value if total_portfolio_value > 0 else 0
        return holdings_ratio

    def _calculate_unrealized_pl_for_symbol(self, symbol):
        # Calculate unrealized P/L for a specific symbol
        qty = self.order_history[symbol]['qty']
        current_price = self.time_series_data[symbol]['close'][self.current_step]
        avg_entry_price = self.order_history[symbol]['avg_entry_price']

        # Calculate market values and total P/L
        market_value = qty * current_price
        total_pl = market_value - (qty * avg_entry_price)

        # Safe division with np.divide, handling division by zero
        denominator = qty * avg_entry_price
        unrealized_pl = np.divide(total_pl, denominator, out=np.zeros_like(
            denominator, dtype=np.float64), where=denominator != 0)  # * 100

        return unrealized_pl

    def _calculate_unrealized_pl_percentage_change(self, symbol):
        # Previous unrealized P/L
        prev_unrealized_pl = self.order_history[symbol]['prev_unrealized_pl']

        # Current unrealized P/L (from your existing method)
        current_unrealized_pl = self._calculate_unrealized_pl_for_symbol(
            symbol)

        # Calculate percentage change
        if prev_unrealized_pl != 0:
            percentage_change = current_unrealized_pl - prev_unrealized_pl
        else:
            percentage_change = 0

        # Update the order history with the current unrealized P/L
        self.order_history[symbol]['prev_unrealized_pl'] = current_unrealized_pl

        # track order history max drawdown
        if percentage_change < 0:
            self.order_history[symbol]['max_drawdown'] += percentage_change

        # decay max drawdown
        self.order_history[symbol]['max_drawdown'] = self.order_history[symbol]['max_drawdown'] * 0.95

        return percentage_change

    def _scale_action_vector(self, action_vector):
        # output is tanh, we want deadzone safespace between -0.5 and -1
        # Scale values from [-0.5, 1] to [0, 1]
        min_value = -0.75
        max_value = 1

        return np.clip((action_vector - min_value) / (max_value - min_value), 0, 1)

    def _normalize_action_vector(self, action_vector):
        # scale sum of action vector to 1
        action_vector_sum = np.sum(action_vector)
        action_vector = action_vector / action_vector_sum

        return action_vector

    def _bin_action_values(self, action_vector):
        # Define the bin edges
        bins = np.arange(0, 1.0, 0.025)
        action_vector_reshaped = action_vector.reshape(-1, 1)
        abs_diff = np.abs(action_vector_reshaped - bins)
        closest_bin_indices = np.argmin(abs_diff, axis=1)
        binned_action_vector = bins[closest_bin_indices]

        return binned_action_vector

    def _bin_single_values(self, value):
        # Define the bin edges
        bins = np.arange(0, 1.0, 0.025)
        # Calculate the absolute difference between the value and each bin
        abs_diff = np.abs(value - bins)
        # Find the index of the closest bin
        closest_bin_index = np.argmin(abs_diff)
        # Get the value of the closest bin
        binned_value = bins[closest_bin_index]
        return binned_value

    def _take_action(self, action):

        self.trades = []

        action = self._scale_action_vector(action)
        action = self._normalize_action_vector(action)
        action = self._bin_action_values(action)

        # Calculate total portfolio value (buying and selling might change the portfolio value)
        # might need to update this for each symbol after buying and selling
        total_portfolio_value = self._calculate_total_portfolio_value()

        # First, process all sell actions
        for symbol_idx in range(self.number_of_symbols):
            symbol = list(self.time_series_data.keys())[symbol_idx]
            desired_percentage = action[symbol_idx]
            current_percentage = self.order_history[symbol]['holdings_ratio']

            current_percentage = self._bin_single_values(current_percentage)

            # Execute sell action if desired percentage less than current percentage
            if desired_percentage < current_percentage:

                self._sell(symbol, desired_percentage, total_portfolio_value)

        # Next, process all buy actions
        for symbol_idx in range(self.number_of_symbols):
            symbol = list(self.time_series_data.keys())[symbol_idx]
            desired_percentage = action[symbol_idx]
            current_percentage = self.order_history[symbol]['holdings_ratio']
            current_percentage = self._bin_single_values(current_percentage)

            # Execute buy action if current holdings less than desired percentage
            if current_percentage < desired_percentage:
                self._buy(symbol, desired_percentage, total_portfolio_value)

        # After processing the actions, update the unrealized PL and holdings ratio
        for symbol in self.time_series_data.keys():
            self.order_history[symbol]['unrealized_pl'] = self._calculate_unrealized_pl_for_symbol(
                symbol)

            self.order_history[symbol]['percentage_change'] = self._calculate_unrealized_pl_percentage_change(
                symbol)

            self.order_history[symbol]['holdings_ratio'] = self._get_holdings_ratio_for_symbol(
                symbol)

    def _sell(self, symbol, desired_percentage, total_portfolio_value):
        # Clip the desired percentage to make sure it's between 0 and 1
        desired_percentage = np.clip(desired_percentage, 0, 1)

        # get the current price and quantity for symbol
        current_price = self.time_series_data[symbol]['close'][self.current_step]
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

        liquidate = False
        # write logic to check if sell quantity is everything or within threshold, if so, liquidate
        if sell_quantity == current_quantity or sell_quantity / current_quantity > 0.98:
            liquidate = True

        # Check if we are selling less than the minimum dollar value and it's not the entire holding
        if sell_dollar_amount < self.min_dollar_value and sell_quantity != current_quantity:
            if self.verbose:
                print(
                    f'Skipping sell for {symbol} because order ${sell_dollar_amount:.2f} is less than min ${self.min_dollar_value:.2f}')
            return  # skip if we are selling less than min dollar value

        # Use LiveTrader to execute the sell
        if self.live_trader and sell_quantity > 0:
            try:
                order = self.live_trader.sell(
                    symbol, sell_quantity, current_price, liquidate=liquidate)

                # Wait for order to be filled
                if not self.live_trader.wait_for_order_fill(order.id):
                    # If order is not filled within the MAX_RETRIES, cancel the order
                    self.live_trader.cancel_order(order.id)
            except Exception as e:
                # Handle exceptions, such as connectivity issues or API errors
                print(f"An error occurred when trying to sell: {e}")

        # Update the quantity in the dictionary
        self.order_history[symbol]['qty'] -= sell_quantity

        # Update the balance, considering the transaction cost
        self.balance += sell_dollar_amount - \
            (sell_dollar_amount * self.transaction_cost)

        # Calculate the proportion of the position being sold
        sell_proportion = sell_quantity / current_quantity if current_quantity > 0 else 0

        # Update the total invested amount based on the proportion being sold
        self.order_history[symbol]['total_invested'] *= (1 - sell_proportion)

        # Calculate realized profit or loss
        average_entry_price = self.order_history[symbol]['avg_entry_price']

        current_date = self.time_series_data[symbol]['date'][self.current_step]

        realized_profit = (current_price - average_entry_price) * \
            sell_quantity - (sell_dollar_amount * self.transaction_cost)

        # Append trade details including holding period
        self.trades.append({
            'symbol': symbol,
            'qty': sell_quantity,
            'close': current_price,
            'cost': sell_dollar_amount,
            'profit': realized_profit,
            'date': current_date,
            'side': 'sell'
        })

        # Update the average entry price if all shares are sold
        if self.order_history[symbol]['qty'] <= 0:
            self.order_history[symbol]['avg_entry_price'] = 0
            self.order_history[symbol]['percentage_change'] = 0
            self.order_history[symbol]['prev_unrealized_pl'] = 0
            self.order_history[symbol]['holdings_ratio'] = 0
            self.order_history[symbol]['unrealized_pl'] = 0
            self.order_history[symbol]['max_drawdown'] = 0
            self.order_history[symbol]['max_unrealized'] = 0

        if self.verbose:
            print(f"Sold {sell_quantity:.2f} shares of {symbol} at ${current_price:.2f} for a gain/loss of ${realized_profit:.2f}, dollar amount: ${sell_dollar_amount:.2f}")

    def _buy(self, symbol, desired_percentage, total_portfolio_value):
        # Clip the desired percentage to make sure it's between 0 and 1
        desired_percentage = np.clip(desired_percentage, 0, 1)

        # Current price for this symbol and current quantity
        current_price = self.time_series_data[symbol]['close'][self.current_step]
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
                try:
                    order = self.live_trader.buy(
                        symbol, purchased_quantity, current_price)

                    # Wait for order to be filled
                    if not self.live_trader.wait_for_order_fill(order.id):
                        # If order is not filled within the MAX_RETRIES, cancel the order
                        self.live_trader.cancel_order(order.id)
                except Exception as e:
                    # Handle exceptions, such as connectivity issues or API errors
                    print(
                        f"An error occurred when trying to buy: {e}")

            self.balance -= total_cost

            # Update the average entry price and quantity directly in the dictionary
            total_quantity = current_quantity + purchased_quantity
            date = self.time_series_data[symbol]['date'][self.current_step]
            self.order_history[symbol]['avg_entry_price'] = (
                (self.order_history[symbol]['avg_entry_price'] * current_quantity) + additional_dollar_amount_needed) / total_quantity
            self.order_history[symbol]['qty'] = total_quantity

            # Calculate the new total invested amount
            new_total_invested = self.order_history[symbol]['total_invested'] + total_cost
            date = self.time_series_data[symbol]['date'][self.current_step]

            # Update the total invested amount
            self.order_history[symbol]['total_invested'] = new_total_invested

            self.trades.append({
                'symbol': symbol,
                'qty': purchased_quantity,
                'close': current_price,
                'cost': total_cost,
                'profit': fee * -1,
                'date': date,
                'side': 'buy'
            })

            if self.verbose:
                print(
                    f"Bought {purchased_quantity:.2f} shares of {symbol} at ${current_price:.2f} for a cost of ${total_cost:.2f}")

    def _calculate_total_portfolio_value(self):
        # Extract quantities and current prices for all symbols
        quantities = np.array([self.order_history[symbol]['qty']
                              for symbol in self.time_series_data.keys()])

        current_prices = np.array(
            [self.time_series_data[symbol]['close'][self.current_step] for symbol in self.time_series_data.keys()])

        # Calculate total portfolio value
        portfolio_value = self.balance + np.sum(quantities * current_prices)

        # return dollar value
        return portfolio_value

    def _calculate_drawdown(self, current_portfolio_value):
        if current_portfolio_value > self.highest_portfolio_value:
            self.highest_portfolio_value = current_portfolio_value
        drawdown = (self.highest_portfolio_value -
                    current_portfolio_value) / self.highest_portfolio_value
        return drawdown if drawdown > 0 else 0  # Return only positive drawdowns

    def _calculate_log_drawdown(self, current_portfolio_value):
        if current_portfolio_value > self.highest_portfolio_value:
            self.highest_portfolio_value = current_portfolio_value
        # Calculate logarithmic drawdown
        log_drawdown = math.log(
            current_portfolio_value / self.highest_portfolio_value)
        # Return positive drawdowns; if you want to keep the drawdown negative, remove the abs() function
        return abs(log_drawdown) if log_drawdown < 0 else 0

    def _calculate_estimated_sharpe_ratio(self, returns_list):
        if len(returns_list) > 0:
            excess_returns = np.array(
                returns_list) - self.risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
            return sharpe_ratio if np.std(excess_returns) > 0 else 0
        else:
            return 0

    def _calculate_skewness(self, returns_list):
        if len(returns_list) > 2:
            return scipy.stats.skew(returns_list)
        else:
            return 0

    def _calculate_kurtosis(self, returns_list):
        if len(returns_list) > 3:
            # Using Fisher’s definition of kurtosis (kurtosis of the normal distribution is zero).
            return scipy.stats.kurtosis(returns_list, fisher=True)
        else:
            return 0

    def _calculate_reward(self):
        reward = 0

        # Sharpe Ratios
        estimated_sr = self._calculate_estimated_sharpe_ratio(
            self.strategy_returns_list)

        benchmark_sr = self._calculate_estimated_sharpe_ratio(
            self.benchmark_returns_list)  # Dynamic benchmark

        # Skewness and Kurtosis
        skewness = self._calculate_skewness(self.strategy_returns_list)
        kurtosis = self._calculate_kurtosis(self.strategy_returns_list)

        # Adjusting the PSR calculation
        square_root_expression = max(
            0, 1 - skewness * estimated_sr + (kurtosis - 1) / 4 * estimated_sr ** 2)
        denominator = np.sqrt(square_root_expression)

        if denominator != 0 and np.isfinite(denominator):
            track_record_length = len(self.strategy_returns_list)
            psr = norm.cdf(np.sqrt(track_record_length - 1) *
                           (estimated_sr - benchmark_sr) / denominator)
            psr = 0 if not np.isfinite(psr) else psr
        else:
            psr = 0

        # Penalize for maximum drawdown and excessive trading
        # or any other functional form
        reward += self._scale_psr(float(psr))

        return reward

    def _scale_psr(self, normalized_psr):

        # Define thresholds
        threshold = 0.69  # 69% of the track record is positive

        # Scale PSR
        if normalized_psr >= threshold:
            # Scale positive PSR to a suitable range, e.g., 0 to 1
            scaled_psr = (normalized_psr - threshold) / (1 - threshold)
        else:
            # Scale negative PSR to a suitable range, e.g., -1 to 0
            scaled_psr = normalized_psr / threshold - 1

        return scaled_psr

    def _calculate_log_return(self, current_portfolio_value):
        if self.last_portfolio_value <= 0:
            return 0
        strategy_log_return = math.log(
            current_portfolio_value / self.last_portfolio_value)
        # Update last portfolio value for next calculation
        self.last_portfolio_value = current_portfolio_value

        return 0 if math.isnan(strategy_log_return) or math.isinf(strategy_log_return) else strategy_log_return

    def _calculate_benchmark_log_return(self):
        self.symbols_log_return = np.array([self.time_series_data[symbol]['log_pct_change'][self.current_step]
                                            for symbol in self.time_series_data], dtype=np.float32)

        return self.symbols_log_return.mean()

    def _update_running_totals(self, strategy_log_return, benchmark_log_return):
        log_return_percentage = (np.exp(strategy_log_return) - 1) * 100
        benchmark_performance_percentage = (
            np.exp(benchmark_log_return) - 1) * 100
        self.running_strategy += log_return_percentage
        self.running_benchmark += benchmark_performance_percentage

    def _update_rendering_history(self, current_portfolio_value):
        self.portfolio_value_history.append(current_portfolio_value)
        self.strategy_history.append(self.running_strategy)
        self.benchmark_history.append(self.running_benchmark)

    def initialize_pygame(self):
        self.window_size = 800
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def _draw_holdings_ratio_chart(self):
        # Adjust the size and position of the bar chart area to take up the remaining bottom section
        bar_chart_width = self.window_size  # Span the entire width of the window
        # Increased height to take up the remaining space
        bar_chart_height = self.window_size // 4
        bar_chart_x = 0  # Start from the left edge of the window
        bar_chart_y = self.window_size - bar_chart_height  # Position at the bottom

        # Background for the bar chart area
        pygame.draw.rect(self.screen, (220, 220, 220), (bar_chart_x,
                         bar_chart_y, bar_chart_width, bar_chart_height))

        # Number of symbols and width for each bar
        num_symbols = len(self.time_series_data.keys())
        bar_width = bar_chart_width / num_symbols

        for i, ratio in enumerate(self.order_history[symbol]['holdings_ratio'] for symbol in self.time_series_data.keys()):
            # Calculate the height of the bar based on the holdings ratio
            bar_height = ratio * bar_chart_height

            # Calculate the position of the bar
            x = bar_chart_x + i * bar_width
            y = bar_chart_y + bar_chart_height - bar_height  # Adjust y-position for the bar

            # Draw the bar
            pygame.draw.rect(self.screen, (0, 128, 0),
                             (x, y, bar_width, bar_height))

            # Draw labels for each bar
            font = pygame.font.SysFont(None, 12)
            label = font.render(self.symbol_names[i], True, (0, 0, 0))
            label_x = x + (bar_width - label.get_width()) / \
                2  # Center the label in the bar
            label_y = y - 20  # Position the label 20 pixels above the bar
            self.screen.blit(label, (label_x, label_y))

    def _draw_portfolio_value_chart(self):
        # Find the maximum value from both histories
        max_value = max(max(self.strategy_history),
                        max(self.benchmark_history))
        min_value = min(min(self.strategy_history),
                        min(self.benchmark_history))

        # Normalize strategy values
        normalized_strategy_values = [(value - min_value) / (max_value - min_value)
                                      for value in self.strategy_history]

        # Normalize benchmark values
        normalized_benchmark_values = [(value - min_value) / (max_value - min_value)
                                       for value in self.benchmark_history]

        # Adjust the chart area
        chart_height = self.window_size * (6 / 8)

        # Plot strategy line
        for i in range(len(normalized_strategy_values) - 1):
            start_pos = (i * (self.window_size / len(self.strategy_history)),
                         chart_height * (1 - normalized_strategy_values[i]))
            end_pos = ((i + 1) * (self.window_size / len(self.strategy_history)),
                       chart_height * (1 - normalized_strategy_values[i + 1]))
            pygame.draw.line(self.screen, (0, 0, 255), start_pos, end_pos, 2)

        # Plot benchmark line
        for i in range(len(normalized_benchmark_values) - 1):
            start_pos = (i * (self.window_size / len(self.benchmark_history)),
                         chart_height * (1 - normalized_benchmark_values[i]))
            end_pos = ((i + 1) * (self.window_size / len(self.benchmark_history)),
                       chart_height * (1 - normalized_benchmark_values[i + 1]))
            pygame.draw.line(self.screen, (255, 0, 0), start_pos, end_pos, 2)

        # Draw legend
        font = pygame.font.SysFont(None, 24)
        strategy_label = font.render('Strategy (Blue)', True, (0, 0, 255))
        benchmark_label = font.render('Benchmark (Red)', True, (255, 0, 0))
        self.screen.blit(strategy_label, (10, 10))  # Adjust position as needed
        # Adjust position as needed
        self.screen.blit(benchmark_label, (10, 35))

    def render(self, done=False, mode='human'):
        if mode == 'human':
            self.screen.fill((255, 255, 255))  # Clear screen (fill with white)

            # self._draw_action_space()
            self._draw_portfolio_value_chart()
            self._draw_holdings_ratio_chart()

            pygame.display.flip()  # Update the full display Surface to the screen
            self.clock.tick(120)  # Limit frames per second

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
