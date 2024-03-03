import json
import logging
import os
import numpy as np
import asyncio
from queue import Queue
from stable_baselines3 import PPO
from environment.trading_game_env import TradingGameEnv
from data.data_loader import load_data, SplitOption
from data.historical_data import TimeFrequency, fetch_all_data
from datetime import datetime
from util.alpaca_trade import LiveTrader
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


class RenderClass:
    def __init__(self):
        self.data = {
            'strategy_return': [],
            'benchmark_return': [],
            'trades_data': [],
            'portfolio': [],
            'symbols': [],
            'date': [],
            'step': None,
            'holdings_ratio': [],
            'unrealized_pl': [],
            'order_history': [],
            'reward': None,
            'isRunning': False,
            'max_drawdown': None
        }

        self.window_length = 250
        self.trades_history_length = 50

    def reset(self):
        self.data = {
            'strategy_return': [],
            'benchmark_return': [],
            'trades_data': [],
            'portfolio': [],
            'symbols': [],
            'date': [],
            'step': None,
            'holdings_ratio': [],
            'unrealized_pl': [],
            'order_history': [],
            'reward': None,
            'isRunning': False,
            'max_drawdown': None
        }

    def update_trades_data(self, trades_info):
        """
        Update trades data with new information.
        """
        trades_data = self.prepare_trades_data(trades_info)

        # If there's something to update
        if trades_data:
            # Append new data and maintain the window length
            self.data['trades_data'].extend(trades_data)
            self.data['trades_data'] = sorted(
                self.data['trades_data'], key=lambda x: x['date'])
            self.data['trades_data'] = self.data['trades_data'][-self.trades_history_length:]

    def convert_datetime64(self, value):
        """Convert numpy datetime64 objects to ISO format strings."""
        if isinstance(value, np.datetime64):
            # Convert np.datetime64 to a timestamp (seconds since epoch)
            timestamp = value.astype('int64') / 1e9
            # Convert to a datetime object and then to an ISO format string
            return datetime.utcfromtimestamp(timestamp).isoformat()
        else:
            return value

    def convert_to_serializable(self, obj):
        """
        Recursively convert numpy objects in a dictionary or list to Python native types
        for JSON serialization.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.datetime64):
            return self.convert_datetime64(obj)
        return obj

    def prepare_trades_data(self, trades_info):
        """
        Prepare trades data for updating, converting any special types.
        """
        return [{key: self.convert_datetime64(value) if isinstance(value, np.datetime64)
                 else value.tolist() if isinstance(value, np.ndarray)
                 else value
                for key, value in trade.items()} for trade in trades_info]

    def update_data(self, key, value):
        if key in ['strategy_return', 'benchmark_return', 'date', 'portfolio']:
            self.data[key].append(value)
            self.data[key] = self.data[key][-self.window_length:]

        elif key in ['trades_data']:
            self.update_trades_data(value)

        elif key in ['order_history']:
            self.data[key] = self.convert_to_serializable(value)
        else:
            self.data[key] = value

    def get_data(self):
        # Returns a copy of the data for thread-safety, could be adjusted based on actual needs
        return {
            'strategy_return': self.data['strategy_return'].copy(),
            'benchmark_return': self.data['benchmark_return'].copy(),
            'trades_data': self.data['trades_data'].copy(),
            'portfolio': self.data['portfolio'],
            'symbols': self.data['symbols'].copy(),
            'date': self.data['date'],
            'step': self.data['step'],
            'holdings_ratio': self.data['holdings_ratio'],
            'unrealized_pl': self.data['unrealized_pl'],
            'reward': self.data['reward'],
            'isRunning': self.data['isRunning'],
            'max_drawdown': self.data['max_drawdown'],
            'order_history': json.dumps([self.data['order_history']]),
            'newDataPoint': True,
        }


class TradingBot:
    def __init__(self):
        self.configure_logger()
        # Initialize variables
        self.model = None
        self.env = None
        self.obs = None
        self.running = False
        self.mode = SplitOption.TEST_SPLIT
        self.live_trader = None
        self.run_live_once = False
        self.init_sync = False  # used to sync with live trader on first run
        self.data_queue = asyncio.Queue()  # Asynchronous queue for data communication
        self.data_renderer = RenderClass()

    def configure_logger(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TradingBot')

    async def async_init(self):
        self.load_configuration()
        await self.load_data_and_env()
        await self.load_model()
        self.obs, _ = self.env.reset()

    def load_live_trader(self):
        # Initialize for live trading (you may need to adjust these)
        self.api_key = os.environ.get("APCA_API_KEY_ID")
        self.secret_key = os.environ.get("APCA_API_SECRET_KEY")
        self.polygon_key = os.environ.get("POLYGON_API_KEY")
        self.live_trader = LiveTrader(
            self.api_key, self.secret_key, self.polygon_key)

    def load_configuration(self):
        # Example of loading configurations, replace with your method
        self.FRAMESTACK = 5
        self.MODEL_TYPE = "PPO"
        self.MODEL_TIMESTAMP = "1707861001"
        self.BEST_MODEL_NUMBER = "1966_35"
        self.features = ['f_price_1', 'f_price_2', 'f_price_3',
                         'f_hour_low', 'f_hour_high', 'f_day_low', 'f_day_high', 'f_week_low', 'f_week_high']
        self.crypto_symbols = ['BAT', 'BCH', 'BTC', 'CRV', 'DOT',
                               'ETH', 'GRT', 'LINK', 'LTC', 'MKR', 'UNI', 'XTZ']
        self.symbols = ['X:' + symbol +
                        'USD' for symbol in self.crypto_symbols]
        self.table_name = 'crypto_data_hourly'

    async def load_data_and_env(self):
        try:
            if self.mode == SplitOption.LIVE_MODE:
                self.load_live_trader()
            else:
                self.live_trader = None

            data = load_data(ratio=0.8, split_option=self.mode,
                             symbols=self.symbols, table_name=self.table_name)

            self.env = TradingGameEnv(
                data=data, features=self.features, framestack=self.FRAMESTACK, rendering=False, live_trader=self.live_trader, random_reset=False)
            await self.load_model()
            self.obs, _ = self.env.reset()
        except Exception as e:
            self.logger.error(f"Error loading data and environment: {e}")
            raise

    async def load_model(self):
        try:
            models_dir = f"models/{self.MODEL_TYPE}-{self.MODEL_TIMESTAMP}"
            best_model_path = os.path.join(
                models_dir, f"best_models/best_model_{self.BEST_MODEL_NUMBER}.zip")
            self.model = PPO.load(best_model_path, env=self.env)
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    async def handle_commands(self, command):
        # Handle commands to start, stop, reset the bot or change its mode
        if command == 'start':
            self.running = True
        elif command == 'stop':
            self.running = False
        elif command == 'reset':
            self.data_renderer.reset()  # Reset the data renderer
            self.data_queue = asyncio.Queue()  # Reset the data queue
            await self.load_data_and_env()
        elif command.startswith('set_mode:'):
            _, mode_str = command.split(':')
            if mode_str in ['LIVE_MODE', 'TEST_SPLIT', 'TRAIN_SPLIT', 'NO_SPLIT']:
                # Map the command string to the corresponding SplitOption
                mode_mapping = {
                    'LIVE_MODE': SplitOption.LIVE_MODE,
                    'TEST_SPLIT': SplitOption.TEST_SPLIT,
                    'TRAIN_SPLIT': SplitOption.TRAIN_SPLIT,
                    'NO_SPLIT': SplitOption.NO_SPLIT
                }
                self.mode = mode_mapping.get(mode_str, SplitOption.TEST_SPLIT)
                await self.load_data_and_env()  # Reload data and environment based on the new mode

    def convert_datetime64(self, value):
        """Convert numpy datetime64 objects to ISO format strings."""
        if isinstance(value, np.datetime64):
            # Convert np.datetime64 to a timestamp (seconds since epoch)
            timestamp = value.astype('int64') / 1e9
            # Convert to a datetime object and then to an ISO format string
            return datetime.utcfromtimestamp(timestamp).isoformat()
        else:
            return value

    def run_step(self):
        # Function to encapsulate running a step
        action, _states = self.model.predict(self.obs, deterministic=True)
        self.obs, rewards, done, _, info = self.env.step(action)

        if done:
            self.running = False

        try:

            # Update RenderClass with data
            self.data_renderer.update_data('portfolio', info['portfolio'])
            self.data_renderer.update_data(
                'strategy_return', info['strategy_return'])
            self.data_renderer.update_data(
                'benchmark_return', info['benchmark_return'])
            # Assuming 'trades' is already in a suitable format
            self.data_renderer.update_data('trades_data', info['trades'])
            # Update other data as needed
            self.data_renderer.update_data('symbols', info['symbols'])

            self.data_renderer.update_data('date', str(datetime.utcfromtimestamp(
                info['date'].astype('int64') / 1_000_000_000)))
            self.data_renderer.update_data('step', info['step'])
            self.data_renderer.update_data('holdings_ratio', info['holdings_ratio'].tolist(
            ) if isinstance(info['holdings_ratio'], np.ndarray) else info['holdings_ratio'])
            self.data_renderer.update_data('unrealized_pl', info['unrealized_pl'].tolist(
            ) if isinstance(info['unrealized_pl'], np.ndarray) else info['unrealized_pl'])
            self.data_renderer.update_data('reward', rewards)
            self.data_renderer.update_data('isRunning', self.running)
            self.data_renderer.update_data(
                'order_history', info['order_history'])
            self.data_renderer.update_data(
                'max_drawdown', info['max_drawdown'])
        except Exception as e:
            self.logger.error(f"Error updating RenderClass data: {e}")

    async def run_live_step(self):

        # get current time
        current_time = datetime.utcnow()
        data_to_send = {'isRunning': self.running, 'newData': False}

        # Run trading logic at the beginning of every hour
        if (0 == current_time.minute) and not self.run_live_once:
            self.run_live_once = True

            # Check for new data and update database file
            fetch_all_data(self.symbols, days=3, table_name=self.table_name,
                           frequency=TimeFrequency.HOUR)

            # Load new data into environment from database file
            new_data = load_data(split_option=SplitOption.LIVE_MODE,
                                 symbols=self.symbols, trim_data=True, table_name=self.table_name)

            # Update environment with new live data
            self.env.sync_with_live_trader()  # sync cash and positions

            # Update environment with new data
            self.env.update_data(new_data)

            # Get observation (stacked data features, prices and history)
            self.obs = self.env.next_observation()

            self.run_step()
            data_to_send = self.data_renderer.get_data()  # Get the latest data state

        elif (0 != current_time.minute) and self.run_live_once:
            self.run_live_once = False

        # Place the data in the queue for WebSocket to send
        await self.data_queue.put(data_to_send)

    async def run_backtest_step(self):
        self.run_step()
        data_to_send = self.data_renderer.get_data()  # Get the latest data state
        # Place the data in the queue for WebSocket to send
        await self.data_queue.put(data_to_send)

    async def run_logic_loop(self):
        """Core logic loop of the bot, running independently of WebSocket."""
        while True:
            if self.running:
                if bot.mode == SplitOption.LIVE_MODE:
                    await bot.run_live_step()
                else:
                    await bot.run_backtest_step()
            # Sleep briefly to yield control
            await asyncio.sleep(0.1)


bot = TradingBot()  # Initialize your TradingBot

# Start the bot asynchronously


# Define the lifespan context manager
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    await bot.async_init()
    # Start the bot's logic loop as a background task
    task = asyncio.create_task(bot.run_logic_loop())
    yield  # The application is now running
    task.cancel()  # Cancel the bot's logic loop task when app shuts down

# Create the FastAPI application and pass the lifespan context manager
app = FastAPI(lifespan=app_lifespan)

# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# WebSocket endpoint for live data streaming


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await asyncio.sleep(1)
    # Send initial state sync message upon connection
    initial_state = {
        'type': 'state_sync',  # Indicate that this message is for state synchronization
        'data': {
            'running': bot.running,
            'mode': bot.mode.name,  # Assuming mode is an Enum and we send its name as a string
        }
    }
    # asyncio delay to allow the bot to initialize
    await asyncio.sleep(1)
    await websocket.send_text(json.dumps(initial_state))

    # send last data points
    if not bot.data_queue.empty():
        data_to_send = await bot.data_queue.get()
        await websocket.send_text(json.dumps(data_to_send))

    try:
        while True:
            # Non-blocking check for incoming WebSocket messages
            try:
                # Await a new message with a timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                # Handle commands if a message is received
                await bot.handle_commands(data)
            except asyncio.TimeoutError:
                # If no message is received within the timeout, just continue
                pass
            except WebSocketDisconnect:
                # Handle WebSocket disconnection
                print("Client disconnected")
                break  # Exit the loop if the client disconnects

            # Check if the bot is running and if there's data in the queue to send
            if bot.running and not bot.data_queue.empty():
                data_to_send = await bot.data_queue.get()
                # Optionally, you could also include 'running' and 'mode' in each message,
                # but it might be redundant unless these states change frequently.
                await websocket.send_text(json.dumps(data_to_send))

            # Short sleep to yield control and prevent high CPU usage
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("Client disconnected")
