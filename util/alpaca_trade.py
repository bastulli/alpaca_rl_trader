import sqlite3
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

from polygon import ReferenceClient

MAX_RETRIES = 5


class LiveTrader:
    def __init__(self, api_key, secret_key, polygon_key):
        self.trading_client = TradingClient(api_key, secret_key)
        self.stock_client = StockHistoricalDataClient(api_key, secret_key)
        self.polygon_client = ReferenceClient(polygon_key)
        self._init_db()  # Initialize database

    def _init_db(self):
        self.conn = sqlite3.connect('trading_data.db')
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS framestack (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          obs BLOB)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS account_info (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          balance REAL,
                          trades TEXT)''')
        self.conn.commit()

    def convert_cyrpto_ticker_symbol(self, symbol):
        if 'X:' in symbol:
            # Remove the 'X:' prefix and replace 'USD' with '/USD'
            return symbol.replace('X:', '')
        else:
            return symbol

    def buy(self, symbol, qty, price=None):
        symbol = self.convert_cyrpto_ticker_symbol(symbol)

        print(f"Buying {qty:.2f} shares of {symbol} at {price:.2f}")
        # preparing orders
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )

        # Market order
        return self.trading_client.submit_order(
            order_data=market_order_data
        )

    def get_qty_for_liquidation(self, symbol):
        positions = self.get_all_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return float(pos.qty_available)
        return None

    def sell(self, symbol, qty, price=None, liquidate=False):
        symbol = self.convert_cyrpto_ticker_symbol(symbol)

        if liquidate:
            print(f'{symbol} being fully liquidated')
            qty = self.get_qty_for_liquidation(symbol)
            if qty is None:
                print(f"No position found for {symbol} to liquidate.")
                return None

        print(f"Selling {qty:.2f} shares of {symbol} at {price:.2f}")

        # preparing orders
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )

        # Market order
        return self.trading_client.submit_order(
            order_data=market_order_data
        )

    def get_all_positions(self):
        return self.trading_client.get_all_positions()

    def get_account(self):
        return self.trading_client.get_account()

    def check_order(self, order_id):
        return self.trading_client.get_order_by_id(order_id)

    def cancel_order(self, order_id):
        return self.trading_client.cancel_order_by_id(order_id)

    def wait_for_order_fill(self, order_id):
        """ Waits for the order to be filled, checking the status every second for a maximum of MAX_RETRIES times. """
        for _ in range(MAX_RETRIES):
            time.sleep(1)
            order_status = self.check_order(order_id)
            if order_status.status == 'FILLED':
                return True
        return False
