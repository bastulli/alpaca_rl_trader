from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

from polygon import RESTClient
import pandas as pd


class LiveTrader:
    def __init__(self, api_key, secret_key, polygon_key, symbols):
        self.trading_client = TradingClient(api_key, secret_key)
        self.stock_client = StockHistoricalDataClient(api_key, secret_key)
        self.polygon_client = RESTClient(polygon_key)
        self.symbols = symbols

    def buy(self, symbol, qty):
        print(f"Buying {qty} shares of {symbol}")
        # preparing orders
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        # Market order
        return self.trading_client.submit_order(
            order_data=market_order_data
        )

    def sell(self, symbol, qty):
        print(f"Selling {qty} shares of {symbol}")
        # preparing orders
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        # Market order
        return self.trading_client.submit_order(
            order_data=market_order_data
        )

    def get_all_positions(self):
        return self.trading_client.get_all_positions()

    def get_account(self):
        return self.trading_client.get_account()
