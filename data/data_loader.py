import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from enum import Enum
from util.feature_creation import gen_features
import os


class SplitOption(Enum):
    NO_SPLIT = 0
    TRAIN_SPLIT = 1
    TEST_SPLIT = 2


def load_data(ratio=0.8, split_option=SplitOption.NO_SPLIT, symbols=[], trim_data=True):
    data = {}
    db_file = 'historical_stock_data.db'

    with sqlite3.connect(db_file) as conn:
        if not symbols:
            # Fetch all data
            query = "SELECT * FROM stock_data"
        else:
            # Fetch data for specified symbols
            symbols_str = ','.join(f"'{symbol}'" for symbol in symbols)
            query = f"SELECT * FROM stock_data WHERE ticker IN ({symbols_str})"

        # Load data into DataFrame
        df = pd.read_sql_query(query, conn, index_col=[
                               'ticker', 'timestamp'], parse_dates=['timestamp'])
        tickers = list(set(df.index.get_level_values(0)))

    # Sort the MultiIndex
    df.sort_index(inplace=True)

    # Generate features
    df = gen_features(data=df)

    if trim_data:
        # Find the minimum length of price data among all tickers
        min_length = min(len(df.loc[symbol]) for symbol in tickers)

        # Trimming data to the smallest length from the most recent date
        df = pd.concat([df.loc[ticker].iloc[-min_length:]
                       for ticker in tickers], keys=tickers, names=['ticker', 'timestamp'])

    # Create a dictionary of DataFrames, one for each ticker
    symbol_dict = {symbols: df.xs(symbols) for symbols in tickers}

    # Split data into train and test sets for each ticker
    for ticker in symbol_dict:
        ticker_data = df.xs(ticker)
        ticker_train, ticker_test = train_test_split(
            ticker_data, train_size=ratio, shuffle=False)

        if split_option == SplitOption.TRAIN_SPLIT:
            data[ticker] = ticker_train.dropna()

        elif split_option == SplitOption.TEST_SPLIT:
            data[ticker] = ticker_test.dropna()

        else:
            data[ticker] = ticker_data.dropna()

    # Convert each DataFrame column to a NumPy array and store in a new dictionary
    data_dict = {}
    for ticker, df_ticker in data.items():
        data_dict[ticker] = {}
        for col in df_ticker.columns:
            data_dict[ticker][col] = df_ticker[col].values

    return data_dict
