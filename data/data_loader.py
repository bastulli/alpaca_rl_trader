import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from enum import Enum
from util.feature_creation import gen_features


class SplitOption(Enum):
    NO_SPLIT = 0
    TRAIN_SPLIT = 1
    TEST_SPLIT = 2


def load_data(ratio=0.8, split_option=SplitOption.NO_SPLIT, symbols=[], trim_data=True, table_name='stock_data'):
    """
    Load stock data from a SQLite database, generate features, optionally trim the data, 
    and split it into training and testing sets.

    :param ratio: The proportion of the data to be used for training (default 0.8).
    :param split_option: Enum for choosing between no split, train split, or test split.
    :param symbols: List of stock symbols to filter data (empty list loads all data).
    :param trim_data: Boolean to decide whether to trim data to uniform length (default True).
    :return: Dictionary of processed data for each ticker.
    """

    data = {}
    db_file = 'historical_time_series.db'
    try:
        with sqlite3.connect(db_file) as conn:
            if symbols:
                # Prepare a string of comma-separated symbols for the SQL query
                symbols_str = ','.join(f"'{symbol}'" for symbol in symbols)
                query = f"SELECT * FROM {table_name} WHERE ticker IN ({symbols_str})"
            else:
                query = f"SELECT * FROM {table_name}"

            # Load data into DataFrame
            df = pd.read_sql_query(query, conn, index_col=[
                                   'ticker', 'timestamp'], parse_dates=['timestamp'])

            # Get a list of unique tickers
            tickers = list(set(df.index.get_level_values(0)))

    except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
        print(f"Error connecting to database or executing query: {e}")
        return {}

    # Sort the DataFrame and generate features
    df.sort_index(inplace=True)
    df = gen_features(df)

    # Calculate original lengths before trimming
    original_lengths = {symbol: len(df.loc[symbol]) for symbol in tickers}

    if trim_data:
        # Find the minimum length of price data among all tickers
        min_length = min(original_lengths.values())

        # Identify the symbol(s) with the minimum length
        min_length_symbols = [
            symbol for symbol, length in original_lengths.items() if length == min_length]

        # Handle the case where all symbols are the same length
        if len(min_length_symbols) == len(tickers):
            print("All symbols have the same length, no trimming required.")
        else:
            # Print the symbol(s) that led to trimming
            print(
                f"Trimming based on symbol(s): {', '.join(min_length_symbols)}")

        # Trimming data to the smallest length from the most recent date
        df = pd.concat([df.loc[ticker].iloc[-min_length:]
                       for ticker in tickers], keys=tickers, names=['ticker', 'timestamp'])

        # Calculate and print the trimmed percentage for each symbol
        for ticker in tickers:
            trimmed_length = len(df.loc[ticker])
            trimmed_percentage = (
                (original_lengths[ticker] - trimmed_length) / original_lengths[ticker]) * 100
            print(f"{ticker}: Trimmed {trimmed_percentage:.2f}%")

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

        # No split (All data)
        else:
            data[ticker] = ticker_data.dropna()

    # Convert each DataFrame column to a NumPy array and store in a new dictionary
    data_dict = {}
    for ticker, df_ticker in data.items():
        data_dict[ticker] = {}
        for col in df_ticker.columns:
            data_dict[ticker][col] = df_ticker[col].values

    return data_dict
