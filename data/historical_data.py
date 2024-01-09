import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from dotenv import load_dotenv
import os

# Define the time frequency enum


class TimeFrequency(Enum):
    DAY = 'day'
    HOUR = 'hour'
    MINUTE = 'minute'


# Load environment variables
load_dotenv('keys.env')

polygon_key = os.environ.get("POLYGON_API_KEY")
base_url = 'https://api.polygon.io/v2'

# Function to create a new database connection


def new_connection():
    return sqlite3.connect('historical_time_series.db')


def create_table(table_name='stock_data'):
    conn = new_connection()
    with conn:
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                ticker TEXT,
                timestamp DATETIME,
                volume INTEGER,
                price REAL,
                PRIMARY KEY (ticker, timestamp)
            );
        ''')
    conn.close()


def batch_insert_data(conn, data, table_name='stock_data'):
    with conn:
        conn.executemany(
            f'INSERT OR REPLACE INTO {table_name} (ticker, timestamp, volume, price) VALUES (?, ?, ?, ?)', data)


def fetch_data_for_ticker(ticker, start_date, end_date, frequency=TimeFrequency.DAY, table_name='stock_data'):
    conn = new_connection()  # Create a new connection for each thread
    attempts = 0
    start = start_date
    batch_data = []  # List to hold data for batch insert
    max_attempts = 3  # Max attempts for retries

    interval_length = {
        TimeFrequency.DAY: 365,   # Fetch one year of data at a time for 'day'
        TimeFrequency.HOUR: 7,    # Fetch one week of data at a time for 'hour'
        TimeFrequency.MINUTE: 1   # Fetch one day of data at a time for 'minute'
    }
    while start < end_date:
        try:
            # Adjust the batch interval based on the frequency
            days_to_fetch = interval_length[frequency]
            batch_end_date = min(
                start + timedelta(days=days_to_fetch), end_date)
            start_str = start.strftime('%Y-%m-%d')
            end_str = batch_end_date.strftime('%Y-%m-%d')

            print(
                f"Fetching {ticker} data from {start_str} to {end_str} at {frequency.value} frequency")
            response = requests.get(
                f"{base_url}/aggs/ticker/{ticker}/range/1/{frequency.value}/{start_str}/{end_str}?apiKey={polygon_key}")

            response.raise_for_status()

            # Break if no data is returned
            if 'results' not in response.json() or response.json()['resultsCount'] == 0:
                print(
                    f"No data returned for {ticker} from {start_str} to {end_str}")
                # Update the start date for the next batch
                start = batch_end_date
                continue

            data = response.json()['results']

            for freq in data:
                # Convert timestamp to milliseconds
                milliseconds = freq['t'] / 1000

                # Convert milliseconds to datetime object
                date_time = datetime.utcfromtimestamp(milliseconds)

                # Format the datetime object to your desired format
                date = date_time.strftime('%Y-%m-%d %H:%M:%S')

                volume = freq['v']
                price = freq['c']

                # Add the data to the batch list
                batch_data.append((ticker, date, volume, price))

            # Batch insert when a certain size is reached or at the end of data
            if start >= end_date - timedelta(days=1) or len(batch_data) >= 1000 or batch_end_date == end_date:
                batch_insert_data(
                    conn=conn, data=batch_data, table_name=table_name)
                batch_data.clear()

            # Update the start date for the next batch
            start = batch_end_date

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            attempts += 1
            if attempts >= max_attempts:
                print(
                    f"Skipping {ticker} after {max_attempts} failed attempts.")
                break

    if batch_data:  # Insert any remaining data in the batch
        batch_insert_data(conn=conn, data=batch_data)
    conn.close()


def get_date_range(ticker, table_name='stock_data'):
    conn = new_connection()
    with conn:
        query = f"SELECT MIN(timestamp) as start_date, MAX(timestamp) as end_date FROM {table_name} WHERE ticker = '{ticker}'"
        date_range = pd.read_sql_query(query, conn)
    conn.close()
    return date_range.iloc[0]


def fetch_all_data(symbols, days=90, table_name='stock_data', frequency=TimeFrequency.DAY):
    end_date = datetime.now()
    tasks = []

    for ticker in symbols:
        start_date = end_date - timedelta(days=days)
        date_range = get_date_range(ticker, table_name)

        # logic to determine if redownload is needed...
        if date_range is not None and not date_range['end_date'] is None and not date_range['start_date'] is None:
            start_of_database = datetime.strptime(
                date_range['start_date'], '%Y-%m-%d %H:%M:%S')
            end_of_database = datetime.strptime(
                date_range['end_date'], '%Y-%m-%d %H:%M:%S')

            # Decide if a full redownload is needed
            redownload = start_date < start_of_database

            # Update the start date for downloading only missing data
            if end_date > end_of_database and not redownload:
                start_date = end_of_database + timedelta(days=1)
                print(f"Updating data, adding missing days for {ticker}")
            elif end_date <= end_of_database:
                print(f"No new data to download for {ticker}")
                continue
        else:
            redownload = True

        if redownload:
            print(
                f"Redownloading full dataset for {ticker} from {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(
                f"Downloading data for {ticker} from {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        # Schedule the task
        tasks.append((ticker, start_date, end_date))

    with ThreadPoolExecutor(max_workers=5) as executor:
        for ticker, start, end in tasks:
            executor.submit(fetch_data_for_ticker, ticker,
                            start, end, frequency, table_name)


if __name__ == '__main__':

    crypto_symbols = ['AAVE', 'AVAX', 'BAT', 'BCH', 'BTC', 'CRV',
                      'DOT', 'ETH', 'GRT', 'LINK', 'LTC', 'MKR', 'SHIB', 'UNI', 'XTZ']
    # Using list comprehension to modify each element in the list
    symbols = ['X:' + symbol + 'USD' for symbol in crypto_symbols]

    table_name = 'crypto_data_hourly'  # 'stock_data' is the default table name
    create_table(table_name)
    fetch_all_data(symbols, days=90, table_name=table_name,
                   frequency=TimeFrequency.HOUR)
