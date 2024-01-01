import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('keys.env')

polygon_key = os.environ.get("POLYGON_API_KEY")
base_url = 'https://api.polygon.io/v2'

# Function to create a new database connection


def new_connection():
    return sqlite3.connect('historical_stock_data.db')


def create_table():
    conn = new_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                ticker TEXT,
                timestamp DATETIME,
                volume INTEGER,
                price REAL,
                PRIMARY KEY (ticker, timestamp)
            );
        ''')
    conn.close()


def batch_insert_data(conn, data):
    with conn:
        conn.executemany(
            'INSERT OR REPLACE INTO stock_data (ticker, timestamp, volume, price) VALUES (?, ?, ?, ?)', data)


def fetch_data_for_ticker(ticker, start_date, end_date):
    conn = new_connection()  # Create a new connection for each thread
    attempts = 0
    start = start_date
    batch_data = []  # List to hold data for batch insert
    max_attempts = 3  # Max attempts for retries

    while start < end_date:
        try:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            print(f"Fetching {ticker} data from {start_str} to {end_str}")

            response = requests.get(
                f"{base_url}/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}?apiKey={polygon_key}")
            response.raise_for_status()

            data = response.json()['results']
            for day in data:
                date = datetime.fromtimestamp(
                    day['t'] / 1000).strftime('%Y-%m-%d')
                volume = day['v']
                price = day['c']
                batch_data.append((ticker, date, volume, price))

            # Batch insert when a certain size is reached or at the end of data
            if len(batch_data) >= 500 or start >= end_date - timedelta(days=1):
                batch_insert_data(conn, batch_data)
                batch_data.clear()

            days = min(1830, (end_date - start).days)
            start += timedelta(days=days)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            attempts += 1
            if attempts >= max_attempts:
                print(
                    f"Skipping {ticker} after {max_attempts} failed attempts.")
                break

    if batch_data:  # Insert any remaining data in the batch
        batch_insert_data(conn, batch_data)
    conn.close()


def get_date_range(ticker):
    conn = new_connection()
    with conn:
        query = f"SELECT MIN(timestamp) as start_date, MAX(timestamp) as end_date FROM stock_data WHERE ticker = '{ticker}'"
        date_range = pd.read_sql_query(query, conn)
    conn.close()
    return date_range.iloc[0]


def fetch_all_data(symbols, days=90):
    end_date = datetime.now()
    tasks = []

    for ticker in symbols:
        start_date = end_date - timedelta(days=days)
        date_range = get_date_range(ticker)

        # logic to determine if redownload is needed...
        if date_range is not None and not date_range['end_date'] is None and not date_range['start_date'] is None:
            start_of_database = datetime.strptime(
                date_range['start_date'], '%Y-%m-%d')
            end_of_database = datetime.strptime(
                date_range['end_date'], '%Y-%m-%d')

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
                f"Redownloading full dataset for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        else:
            print(
                f"Downloading data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        # Schedule the task
        tasks.append((ticker, start_date, end_date))

    with ThreadPoolExecutor(max_workers=5) as executor:
        for ticker, start, end in tasks:
            executor.submit(fetch_data_for_ticker, ticker, start, end)


if __name__ == '__main__':
    create_table()
    symbols = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC',
               'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT']  # Example tickers
    fetch_all_data(symbols, days=4000)
