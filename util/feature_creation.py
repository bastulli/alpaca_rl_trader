import numpy as np
from fracdiff import fdiff


def compute_sma_diff(data, window=20):
    sma = np.log(data).rolling(window=window).mean()
    return np.log(data) - sma


def calculate_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()

    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return np.sign((rsi / 100)-0.5)


def compute_bbands(data, window=20):
    sma = data.rolling(window=window).mean()
    # Adding a small value to avoid division by zero
    std = data.rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    # Normalized price between 0 and 1
    normalized_price = (data - lower) / (upper - lower)

    return np.sign(normalized_price-0.5)


def compute_macd(data, span1=12, span2=26, signal=9):
    exp1 = data.ewm(span=span1, adjust=False).mean()
    exp2 = data.ewm(span=span2, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return np.sign(macd / signal_line)


def calculate_obv(price, volume, window=20):
    # Calculate the sign of the price difference
    price_change_direction = np.sign(
        price.rolling(window=window).mean().diff())

    # Calculate the rolling average of the volume
    rolling_volume = np.log(volume.rolling(window=window).mean())

    # Calculate OBV by multiplying the direction of price change with the rolling volume
    obv = (price_change_direction * rolling_volume).rolling(window=window).sum()
    return np.sign(obv)


def calculate_historical_volatility(data, window=20):
    log_return = np.log(data / data.shift(1))
    volatility = log_return.rolling(window=window).std() * np.sqrt(252)
    return volatility


def calculate_vmar(volume_series, window=21):
    """
    Calculate the Volume Moving Average Ratio (VMAR).

    Parameters:
    volume_series (pandas.Series): Series containing volume data.
    window (int): Window size for moving average calculation.

    Returns:
    pandas.Series: Series containing VMAR values.
    """
    volume_ma = volume_series.rolling(window=window).mean()
    vmar = volume_series / volume_ma
    return vmar


def calculate_immediate_zscore(data):
    """
    Calculate the z-score for the given data.

    Parameters:
    data (pandas.Series): Series containing data.
    window (int): Window size for moving average calculation.

    Returns:
    pandas.Series: Series containing z-score values.
    """
    return (data - data.mean()) / data.std()


def calculate_rolling_zscore(data, window=20):
    """
    Calculate rolling the z-score for the given data.

    Parameters:
    data (pandas.Series): Series containing data.
    window (int): Window size for moving average calculation.

    Returns:
    pandas.Series: Series containing z-score values.
    """
    return (data - data.rolling(window=window).mean()) / data.rolling(window=window).std()


def min_max_normalize(data):
    """
    Normalize the given data using min-max normalization.

    Parameters:
    data (pandas.Series): Series containing data.

    Returns:
    pandas.Series: Series containing normalized data.
    """
    return (data - data.min()) / (data.max() - data.min())


def gen_features(data):
    """
    Generate features for the given data.

    Parameters:
    data (pandas.DataFrame): DataFrame containing stock data.

    Returns:
    pandas.DataFrame: DataFrame containing stock data with added features.
    """
    df = data.copy()

    multi = 10

    # Calculate normalized dollar volume using z-score
    df['dollar_volume'] = df['price'] * df['volume']

    # Calculate percentage change
    df['percentage_change'] = df.groupby(level=0)[
        'price'].pct_change()

    # Calculate log percentage change
    df['log_pct_change'] = np.log1p(df['percentage_change'])

    ### Call Features starting with f_ ###
    # compare percentage change across different stocks
    df['f_percentage_change_zscore'] = df.groupby(
        level=0)['percentage_change'].transform(lambda x: min_max_normalize(x))

    # compare dollar volume across different stocks
    df['f_dollar_volume_zscore'] = df.groupby(
        level=0)['dollar_volume'].transform(lambda x: min_max_normalize(x))

    # Simple Moving Average difference feature
    df['f_sma'] = df.groupby(level=0)['price'].transform(
        lambda x: compute_sma_diff(x, window=20*multi))

    # Fractional Difference feature
    d = 0.65
    df['f_fractional_difference_price'] = fdiff(
        np.log(df['price']+1), d)

    # Volume Moving Average Ratio feature
    df['f_vmar'] = df.groupby(level=0)['volume'].transform(
        lambda x: calculate_vmar(x, window=20*multi)) / 30

    # Rolling window Cumulative Return feature
    df['f_cumulative_return'] = df.groupby(level=0)['log_pct_change'].transform(
        lambda x: x.rolling(window=20*multi).sum()) * 2

    # scaled log percentage change
    df['f_log_pct_change'] = df['log_pct_change'] * 5

    # rolling z score
    df['f_z_score'] = df.groupby(level=0)['price'].transform(
        lambda x: calculate_rolling_zscore(x, window=20*multi)) / 8

    # RSI
    df['f_rsi'] = df.groupby(level=0)['price'].transform(
        lambda x: calculate_rsi(x, window=14*multi))

    # # Bollinger Bands
    df['f_bbands'] = df.groupby(level=0)['price'].transform(
        lambda x: compute_bbands(x, window=20*multi))

    # # MACD
    df['f_macd'] = df.groupby(level=0)['price'].transform(
        lambda x: compute_macd(x, span1=12*multi, span2=26*multi, signal=9*multi))

    # OBV
    df['f_obv'] = df.groupby(level=0).apply(lambda x: calculate_obv(
        x['price'], x['volume'], window=20*multi)).reset_index(level=0, drop=True)

    # Historical Volatility
    df['f_historical_volatility'] = df.groupby(level=0)['price'].transform(
        lambda x: calculate_historical_volatility(x, window=20*multi))

    # clean up
    df.dropna(inplace=True)

    return df
