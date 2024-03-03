import math
import numpy as np
from fracdiff import fdiff
import pandas as pd


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
    return (rsi / 100)-0.5


def compute_bbands(data, window=20):
    sma = data.rolling(window=window).mean()
    # Adding a small value to avoid division by zero
    std = data.rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    # Normalized price between 0 and 1
    normalized_price = (data - lower) / (upper - lower)

    return normalized_price-0.5


def compute_log_vwap(data, window=20):
    """
    Compute the log Volume-Weighted Average Price (VWAP) using a rolling window and subtract the log price to give a signal.

    Parameters:
    data (DataFrame): A DataFrame with 'close' and 'volume' columns.
    window (int): The rolling window size.

    Returns:
    Series: A pandas Series containing the log VWAP signal.
    """
    # Calculate rolling VWAP
    vwap = (data['close'] * data['volume']).rolling(window=window).sum() / \
        data['volume'].rolling(window=window).sum()

    # Subtract the log price from log VWAP to get the signal
    log_vwap_signal = np.log(data['close']) - np.log(vwap)

    return log_vwap_signal


def compute_macd(data, span1=12, span2=26, signal=9):
    exp1 = np.log(data).ewm(span=span1, adjust=False).mean()
    exp2 = np.log(data).ewm(span=span2, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - signal_line
    return macd_diff * 10


def compute_rolling_obv(data, window=20):
    """
    Compute a rolling window version of the On-Balance Volume (OBV).

    Parameters:
    data (DataFrame): A DataFrame with 'close' and 'volume' columns.
    window (int): The rolling window size.

    Returns:
    Series: A pandas Series containing the rolling OBV.
    """
    # Calculate the price change
    price_change = data['close'].diff()

    # Determine the direction of trade
    direction = np.where(price_change > 0, 1,
                         np.where(price_change < 0, -1, 0))

    # Apply the direction to the volume
    signed_volume = direction * np.log((data['volume'] * data['close']))

    # Calculate rolling OBV
    rolling_obv = signed_volume.rolling(window=window).sum()

    return rolling_obv / 200


def calculate_historical_volatility(data, window=20):
    volatility = data.rolling(window=window).std()
    return volatility


def calculate_vmar(volume_series):
    """
    Calculate the Volume Moving Average Ratio (VMAR) based on percentage change.

    Parameters:
    volume_series (pandas.Series): Series containing volume data.

    Returns:
    pandas.Series: Series containing VMAR values based on percentage change.
    """
    # Calculate the percentage change in volume
    volume_pct_change = volume_series.pct_change()
    # Calculate log percentage change
    volume_pct_change = np.log1p(volume_pct_change)

    # Replace NaN values (first row after pct_change) with 0 or an appropriate value
    volume_pct_change.fillna(0, inplace=True)

    # VMAR is now the percentage change in volume
    vmar = volume_pct_change
    return vmar / 8


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


def calculate_ema_idvwpm(data, window=20):
    """
    Calculate EMA of f_idvwpm feature for data.

    Parameters:
    data (pandas.DataFrame): DataFrame slice containing 'close' and 'volume' columns.

    Returns:
    pandas.Series: EMA of f_idvwpm.
    """
    price = data['close']
    volume = data['volume']
    dollar_volume = np.log(price * volume)
    log_pct_change = np.log1p(price.pct_change())
    idvwpm = (log_pct_change / dollar_volume) * 1000
    return idvwpm.ewm(span=window, adjust=False).mean()


def calculate_ema_dvwpm(data, window=20):
    """
    Calculate EMA of f_dvwpm feature for data.

    Parameters:
    data (pandas.DataFrame): DataFrame slice containing 'close' and 'volume' columns.

    Returns:
    pandas.Series: EMA of f_dvwpm.
    """
    price = data['close']
    volume = data['volume']
    dollar_volume = np.log(price * volume)
    log_pct_change = np.log1p(price.pct_change())
    dvwpm = (dollar_volume * log_pct_change)
    return dvwpm.ewm(span=window, adjust=False).mean()


def calculate_idvwpm(data):
    """
    Calculate f_idvwpm feature for data.

    Parameters:
    data (pandas.DataFrame): DataFrame slice containing 'close' and 'volume' columns.

    Returns:
    pandas.Series: EMA of f_idvwpm.
    """
    price = data['close']
    volume = data['volume']
    dollar_volume = np.log(price * volume)
    log_pct_change = np.log1p(price.pct_change())
    return (log_pct_change / dollar_volume) * 1000


def calculate_dvwpm(data):
    """
    Calculate f_dvwpm feature for data.

    Parameters:
    data (pandas.DataFrame): DataFrame slice containing 'close' and 'volume' columns.

    Returns:
    pandas.Series: EMA of f_dvwpm.
    """
    price = data['close']
    volume = data['volume']
    dollar_volume = np.log(price * volume)
    log_pct_change = np.log1p(price.pct_change())
    return (dollar_volume * log_pct_change)


def calculate_weighted_ema_log(data, window=20):
    return np.log(data) - np.log(data).ewm(span=window, adjust=False).mean()


def calculate_weighted_ema(data, window=20):
    return data - data.ewm(span=window, adjust=False).mean()


def upper_lower_signal(data, window=20, upper_quantile=0.99, lower_quantile=0.01):
    """
    Generate signals based on whether data is above or below certain quantile thresholds within a rolling window.

    Parameters:
    data (Pandas Series): The data series to analyze.
    window (int): The rolling window size.
    upper_quantile (float): The upper quantile threshold.
    lower_quantile (float): The lower quantile threshold.

    Returns:
    Pandas Series: A series where 1 indicates data is significantly above the upper quantile,
                   0.5 indicates above the mid quantile and below the upper quantile, 
                   -0.5 indicates below the mid quantile and above the lower quantile, 
                   -1 indicates significantly below the lower quantile, and 0 otherwise.
    """

    upper = data.rolling(window=window).quantile(upper_quantile)
    lower = data.rolling(window=window).quantile(lower_quantile)

    # Normalized price between 0 and 1
    normalized_price = (data - lower) / (upper - lower)

    return normalized_price-0.5


def upper_lower_signal_volume(data, window=20, upper_quantile=0.99, lower_quantile=0.01):
    """
    Generate signals based on whether volume data is above or below certain quantile thresholds within a rolling window.

    Parameters:
    data (Pandas Series): The volume data series to analyze.
    window (int): The rolling window size.
    upper_quantile (float): The upper quantile threshold.
    lower_quantile (float): The lower quantile threshold.

    Returns:
    Pandas Series: A series where 2 indicates data is significantly above the upper quantile,
                   1 indicates above the upper quantile, -1 indicates below the lower quantile, 
                   -2 indicates significantly below the lower quantile, and 0 otherwise.
    """
    # Normalize data if necessary
    data = np.log(data)

    upper = data.rolling(window=window).quantile(upper_quantile)
    lower = data.rolling(window=window).quantile(lower_quantile)
    median = data.rolling(window=window).median()

    # Determine midpoints between median and quantiles
    upper_mid = (upper + median) / 2
    lower_mid = (lower + median) / 2

    # Generate signals
    signals = np.where(data >= upper, np.where(data > upper_mid, 1, 0.5),
                       np.where(data < lower, np.where(data <= lower_mid, -1, -0.5), 0))

    return signals


def rolling_low_diff(data, window=20):
    """
    Calculate the percentage difference between the rolling low and the current price.

    Parameters:
    data (Pandas Series): The price data series to analyze.
    window (int): The rolling window size.

    Returns:
    Pandas Series: A series containing the percentage difference between the rolling low and the current price.
    """
    return (data - data.rolling(window=window).min()) / data.rolling(window=window).min()


def rolling_high_diff(data, window=20):
    """
    Calculate the percentage difference between the rolling high and the current price.

    Parameters:
    data (Pandas Series): The price data series to analyze.
    window (int): The rolling window size.

    Returns:
    Pandas Series: A series containing the percentage difference between the rolling high and the current price.
    """
    return (data - data.rolling(window=window).max()) / data.rolling(window=window).max()


def rolling_distance_diff(data, window=20):
    """
    Calculate the distance between the rolling maximum of 'high' and the rolling minimum of 'low'
    within a specified window, either as a percentage of the rolling maximum or as an absolute value.

    Parameters:
    - data: pandas DataFrame with at least two columns: 'high' and 'low'.
    - window: Integer specifying the rolling window size.

    Returns:
    - A pandas Series containing the calculated distances.
    """
    # Calculate rolling max of 'high' and rolling min of 'low'
    rolling_max = data['high'].rolling(window=window).max()
    rolling_min = data['low'].rolling(window=window).min()

    # Normalized price between 0 and 1
    normalized_price = (data['close'] - rolling_min) / \
        (rolling_max - rolling_min)

    return (normalized_price-0.5)*2.5


def rolling_avg_diff(data, window=20):
    """
    Calculate the percentage difference between the rolling average data and the current data.

    Parameters:
    data (Pandas Series): The data data series to analyze.
    window (int): The rolling window size.

    Returns:
    Pandas Series: A series containing the percentage difference between the rolling average data and the current data.
    """
    return (data - data.rolling(window=window).mean()) / data.rolling(window=window).mean()


def calculate_atr(data, period=14, normalize='none'):
    """
    Calculate and optionally normalize the Average True Range (ATR).

    :param data: DataFrame with 'high', 'low', and 'close' columns.
    :param period: Number of periods to use for ATR calculation (default 14).
    :param normalize: Type of normalization ('log', 'percentage', or 'none').
    :return: ATR values as a DataFrame.
    """
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    # Calculate ATR
    atr = true_range.rolling(window=period, min_periods=1).mean()

    # atr = np.log(atr + 1)  # Logarithmic scaling

    atr = (atr / data['close']) * 100  # Percentage scaling

    return atr


def calculate_adx(data, period=14):
    """
    Calculate the Average Directional Index (ADX).

    :param data: DataFrame with 'high', 'low', and 'close' columns.
    :param period: Number of periods to use for the ADX calculation (default 14).
    :return: ADX values as a DataFrame.
    """
    # Calculate True Range and Directional Movement
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()
    tr = pd.DataFrame({'tr': high_low, 'high_close': high_close,
                      'low_close': low_close}).max(axis=1)

    plus_dm = data['high'].diff()
    minus_dm = data['low'].diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Smoothed values
    smoothed_tr = tr.rolling(window=period, min_periods=1).mean()
    smoothed_plus_dm = plus_dm.rolling(window=period, min_periods=1).mean()
    smoothed_minus_dm = minus_dm.rolling(window=period, min_periods=1).mean()

    # Calculate Directional Indicators
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

    # Calculate ADX
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period, min_periods=1).mean()

    return adx


def gen_features(data):
    """
    Generate features for the given data.

    Parameters:
    data (pandas.DataFrame): DataFrame containing stock data.

    Returns:
    pandas.DataFrame: DataFrame containing stock data with added features.
    """
    df = data.copy()

    df['date'] = df.index.get_level_values(1)

    # Calculate normalized dollar volume using z-score
    df['dollar_volume'] = df['close'] * df['volume']

    # Calculate percentage change
    df['percentage_change'] = df.groupby(level=0)[
        'close'].pct_change()

    # Calculate log percentage change
    df['log_pct_change'] = np.log1p(df['percentage_change'])

    df['f_return_1'] = df['log_pct_change']

    # df['f_high_low_diff'] = np.log(df['high'] / df['low']) * 3
    # df['f_open_close_diff'] = np.log(df['open'] / df['close']) * 5

    # df['f_volume'] = df.groupby(level=0)['volume'].transform(
    #     lambda x: calculate_vmar(x))

    # df['f_rolling_return'] = df.groupby(level=0)['f_return_1'].transform(
    #     lambda x: x.rolling(window=24).sum())

    # df['f_rolling_high_low_diff'] = df.groupby(level=0)['f_high_low_diff'].transform(
    #     lambda x: x.rolling(window=24).sum())/8

    # df['f_rolling_f_open_close_diff'] = df.groupby(level=0)['f_open_close_diff'].transform(
    #     lambda x: x.rolling(window=24).sum())

    # df['f_rolling_f_volume'] = df.groupby(level=0)['f_volume'].transform(
    #     lambda x: x.rolling(window=24).sum())

    # df['f_month_dist'] = df.groupby(level=0).apply(
    #     lambda x: rolling_distance_diff(x, window=672)).reset_index(level=0, drop=True)

    df['f_month_high'] = df.groupby(level=0)['high'].transform(
        lambda x: rolling_high_diff(x, window=672))

    df['f_month_low'] = df.groupby(level=0)['low'].transform(
        lambda x: rolling_low_diff(x, window=672))

    # df['f_half_month_dist'] = df.groupby(level=0).apply(
    #     lambda x: rolling_distance_diff(x, window=336)).reset_index(level=0, drop=True)

    df['f_half_month_high'] = df.groupby(level=0)['high'].transform(
        lambda x: rolling_high_diff(x, window=336))

    df['f_half_month_low'] = df.groupby(level=0)['low'].transform(
        lambda x: rolling_low_diff(x, window=336))

    # df['f_week_dist'] = df.groupby(level=0).apply(
    #     lambda x: rolling_distance_diff(x, window=168)).reset_index(level=0, drop=True)

    df['f_week_high'] = df.groupby(level=0)['high'].transform(
        lambda x: rolling_high_diff(x, window=168))

    df['f_week_low'] = df.groupby(level=0)['low'].transform(
        lambda x: rolling_low_diff(x, window=168))

    # df['f_day_dist'] = df.groupby(level=0).apply(
    #     lambda x: rolling_distance_diff(x, window=84)).reset_index(level=0, drop=True)

    df['f_day_high'] = df.groupby(level=0)['high'].transform(
        lambda x: rolling_high_diff(x, window=84))

    df['f_day_low'] = df.groupby(level=0)['low'].transform(
        lambda x: rolling_low_diff(x, window=84))

    # df['f_hour_dist'] = df.groupby(level=0).apply(
    #     lambda x: rolling_distance_diff(x, window=24)).reset_index(level=0, drop=True)

    df['f_hour_high'] = df.groupby(level=0)['high'].transform(
        lambda x: rolling_high_diff(x, window=24))

    df['f_hour_low'] = df.groupby(level=0)['low'].transform(
        lambda x: rolling_low_diff(x, window=24))

    # df['f_price_1'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_sma_diff(x, window=24))

    # df['f_price_2'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_sma_diff(x, window=84))

    # df['f_price_3'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_sma_diff(x, window=168))

    # # df['f_price_4'] = df.groupby(level=0)['close'].transform(
    # #     lambda x: compute_sma_diff(x, window=336))

    # df['f_price_5'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_sma_diff(x, window=672))

    # df['f_volume_1'] = df.groupby(level=0)['volume'].transform(
    #     lambda x: compute_sma_diff(x, window=672)) / 3

    df['f_price_ema_1'] = df.groupby(level=0)['close'].transform(
        lambda x: calculate_weighted_ema_log(x, window=24))

    df['f_price_ema_2'] = df.groupby(level=0)['close'].transform(
        lambda x: calculate_weighted_ema_log(x, window=168))

    df['f_price_ema_3'] = df.groupby(level=0)['close'].transform(
        lambda x: calculate_weighted_ema_log(x, window=672))

    # df['f_volume'] = df.groupby(level=0)['volume'].transform(
    #     lambda x: calculate_weighted_ema_log(x, window=672)) / 4

    # df['f_adx_1'] = df.groupby(level=0).apply(
    #     lambda x: calculate_adx(x, period=24)).reset_index(level=0, drop=True) * 0.015

    # df['f_adx_2'] = df.groupby(level=0).apply(
    #     lambda x: calculate_adx(x, period=84)).reset_index(level=0, drop=True) * 0.015

    # df['f_adx_3'] = df.groupby(level=0).apply(
    #     lambda x: calculate_adx(x, period=168)).reset_index(level=0, drop=True) * 0.015

    # df['f_price_rsi_1'] = df.groupby(level=0)['close'].transform(
    #     lambda x: calculate_rsi(x, window=24))

    # df['f_price_rsi_2'] = df.groupby(level=0)['close'].transform(
    #     lambda x: calculate_rsi(x, window=168))

    # df['f_price_rsi_3'] = df.groupby(level=0)['close'].transform(
    #     lambda x: calculate_rsi(x, window=672))

    # df['f_price_rsi_1'] = df.groupby(level=0)['close'].transform(
    #     lambda x: calculate_rsi(x, window=14)) * 2

    # df['f_price_rsi_2'] = df.groupby(level=0)['close'].transform(
    #     lambda x: calculate_rsi(x, window=24)) * 2

    # df['f_price_rsi_3'] = df.groupby(level=0)['close'].transform(
    #     lambda x: calculate_rsi(x, window=168)) * 3

    # df['f_price_roc_1'] = df.groupby(level=0)['close'].transform(
    #     lambda x: x.pct_change(periods=12))

    # df['f_price_roc_2'] = df.groupby(level=0)['close'].transform(
    #     lambda x: x.pct_change(periods=24))

    # df['f_price_roc_3'] = df.groupby(level=0)['close'].transform(
    #     lambda x: x.pct_change(periods=168))

    # # df['f_price_hist_vol_2'] = df.groupby(level=0)['percentage_change'].transform(
    #     lambda x: calculate_historical_volatility(x, window=24))

    # df['f_price_hist_vol_3'] = df.groupby(level=0)['percentage_change'].transform(
    #     lambda x: calculate_historical_volatility(x, window=168))

    # # df['f_price_hist_vol_4'] = df.groupby(level=0)['percentage_change'].transform(
    # #     lambda x: calculate_historical_volatility(x, window=730))

    # df['f_volume_obv_1'] = df.groupby(level=0).apply(
    #     lambda x: compute_rolling_obv(x, window=24)).reset_index(level=0, drop=True)

    # df['f_volume_obv_2'] = df.groupby(level=0).apply(
    #     lambda x: compute_rolling_obv(x, window=84)).reset_index(level=0, drop=True) * 0.7

    # df['f_volume_obv_3'] = df.groupby(level=0).apply(
    #     lambda x: compute_rolling_obv(x, window=168)).reset_index(level=0, drop=True) * 0.2

    # df['f_bbands_1'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_bbands(x, window=24))

    # df['f_bbands_2'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_bbands(x, window=168))

    # df['f_bbands_3'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_bbands(x, window=672))

    # df['f_bbands_1'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_bbands(x, window=24))

    # df['f_bbands_2'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_bbands(x, window=84))

    # df['f_bbands_3'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_bbands(x, window=168))

    # df['f_bbands_4'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_bbands(x, window=168))

    # df['f_bbands_5'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_bbands(x, window=168))

    # df['f_macd_1'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_macd(x, span1=6, span2=13, signal=3)) * 2.4

    # df['f_macd_2'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_macd(x, span1=12, span2=26, signal=9)) * 1.4

    # df['f_macd_3'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_macd(x, span1=12*7, span2=26*7, signal=9*7))

    # df['f_macd_1'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_macd(x, span1=12, span2=26, signal=9)) * 5

    # df['f_macd_2'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_macd(x, span1=12*6, span2=26*6, signal=9*6)) * 3

    # df['f_macd_3'] = df.groupby(level=0)['close'].transform(
    #     lambda x: compute_macd(x, span1=12*25, span2=26*25, signal=9*25)) * 1.5

    # # # Fractional Difference feature
    # d = 0.65
    # df['f_fractional_difference_price'] = fdiff(
    #     np.log(df['close']), d)

    # # Calculate the z-score for percentage change
    # df['f_percentage_change_minmax'] = df.groupby(
    #     level=0)['percentage_change'].transform(lambda x: min_max_normalize(x))

    # # # compare dollar volume across different stocks
    # df['f_dollar_volume_minmax'] = df.groupby(
    #     level=0)['dollar_volume'].transform(lambda x: min_max_normalize(x))

    # df['f_dvwpm'] = df.groupby(level=0).apply(
    #     lambda x: calculate_dvwpm(x)).reset_index(level=0, drop=True) * 0.12

    # used for labeling and testing the data
    df['future_price'] = df.groupby(
        level=0)['close'].shift(-12)

    # clean up
    df.dropna(inplace=True)

    return df
