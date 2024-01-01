import numpy as np
from fracdiff import fdiff


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = (100 - (100 / (1 + rs)))/100
    return rsi


def compute_macd(data, span1=12, span2=26, signal=9):
    exp1 = data.ewm(span=span1, adjust=False).mean()
    exp2 = data.ewm(span=span2, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_h = macd - signal_line
    return macd_h / 5


def compute_bbands(data, window=20):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / sma


def compute_sma_diff(data, window=20):
    sma = np.log(data).rolling(window=window).mean()
    return np.log(data) - sma


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


def calculate_vcr(self, volume_series):
    """
    Calculate the Volume Change Rate (VCR).

    Parameters:
    volume_series (pandas.Series): Series containing volume data.

    Returns:
    pandas.Series: Series containing VCR values.
    """
    vcr = volume_series.pct_change()
    return vcr


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


def gen_features(data):
    """
    Generate features for the given data.

    Parameters:
    data (pandas.DataFrame): DataFrame containing stock data.

    Returns:
    pandas.DataFrame: DataFrame containing stock data with added features.
    """
    df = data.copy()
    # Calculate normalized dollar volume using z-score
    df['dollar_volume'] = df['price'] * df['volume']

    # Calculate percentage change
    df['percentage_change'] = df.groupby(level=0)[
        'price'].pct_change()
    df['log_pct_change'] = np.log1p(df['percentage_change'])

    # compare percentage change across different stocks
    df['f_percentage_change_zscore'] = df.groupby(
        level=0)['percentage_change'].transform(lambda x: calculate_immediate_zscore(x)) / 10

    # Calculate simple moving average difference
    df['f_sma_diff_10'] = df.groupby(level=0)['price'].transform(
        lambda x: compute_sma_diff(x, window=10)) * 3

    # Calculate Volitility
    df['f_volitility_10'] = df.groupby(level=0)['log_pct_change'].transform(
        lambda x: x.rolling(window=10).std()) * 10

    # Calculate Volume Moving Average Ratio
    df['f_vmar_10'] = df.groupby(level=0)['volume'].transform(
        lambda x:  calculate_vmar(x, window=10)) / 5

    # Example usage
    d = 0.7  # Degree of differencing
    df['f_fractional_difference_price'] = fdiff(
        np.log(df['price']), d)

    # clean up
    df.dropna(inplace=True)

    return df
