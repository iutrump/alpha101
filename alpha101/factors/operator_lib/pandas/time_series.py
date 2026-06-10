from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "ts_sum",
    "sma",
    "ts_ema",
    "ts_std_dev",
    "ts_median",
    "ts_ewma",
    "correlation",
    "ts_pctchange",
    "ts_covariance",
    "ts_cov",
    "ts_rank",
    "rolling_prod",
    "ts_product",
    "product",
    "ts_min",
    "ts_max",
    "ts_delta",
    "delta",
    "ts_delay",
    "delay",
    "ts_arg_max",
    "ts_argmax",
    "ts_arg_min",
    "ts_argmin",
    "ts_decay_linear",
    "decay_linear",
    "ts_skewness",
    "ts_kurtosis",
    "ts_quantile",
    "atr",
    "atr_sma",
    "ts_zscore",
    "ts_pos",
    "ts_drawdown",
    "ts_slope",
    "ts_mean",
    "ts_std",
    "ts_corr",
    "ts_volatility",
    "ts_percentile",
    "ts_vwap",
    "signed_power",
]

def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    
    return df.rolling(window).sum()

def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()

def ts_ema(df, window=10):
    """
    Wrapper function to estimate EMA.
    :param df: a pandas DataFrame or Series.
    :param window: the exponential moving average span.
    :return: a pandas DataFrame/Series with the EMA over the past 'window' periods.
    """
    return df.ewm(span=window, adjust=False, min_periods=window).mean()

def ts_std_dev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()

def ts_median(df: pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling median.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).median()

def ts_ewma(df: pd.DataFrame, span=10):
    """
    Wrapper function to estimate exponentially weighted moving average.
    :param df: a pandas DataFrame.
    :param span: the span for EWMA.
    :return: a pandas DataFrame with the exponentially weighted moving average.
    """
    return df.ewm(span=span, adjust=False).mean()

def correlation(x: pd.DataFrame, y: pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    sum_x = x.rolling(window, min_periods=window).sum()
    sum_y = y.rolling(window, min_periods=window).sum()
    sum_xy = (x * y).rolling(window, min_periods=window).sum()
    sum_x2 = (x * x).rolling(window, min_periods=window).sum()
    sum_y2 = (y * y).rolling(window, min_periods=window).sum()

    cov_xy = sum_xy - (sum_x * sum_y) / window
    var_x = sum_x2 - (sum_x * sum_x) / window
    var_y = sum_y2 - (sum_y * sum_y) / window
    denom = np.sqrt(var_x * var_y)
    corr = cov_xy / denom
    return corr.where((var_x > 0) & (var_y > 0))

def ts_pctchange(df: pd.DataFrame, period=1):
    """
    Wrapper function to estimate percentage change.
    :param df: a pandas DataFrame.
    :param period: the period for percentage change.
    :return: a pandas DataFrame with the percentage change over the past 'period' days.
    """
    return df.pct_change(periods=period)

def ts_covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    sum_x = x.rolling(window, min_periods=window).sum()
    sum_y = y.rolling(window, min_periods=window).sum()
    sum_xy = (x * y).rolling(window, min_periods=window).sum()
    return sum_xy / window - (sum_x / window) * (sum_y / window)

def ts_rank(df, window=10):
    """
    Fast vectorized time-series rank (percentile rank of last value in window)
    """
    arr = df.values
    T, N = arr.shape
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, T):
        window_slice = arr[i - window + 1:i + 1]
        current = window_slice[-1]
        rank = (window_slice <= current).sum(axis=0) / window
        result[i] = rank
    return pd.DataFrame(result, index=df.index, columns=df.columns)

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def ts_product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Fast rolling product using log-sum-exp with sign/zero handling.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    window : int, default 10
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Rolling product over the past `window` rows.

    Notes
    -----
    This implementation handles:
    - zeros: any zero in the window -> product = 0
    - negatives: sign determined by parity of negative count
    - NaNs: same behavior as rolling(window) default, i.e. require `window` valid observations
    - overflow/underflow: exponent input is clipped to float-safe range
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")

    arr = df.to_numpy(dtype=np.float64, copy=False)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.shape[0] >= window:
        windows = sliding_window_view(arr, window_shape=window, axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            out[window - 1:] = np.prod(windows, axis=2)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


product = ts_product

def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()

def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling max.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()

def ts_delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)

def ts_delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)


delta = ts_delta
delay = ts_delay

def ts_arg_max(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    arr = df.values
    T, N = arr.shape

    result = np.full((T, N), np.nan)

    for i in range(window - 1, T):
        window_slice = arr[i - window + 1:i + 1]

        # numpy argmax 在 axis=0 上操作
        argmax = np.argmax(window_slice, axis=0)

        # +1 follows the original Alpha101 convention.
        result[i] = argmax + 1

    return pd.DataFrame(result, index=df.index, columns=df.columns)

def signed_power(df: pd.DataFrame | pd.Series, power: float) -> pd.DataFrame | pd.Series:
    """Return sign(x) * abs(x) ** power while preserving the input shape."""
    return np.sign(df) * np.power(np.abs(df), power)


def ts_arg_min(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    arr = df.values
    T, N = arr.shape

    result = np.full((T, N), np.nan)

    for i in range(window - 1, T):
        window_slice = arr[i - window + 1:i + 1]

        # numpy argmin 在 axis=0 上操作
        argmin = np.argmin(window_slice, axis=0)

        # +1 follows the original Alpha101 convention.
        result[i] = argmin + 1

    return pd.DataFrame(result, index=df.index, columns=df.columns)


ts_argmax = ts_arg_max
ts_argmin = ts_arg_min

def ts_decay_linear(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Linear decay (weights increase linearly from past to present).
    Weight for oldest point = 1, newest = period.
    Returns same shape as input, with NaN for first (period-1) rows.
    """
    weights = np.arange(1, period + 1, dtype=np.float32)  # [1, 2, ..., period]
    weights /= weights.sum()

    arr = df.values
    T, N = arr.shape

    result = np.full((T, N), np.nan)

    for col in range(N):
        x = arr[:, col]

        # 跳过全nan列
        if np.all(np.isnan(x)):
            continue

        # 卷积（mode='valid' 对应完整窗口）
        conv = np.convolve(x, weights, mode='valid')

        result[period-1:, col] = conv

    return pd.DataFrame(result, index=df.index, columns=df.columns)


decay_linear = ts_decay_linear

def ts_skewness(x: pd.DataFrame, window=7):
    """
    计算滚动窗口内的偏度
    偏度 > 0: 右偏（长右尾）
    偏度 < 0: 左偏（长左尾）
    """
    return x.rolling(window=window).skew()

def ts_kurtosis(x: pd.DataFrame, window):
    """
    计算滚动窗口内的峰度
    峰度 > 3: 尖峰厚尾
    峰度 < 3: 平峰薄尾
    fisher=True 返回超额峰度(峰度-3)
    """
    return x.rolling(window=window).kurt()

def ts_quantile(x: pd.DataFrame, window=7, q=0.5):
    """
    计算滚动窗口内的分位数
    q: 0~1之间，如0.5为中位数，0.25为下四分位
    """
    return x.rolling(window=window).quantile(q)


def ts_percentile(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling percentile rank of the current value in each lookback window."""
    arr = df.to_numpy(dtype=np.float64, copy=False)
    result = np.full(arr.shape, np.nan, dtype=np.float64)
    if window <= 1:
        return pd.DataFrame(result, index=df.index, columns=df.columns)
    for i in range(window - 1, arr.shape[0]):
        window_slice = arr[i - window + 1:i + 1]
        current = window_slice[-1]
        finite = np.isfinite(window_slice) & np.isfinite(current)
        counts = finite.sum(axis=0)
        values = ((window_slice < current) & finite).sum(axis=0) / np.maximum(counts - 1, 1)
        values[counts < window] = np.nan
        result[i] = values
    return pd.DataFrame(result, index=df.index, columns=df.columns)

def atr(self, window=14):
    """
    Calculate Average True Range using Wilder's smoothing.
    """
    # Step 1: Compute True Range
    prev_close = self.close.shift(1)
    tr1 = self.high - self.low
    tr2 = (self.high - prev_close).abs()
    tr3 = (self.low - prev_close).abs()
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    # Step 2: Initialize ATR with SMA of first 'window' TR values
    atr = pd.Series(index=tr.index, dtype='float64')
    atr.iloc[:window] = np.nan
    atr.iloc[window - 1] = tr.iloc[:window].mean()

    # Step 3: Apply Wilder's smoothing
    for i in range(window, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (window - 1) + tr.iloc[i]) / window

    return atr

def atr_sma(self, window=14):
    """Simplified ATR using simple moving average."""
    prev_close = self.close.shift(1)
    tr = pd.DataFrame({
        'tr1': self.high - self.low,
        'tr2': (self.high - prev_close).abs(),
        'tr3': (self.low - prev_close).abs()
    }).max(axis=1)
    return sma(tr, window=window)  # 使用你已有的 sma 函数

def ts_zscore(df, window=10, eps=1e-8):
    """
    Time-series zscore for pandas DataFrame or Series.
    """
    df = df.replace([np.inf, -np.inf], np.nan)

    mean = df.rolling(window=window, min_periods=window).mean()
    std = df.rolling(window=window, min_periods=window).std()

    std = std.where(std.abs() > eps)

    out = (df - mean) / std
    return out.replace([np.inf, -np.inf], np.nan)


def ts_volatility(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.pct_change(fill_method=None).rolling(window=window, min_periods=window).std()


def ts_vwap(vwap: pd.DataFrame, volume: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    dollar_volume = (vwap * volume).rolling(window=window, min_periods=window).sum()
    volume_sum = volume.rolling(window=window, min_periods=window).sum()
    return dollar_volume / volume_sum.replace(0, np.nan)

def ts_pos(x: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Position in Range: (x - rolling_min) / (rolling_max - rolling_min)
    
    描述：当前值在滚动窗口内的相对位置，常用于识别极端状态。
    特点：结构信息算子，稳定有效，尤其在震荡市中表现优异。
    
    Parameters
    ----------
    x : array-like or pd.Series
        输入价格序列（如收盘价）
    window : int
        滚动窗口长度
        
    Returns
    -------
    pd.Series
        值域 [0, 1]，若 max == min 则返回 NaN
    """
    roll_min = x.rolling(window=window).min()
    roll_max = x.rolling(window=window).max()
    numerator = x - roll_min
    denominator = roll_max - roll_min
    
    # 避免除零：当 max == min 时，设为 NaN
    pos = numerator / denominator
    pos = pos.where(denominator != 0, np.nan)
    return pos

def ts_drawdown(x: pd.DataFrame, window: int=7) -> pd.DataFrame:
    """
    Rolling Drawdown: x / rolling_max - 1
    
    描述：过去 window 期内的最大回撤（负值），用于识别趋势衰竭。
    应用：趋势 exhaustion 信号，常与动量因子正交使用。
    
    Parameters
    ----------
    x : array-like or pd.Series
        输入价格序列（通常为累计净值或价格）
    window : int
        滚动窗口长度
        
    Returns
    -------
    pd.DataFrame
        回撤值（≤ 0），创新高时为 0
    """
    roll_max = x.rolling(window=window).max()
    drawdown = x / roll_max - 1.0
    # 确保不会因数值误差出现正值
    drawdown = drawdown.clip(upper=0.0)
    return drawdown

def ts_slope(x: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Rolling Linear Regression Slope: slope of x over rolling window

    描述：计算序列在过去 window 期内的线性回归斜率。
    含义：衡量过去 window 期内 x 的线性趋势方向和强度。
        - slope > 0：上升趋势
        - slope < 0：下降趋势
        - slope 接近 0：趋势不明显

    特点：
    - 趋势结构算子，比简单的 x - delay(x, window) 更平滑；
    - 适合用于价格、成交量、波动率等序列的趋势刻画；
    - 对分钟级 K 线中的“持续走高 / 持续走弱”判断比较有用。

    Parameters
    ----------
    x : pd.Series or pd.DataFrame
        输入序列，例如 high、volume、close 等。
    window : int
        滚动窗口长度。

    Returns
    -------
    pd.Series or pd.DataFrame
        每个滚动窗口内的线性回归斜率。
        若窗口内存在 NaN，则返回 NaN。
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    t = np.arange(window, dtype=float)
    t_demean = t - t.mean()
    denominator = float(np.sum(t_demean ** 2))
    weighted_sum = sum(
        x.shift(lag) * t_demean[window - lag - 1]
        for lag in range(window)
    )
    valid_count = x.notna().rolling(window=window).sum()
    return (weighted_sum / denominator).where(valid_count == window)


ts_mean = sma
ts_std = ts_std_dev
ts_corr = correlation
ts_cov = ts_covariance
