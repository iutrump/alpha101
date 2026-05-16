import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
import os
from talib import MA    
from talib import ATR
from talib import SMA
from typing import Callable
from alpha101.world_quant.combine_ops import *
os.environ["NUMPY_WARN_IF_NO_MEM_POLICY"] = "1"  # 较新版本
# 或更通用的方式：
os.environ["NPY_DISABLE_NUMA"] = "1"  # 不相关，忽略

# 关键：强制 NumPy 在写只读数组时抛出明确错误 + 调用栈
import numpy as np
np.seterr(all='raise')  # 可选，增强数值错误检测

# region Auxiliary functions
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
ts_mean = sma  # 别名，保持一致性

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
ts_std = ts_std_dev  # 别名，保持一致性
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
    return x.rolling(window).corr(y)

ts_corr = correlation  # 别名，保持一致性
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
    xy = x * y

    mean_x = x.rolling(window).mean()
    mean_y = y.rolling(window).mean()
    mean_xy = xy.rolling(window).mean()
    cov = mean_xy - mean_x * mean_y
    return cov


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

    x = df.astype(np.float64)

    # 有效值计数：模仿 rolling(window) 默认 min_periods=window 的行为
    valid_count = x.notna().rolling(window, min_periods=window).sum()

    # 0 的位置
    is_zero = x.eq(0)
    zero_count = is_zero.rolling(window, min_periods=window).sum()

    # 负数个数决定符号
    neg_count = x.lt(0).rolling(window, min_periods=window).sum()
    sign = pd.DataFrame(
        np.where((neg_count % 2) == 0, 1.0, -1.0),
        index=x.index,
        columns=x.columns,
    )

    # 对 abs(x) 取 log；0 和 NaN 不参与 log 求和
    abs_x = x.abs()
    log_abs = pd.DataFrame(
        np.where((abs_x > 0) & abs_x.notna(), np.log(abs_x), 0.0),
        index=x.index,
        columns=x.columns,
    )
    log_sum = log_abs.rolling(window, min_periods=window).sum()

    # 防止 exp 溢出/下溢
    finfo = np.finfo(np.float64)
    max_log = np.log(finfo.max)          # ~709.78
    min_log = np.log(finfo.tiny)         # ~-708.40

    clipped_log_sum = log_sum.clip(lower=min_log, upper=max_log)
    mag = np.exp(clipped_log_sum)

    out = sign * mag

    # 窗口里有 0 -> 结果直接为 0
    out = out.mask(zero_count > 0, 0.0)

    # 窗口内有效值不足 window -> NaN
    out = out.mask(valid_count < window)

    return out

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

def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    return df.rank(axis=1, pct=True)

def scale(df, scale=1, longscale=None, shortscale=None):
    """
    WorldQuant-style cross-sectional scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Wide format (date × stocks)
    scale : float
        Total absolute exposure
    longscale : float
        Sum of long side exposure
    shortscale : float
        Sum of short side exposure (absolute)

    Returns
    -------
    pd.DataFrame
    """

    result = df.copy()

    # 默认模式：整体归一
    if longscale is None and shortscale is None:

        abs_sum = result.abs().sum(axis=1)
        abs_sum = abs_sum.replace(0, np.nan)

        result = result.div(abs_sum, axis=0) * scale

        return result

    # 多空分开缩放模式
    else:

        longscale = 1 if longscale is None else longscale
        shortscale = 1 if shortscale is None else shortscale

        scaled = pd.DataFrame(index=df.index, columns=df.columns)

        for dt in df.index:

            row = df.loc[dt]

            longs = row[row > 0]
            shorts = row[row < 0]

            new_row = pd.Series(0, index=row.index)

            # 处理多头
            if len(longs) > 0:
                long_sum = longs.sum()
                if long_sum != 0:
                    new_row.loc[longs.index] = longs * (longscale / long_sum)

            # 处理空头
            if len(shorts) > 0:
                short_sum = shorts.abs().sum()
                if short_sum != 0:
                    new_row.loc[shorts.index] = shorts * (shortscale / short_sum)

            scaled.loc[dt] = new_row

        return scaled

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

        # +1 符合 WorldQuant 习惯
        result[i] = argmax + 1

    return pd.DataFrame(result, index=df.index, columns=df.columns)

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

        # +1 符合 WorldQuant 习惯
        result[i] = argmin + 1

    return pd.DataFrame(result, index=df.index, columns=df.columns)

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
def winsorize_group(df):
    # Clip each feature column to its 1st and 99th percentiles
    lower = df.quantile(0.01, axis=1)
    upper = df.quantile(0.99, axis=1)
    df = df.clip(lower, upper, axis=1)
    return df
def winsorize_mad(df, n=3.0):
    """
    针对宽表（行=时间，列=股票）进行截面 MAD 去极值
    """
    # 1. 计算每一行（每个时间点）的中位数
    median = df.median(axis=1)
    
    # 2. 计算与中位数的偏差 (利用 sub 自动对齐 index)
    diff = df.sub(median, axis=0) # 宽表对齐需要注意 axis
    
    # 3. 计算 MAD
    mad = diff.abs().median(axis=1)
    
    # 4. 计算上下限 (1.4826 是正态分布调整系数)
    # limit = n * 1.4826 * mad
    limit = mad * n * 1.4826
    
    upper = median + limit
    lower = median - limit
    
    # 5. 执行 Clip (Pandas 的 clip 方法支持 axis=0 自动广播 Series)
    # 这里我们要用 computed series 截断 dataframe
    # 必须先把 upper/lower 转换成与 df 形状匹配或者利用 transpose 技巧
    
    # 技巧：转置后 clip 再转置回来，或者使用 where
    out = df.copy()
    
    # 由于 pandas 的 clip 甚至比较操作对齐比较麻烦，通常这样写最稳健：
    mask_upper = df.gt(upper, axis=0)
    mask_lower = df.lt(lower, axis=0)
    
    # 将大于 upper 的值设为 upper
    out = out.mask(mask_upper, upper, axis=0)
    # 将小于 lower 的值设为 lower
    out = out.mask(mask_lower, lower, axis=0)
    
    return out

def zscore(df):
    """
    Cross-sectional z-score normalization for each row in a DataFrame.
    :param df: A pandas DataFrame where rows are time points and columns are different assets/features.
    :return: A pandas DataFrame with the same shape as input, with z-score normalized values.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)

    # Avoid division by zero
    std_replaced = std.replace(0, np.nan)

    zscored = df.sub(mean, axis=0).div(std_replaced, axis=0)
    zscored = zscored.fillna(0)  # Replace NaN resulting from zero std with 0

    return zscored

def winsorize(df: pd.DataFrame, std: float = 4) -> pd.DataFrame:
    """
    Vectorized cross-sectional winsorization (row-wise).
    
    For each row (date), cap values beyond mean ± std * std_dev.
    
    Parameters
    ----------
    df : pd.DataFrame
        Rows = dates, Columns = stocks.
    std : float, default=4
        Number of standard deviations for threshold.
        
    Returns
    -------
    pd.DataFrame
        Winsorized DataFrame with same shape/index/columns.
    """
    # 计算每行的均值和标准差（ddof=0 表示总体标准差）
    row_mean = df.mean(axis=1)          # shape: (n_rows,)
    row_std = df.std(axis=1, ddof=0)    # shape: (n_rows,)

    # 处理标准差为 NaN 或 0 的情况（避免无效边界）
    # 将 std=0 或 NaN 的行设为无穷大边界 → 实际上不 clip
    valid_std = row_std.copy()
    valid_std = valid_std.where((valid_std > 0) & valid_std.notna(), np.inf)

    # 计算上下界（广播到与 df 相同形状）
    lower_bound = (row_mean - std * valid_std).values[:, None]  # (n_rows, 1)
    upper_bound = (row_mean + std * valid_std).values[:, None]  # (n_rows, 1)

    # 使用 np.clip 进行向量化截断
    # df.values 是 (n_rows, n_cols)，broadcast 自动对齐
    winsorized_values = np.clip(df.values, lower_bound, upper_bound)

    # 重建 DataFrame（保留索引和列名）
    return pd.DataFrame(winsorized_values, index=df.index, columns=df.columns)


def process_factor_wide_format(df, delay_days=0, decay_period=0):
    # 1. MAD 去极值
    df_win = winsorize_mad(df, n=3.0)
    
    # 2. (可选) 行业/市值中性化
    # 如果是纯量价因子，这步可能不需要；如果是基本面因子，通常在这里做中性化
    
    # 3. Z-Score 标准化
    df_std = zscore(df_win)
    # df_std = zscore(df_win, window=120)
    # 延迟 N 天，防止未来函数
    df_std = df_std.shift(delay_days) 
    if decay_period > 1:
        ts_decay_lineared = ts_decay_linear(df_std, period=decay_period)
    else:
        ts_decay_lineared = df_std
    return ts_decay_lineared
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

def if_else(condition, true_val, false_val):
    """
    Vectorized if-else function for pandas DataFrames.
    :param condition: A boolean DataFrame indicating where to apply true_val or false_val.
    :param true_val: The value to use where condition is True (can be scalar or DataFrame).
    :param false_val: The value to use where condition is False (can be scalar or DataFrame).
    :return: A DataFrame with values from true_val or false_val based on the condition.
    """
    return pd.DataFrame(np.where(condition, true_val, false_val), index=condition.index, columns=condition.columns)
def trade_when(condition, alpha, exit=np.nan):
    """
    Generate trading signals based on a condition.
    :param condition: A boolean DataFrame indicating when to enter a trade.
    :param alpha: The value to assign when entering a trade (can be scalar or DataFrame).
    :param exit: The value to assign when exiting a trade (default is -1, can be scalar or DataFrame).
    :return: A DataFrame with trading signals based on the condition.
    """
    return if_else(condition, alpha, exit)

def make_ts_regression(rettype: int) -> Callable[[pd.DataFrame, pd.DataFrame, int, int], pd.DataFrame]:
    def ts_regression(y: pd.DataFrame, x: pd.DataFrame, d: int, lag: int = 0) -> pd.DataFrame:
        if not y.shape == x.shape:
            raise ValueError("y 和 x 必须 shape 一致")
        if not y.index.equals(x.index) or not y.columns.equals(x.columns):
            raise ValueError("y 和 x 的 index/columns 必须一致")

        if lag > 0:
            y = y.shift(lag)
            x = x.shift(lag)

        # 预计算
        sum_x = x.rolling(d, min_periods=d).sum()
        sum_y = y.rolling(d, min_periods=d).sum()
        sum_xy = (x * y).rolling(d, min_periods=d).sum()
        sum_x2 = (x * x).rolling(d, min_periods=d).sum()

        mean_x = sum_x / d
        mean_y = sum_y / d

        cov_xy = sum_xy - d * mean_x * mean_y
        var_x = sum_x2 - d * mean_x * mean_x

        beta = cov_xy / var_x
        beta[var_x == 0] = np.nan

        alpha = mean_y - beta * mean_x

        if rettype == 0:
            return beta
        elif rettype == 1:
            return alpha
        elif rettype == 3:
            # R²
            y_pred = alpha + beta * x
            ss_res = ((y - y_pred) ** 2).rolling(d, min_periods=d).sum()
            ss_tot = ((y - mean_y) ** 2).rolling(d, min_periods=d).sum()

            r2 = 1 - ss_res / ss_tot
            r2[ss_tot == 0] = np.nan
            return r2
        else:
            return beta

    return ts_regression

# ts_regression定义具体函数
ts_beta = make_ts_regression(rettype=0) # 斜率 beta
ts_alpha = make_ts_regression(rettype=1) # 截距 alpha
# history lecay
# ts_slope = ts_alpha # 这是错误的    
ts_resid  = make_ts_regression(rettype=2) # 残差
ts_r2 = make_ts_regression(rettype=3)  

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

def neutralize(
    y: pd.DataFrame,
    x: pd.DataFrame,
    add_intercept: bool = True,
    min_count: int = 3,
) -> pd.DataFrame:
    """
    快速横截面单变量中性化。

    对每个 date，在 columns 方向做：
        y = alpha + beta * x + residual

    y, x:
        index = date
        columns = assets

    return:
        residual DataFrame
    """
    if y.shape != x.shape:
        raise ValueError("y 和 x 必须 shape 一致")
    if not y.index.equals(x.index) or not y.columns.equals(x.columns):
        raise ValueError("y 和 x 的 index/columns 必须一致")

    yv = y.astype(float)
    xv = x.astype(float)

    valid = np.isfinite(yv) & np.isfinite(xv)

    yy = yv.where(valid)
    xx = xv.where(valid)

    n = valid.sum(axis=1)

    if add_intercept:
        mean_x = xx.mean(axis=1, skipna=True)
        mean_y = yy.mean(axis=1, skipna=True)

        x_c = xx.sub(mean_x, axis=0)
        y_c = yy.sub(mean_y, axis=0)

        cov_xy = (x_c * y_c).sum(axis=1, skipna=True)
        var_x = (x_c * x_c).sum(axis=1, skipna=True)

        beta = cov_xy / var_x
        beta = beta.where((var_x != 0) & (n >= min_count))

        fitted = xx.mul(beta, axis=0).add(mean_y - beta * mean_x, axis=0)
        resid = yy - fitted

    else:
        # 不带截距：y = beta * x + residual
        sum_xy = (xx * yy).sum(axis=1, skipna=True)
        sum_x2 = (xx * xx).sum(axis=1, skipna=True)

        beta = sum_xy / sum_x2
        beta = beta.where((sum_x2 != 0) & (n >= min_count))

        fitted = xx.mul(beta, axis=0)
        resid = yy - fitted

    return resid

def ts_fourier_centroid(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Rolling spectral centroid from Fourier transform of returns.
    Higher value => more high-frequency fluctuations.
    """
    def _centroid_single_col(x):
        # x is np.ndarray due to raw=True
        if np.isnan(x).any():
            return np.nan
        x_centered = x - np.nanmean(x)  # use nanmean to be safe
        fft_vals = np.fft.rfft(x_centered)[1:]  # skip DC component
        if len(fft_vals) == 0:
            return np.nan
        amplitudes = np.abs(fft_vals)
        freqs = np.arange(1, len(amplitudes) + 1)
        total_amp = amplitudes.sum()
        if total_amp == 0:
            return 0.0
        return np.sum(freqs * amplitudes) / total_amp

    return df.rolling(window=window).apply(_centroid_single_col, raw=True)

def ts_fourier_high_freq_ratio(df: pd.DataFrame, window: int = 7, high_freq_ratio: float = 0.5) -> pd.DataFrame:
    """
    Rolling ratio of high-frequency energy in returns.
    """
    def _high_freq_ratio_single_col(x):
        if np.isnan(x).any():
            return np.nan
        x_centered = x - np.nanmean(x)
        fft_vals = np.fft.rfft(x_centered)[1:]
        if len(fft_vals) == 0:
            return np.nan
        amplitudes = np.abs(fft_vals)
        total_energy = amplitudes.sum()
        if total_energy == 0:
            return 0.0
        n_high = max(1, int(len(amplitudes) * high_freq_ratio))
        high_energy = np.sum(np.sort(amplitudes)[-n_high:])
        return high_energy / total_energy

    return df.rolling(window=window).apply(_high_freq_ratio_single_col, raw=True)

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

    # 时间变量: 0, 1, 2, ..., window-1
    t = np.arange(window, dtype=float)

    # 去均值后用于计算 slope
    t_mean = t.mean()
    t_demean = t - t_mean
    denominator = np.sum(t_demean ** 2)

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan

        y_mean = y.mean()
        y_demean = y - y_mean

        return np.sum(t_demean * y_demean) / denominator

    return x.rolling(window=window).apply(_slope, raw=True)

# endregion
def get_alpha_good15(df):
    stock=Alphas(df)
    df['alpha077']=stock.alpha077()
    df['alpha088']=stock.alpha088()
    df['alpha101']=stock.alpha101()

    df['alpha001']=stock.alpha001() 
    df['alpha011']=stock.alpha011()
    df['alpha033']=stock.alpha033()
    df['alpha036']=stock.alpha036()
    df['alpha041']=stock.alpha041()
    df['alpha044']=stock.alpha044()
  
    return df
def get_alpha(df):
        stock=Alphas(df)
        df['alpha001']=stock.alpha001() 
        df['alpha002']=stock.alpha002()
        df['alpha003']=stock.alpha003()
        df['alpha004']=stock.alpha004()
        df['alpha005']=stock.alpha005()
        df['alpha006']=stock.alpha006()
        df['alpha007']=stock.alpha007()
        df['alpha008']=stock.alpha008()
        df['alpha009']=stock.alpha009()
        df['alpha010']=stock.alpha010()
        df['alpha011']=stock.alpha011()
        df['alpha012']=stock.alpha012()
        df['alpha013']=stock.alpha013()
        df['alpha014']=stock.alpha014()
        df['alpha015']=stock.alpha015()
        df['alpha016']=stock.alpha016()
        df['alpha017']=stock.alpha017()
        df['alpha018']=stock.alpha018()
        df['alpha019']=stock.alpha019()
        df['alpha020']=stock.alpha020()
        # df['alpha021']=stock.alpha021()
        df['alpha022']=stock.alpha022()
        # df['alpha023']=stock.alpha023()
        df['alpha024']=stock.alpha024()
        df['alpha025']=stock.alpha025()
        df['alpha026']=stock.alpha026()
        df['alpha027']=stock.alpha027()
        df['alpha028']=stock.alpha028()
        df['alpha029']=stock.alpha029()
        df['alpha030']=stock.alpha030()
        df['alpha031']=stock.alpha031()
        df['alpha032']=stock.alpha032()
        df['alpha033']=stock.alpha033()
        df['alpha034']=stock.alpha034()
        df['alpha035']=stock.alpha035()
        df['alpha036']=stock.alpha036()
        df['alpha037']=stock.alpha037()
        df['alpha038']=stock.alpha038()
        df['alpha039']=stock.alpha039()
        df['alpha040']=stock.alpha040()
        df['alpha041']=stock.alpha041()
        df['alpha042']=stock.alpha042()
        df['alpha043']=stock.alpha043()
        df['alpha044']=stock.alpha044()
        df['alpha045']=stock.alpha045()
        df['alpha046']=stock.alpha046()
        df['alpha047']=stock.alpha047()
        df['alpha049']=stock.alpha049()
        df['alpha050']=stock.alpha050()
        df['alpha051']=stock.alpha051()
        df['alpha052']=stock.alpha052()
        df['alpha053']=stock.alpha053()
        # df['alpha054']=stock.alpha054()
        df['alpha055']=stock.alpha055()
        # df['alpha057']=stock.alpha057()
        df['alpha060']=stock.alpha060()
        # df['alpha061']=stock.alpha061()
        df['alpha062']=stock.alpha062()
        df['alpha064']=stock.alpha064()
        df['alpha065']=stock.alpha065()
        df['alpha066']=stock.alpha066()
        df['alpha068']=stock.alpha068()
        # df['alpha071']=stock.alpha071()
        df['alpha072']=stock.alpha072()
        df['alpha073']=stock.alpha073()
        df['alpha074']=stock.alpha074()
        # df['alpha075']=stock.alpha075()
        df['alpha077']=stock.alpha077()
        df['alpha078']=stock.alpha078()
        df['alpha081']=stock.alpha081()
        # df['alpha083']=stock.alpha083()
        # df['alpha084']=stock.alpha084()
        df['alpha085']=stock.alpha085()
        df['alpha086']=stock.alpha086()
        df['alpha088']=stock.alpha088()
        df['alpha092']=stock.alpha092()
        df['alpha094']=stock.alpha094()
        # df['alpha095']=stock.alpha095()
        df['alpha096']=stock.alpha096()
        df['alpha098']=stock.alpha098()
        df['alpha099']=stock.alpha099()
        df['alpha101']=stock.alpha101()  
        return df

import concurrent.futures
from functools import partial

# 辅助函数：单独计算一个 alpha
def compute_single_alpha(alpha_name, alpha_func):
    """
    计算单个 alpha
    返回: (alpha_name, result_series)
    """
    try:
        result = alpha_func()
        return alpha_name, result
    except Exception as e:
        print(f"Error computing {alpha_name}: {e}")
        return alpha_name, None
from tqdm import tqdm
from joblib import Parallel, delayed

def compute_alpha_by_name(alpha_name, stock):
    func = getattr(stock, alpha_name)
    try:
        return alpha_name, func()
    except Exception as e:
        print(f"Error computing {alpha_name}: {e}")
        return alpha_name, None

def get_alpha_parallel(df, n_jobs=-1):
    """
    并行计算多个 Alpha 因子
    
    Parameters:
        df (pd.DataFrame): 输入数据，含 open/high/low/close/volume/vwap 等
        n_jobs (int): 并行进程数，-1 表示使用全部 CPU 核心
    
    Returns:
        pd.DataFrame: 原 df 加上所有 alpha 列
    """

    stock = Alphas(df)
    
    # 定义要计算的 alpha 列表（名称 -> 方法）
    alpha_tasks = {
        # 'alpha_test_returns': stock.alpha_test_returns,
        # 'alpha_test_open': stock.alpha_test_open,
        # 'alpha_test_high': stock.alpha_test_high,
        # 'alpha_test_low': stock.alpha_test_low,
        # 'alpha_test_close': stock.alpha_test_close,
        # 'alpha_test_volume': stock.alpha_test_volume,
        # 'alpha073': stock.alpha073,

        # 'alpha_vcm_10d': stock.alpha_vcm_10d,
        # 'alpha005': stock.alpha005,
        # "alpha_volume_price_correlation": stock.alpha_volume_price_correlation,
        # 'alpha041_momentum_price': stock.alpha041_momentum_price,
        'alpha020': stock.alpha020,
        # 'alpha_Uncertainty_Volume_Factor_10D': stock.alpha_Uncertainty_Volume_Factor_10D,
        # 'alpha035': stock.alpha035,
        # "alpha_momentum_confirmation": stock.alpha_momentum_confirmation,
        # 'alpha078': stock.alpha078,
        # 'alpha055': stock.alpha055,
        # 'alpha003_ts_10': stock.alpha003_ts_10,

        # 'alpha040': stock.alpha040,
        # 'alpha011_ts': stock.alpha011_ts,
        # 'alpha042': stock.alpha042,
        # 'alpha047': stock.alpha047,

        # 'alpha_cr12': stock.alpha_cr12,
        # 'alpha_vr': stock.alpha_vr,

        # 'alpha003': stock.alpha003,

        # need to test
        # 'alpha001': stock.alpha001,
        # 'alpha002': stock.alpha002,
        # 'alpha004': stock.alpha004,
        # 'alpha006': stock.alpha006,
        # 'alpha007': stock.alpha007,
        # 'alpha008': stock.alpha008,
        # 'alpha009': stock.alpha009,
        # 'alpha010': stock.alpha010,
        # 'alpha012': stock.alpha012,
        # 'alpha013': stock.alpha013,
        # 'alpha014': stock.alpha014,
        # 'alpha015': stock.alpha015,
        # 'alpha047': stock.alpha047,
        # de
        # 'alpha011': stock.alpha011,
        # 'alpha016': stock.alpha016,
        # 'alpha017': stock.alpha017,
        # 'alpha018': stock.alpha018,
        # 'alpha019': stock.alpha019,
        # need to test
        # 'alpha032': stock.alpha032,
        # 'alpha066': stock.alpha066,
        # 'alpha037': stock.alpha037,
        # 'alpha024': stock.alpha024,
        # 'alpha036': stock.alpha036,
        # 'alpha052': stock.alpha052,
        # 'alpha044': stock.alpha044,
        # 'alpha094': stock.alpha094,
        # 'alpha026': stock.alpha026,
        # 'alpha072': stock.alpha072,

        # 'alpha022': stock.alpha022,
        # 'alpha025': stock.alpha025,
        # 'alpha027': stock.alpha027,
        # 'alpha028': stock.alpha028,
        # 'alpha029': stock.alpha029,
        # 'alpha030': stock.alpha030,
        # 'alpha031': stock.alpha031,
        # 'alpha033': stock.alpha033,
        # 'alpha034': stock.alpha034,
        # 'alpha038': stock.alpha038,
        # 'alpha039': stock.alpha039,
        # 'alpha041': stock.alpha041,
        # 'alpha043': stock.alpha043,
        # 'alpha045': stock.alpha045,
        # 'alpha046': stock.alpha046,
        # # 'alpha049': stock.alpha049,
        # 'alpha050': stock.alpha050,
        # 'alpha051': stock.alpha051,
        # 'alpha053': stock.alpha053,
        # 'alpha060': stock.alpha060,
        # 'alpha062': stock.alpha062,
        # 'alpha064': stock.alpha064,
        # 'alpha065': stock.alpha065,
        # 'alpha068': stock.alpha068,
        # 'alpha074': stock.alpha074,
        # 'alpha077': stock.alpha077,
        # 'alpha081': stock.alpha081,
        # 'alpha085': stock.alpha085,
        # 'alpha086': stock.alpha086,
        # 'alpha088': stock.alpha088,
        # 'alpha092': stock.alpha092,
        # 'alpha096': stock.alpha096,
        # 'alpha098': stock.alpha098,
        # 'alpha099': stock.alpha099,
        # 'alpha101': stock.alpha101,

    }
    alpha_names = list(alpha_tasks.keys())

    # 使用多进程并行计算
    outputs = Parallel(n_jobs=min(16, len(alpha_names)))(
        delayed(compute_alpha_by_name)(name, stock)
        for name in tqdm(alpha_names, desc="Alphas")
    )
    df_factors = {}
    # 转为 dict（过滤失败的结果）
    for name, result in sorted(outputs, key=lambda x: x[0]):
        if any(col not in df['open'].columns for col in result.columns):
            raise ValueError(f"Alpha {name} columns {list(result.columns)} not in df columns {list(df.columns)}")
        if result is not None:
            df_factors[name] = result
    alpha_frames = []
    for name, factor_df in df_factors.items():
        # factor_df: index=date, columns=symbol
        factor_df.index.name = 'date'
        factor_df.columns.name = 'symbol'
        factor_df = process_factor_wide_format(factor_df)
        # 构造与 panel 一致的列结构：(alpha_name, symbol)
        factor_df.columns = pd.MultiIndex.from_product(
            [[name], factor_df.columns]
        )
        alpha_frames.append(factor_df)
    alpha_panel = pd.concat(alpha_frames, axis=1)

    # 与原 panel 拼接
    df = pd.concat([df, alpha_panel], axis=1)
    
    return df

def get_alpha_good15_test(df):
    stock=Alphas(df)
    

    df['alpha001']=stock.alpha001() 
    df['alpha011']=stock.alpha011()
    df['alpha033']=stock.alpha033()
    df['alpha036']=stock.alpha036()
    df['alpha041']=stock.alpha041()
    df['alpha044']=stock.alpha044()
    df['alpha077']=stock.alpha077()
    df['alpha088']=stock.alpha088()
    df['alpha101']=stock.alpha101()
# 添加 Alpha#41 的变体因子
def get_alpha_041_v(df):
    stock=Alphas(df)
    df['alpha041_v2_ratio'] = stock.alpha041_v2_ratio()
    df['alpha041_v3_abs'] = stock.alpha041_v3_abs()
    df['alpha041_v4_typical_price'] = stock.alpha041_v4_typical_price()
    df['alpha041_v6_ma5_diff'] = stock.alpha041_v6_ma5_diff()
    df['alpha041_v7_zscore'] = stock.alpha041_v7_zscore()    
    df['alpha041_v8_atr_sma_norm'] = stock.alpha041_v8_atr_sma_norm()      
    df['alpha041_v8_atr_norm'] = stock.alpha041_v8_atr_norm()      
    df['alpha041_v9_vol_interaction'] = stock.alpha041_v9_vol_interaction()
    df['alpha041_v10_sign'] = stock.alpha041_v10_sign()
    df['alpha041_v12_decay5'] = stock.alpha041_v12_decay5()
    df['alpha041_v14_pct_rank'] = stock.alpha041_v14_pct_rank()
    df['alpha041_v15_squared'] = stock.alpha041_v15_squared()
    return df

def get_alpha_011_v(df):
    stock = Alphas(df)
    df['alpha011'] = stock.alpha011()
    df['alpha011_v4'] = stock.alpha011_v4_use_adv_instead_of_delta_vol()
    df['alpha011_v5'] = stock.alpha011_v5_pct_rank_output()
    df['alpha011_v6'] = stock.alpha011_v6_adjust_window()
    return df
def get_alpha_033_v(df):
    stock = Alphas(df)
    df['alpha033'] = stock.alpha033()
    df['alpha033_v1'] = stock.alpha033_v1_only_negative_jump()
    df['alpha033_v2'] = stock.alpha033_v2_momentum_of_jump()
    df['alpha033_v3'] = stock.alpha033_v3_volatility_normalized()
    df['alpha033_v4'] = stock.alpha033_v4_reverse_signal()
    df['alpha033_v5'] = stock.alpha033_v5_pct_rank_output()
    df['alpha033_v6'] = stock.alpha033_v6_refined_reversal()
    
    return df
def get_alpha_077_v(df):
    stock=Alphas(df)
    df['alpha077'] = stock.alpha077()
    df['alpha077_v2'] = stock.alpha077_v2_use_pct_rank()
    df['alpha077_v3'] = stock.alpha077_v3_adjust_windows()

    return df
def get_alpha_088_v(df):
    stock=Alphas(df)
    df['alpha088']=stock.alpha088()
    df['alpha088_v1_price_combo']=stock.alpha088_v1_price_combo()
    df['alpha088_v2_window_tuning']=stock.alpha088_v2_window_tuning()
    df['alpha088_v3_use_max_instead_of_min']=stock.alpha088_v3_use_max_instead_of_min()
    df['alpha088_v4_replace_adv_with_turnover']=stock.alpha088_v4_replace_adv_with_turnover()
    df['alpha088_v5_add_volatility_weight']=stock.alpha088_v5_add_volatility_weight()
    df['alpha088_v6_mean_instead_of_min']=stock.alpha088_v6_mean_instead_of_min()
    return df
def get_alpha_enhanced(df):
    df = get_alpha_good15(df)
    df = get_alpha_011_v(df)
    df = get_alpha_033_v(df)
    df = get_alpha_041_v(df)
    df = get_alpha_077_v(df)
    return df
class Alphas(object):
    def __init__(self, wide_data):
        '''
        wide_data: DataFrame with multi-index columns (feature, symbol), index is date
        需要包含至少以下列：open, high, low, close, volume, vwap等，格式为 (feature, symbol)，如 ('open
        '''
        self._open = wide_data['open'] 
        self._high = wide_data['high'] 
        self._low = wide_data['low']   
        self._close = wide_data['close'] 
        self._volume = wide_data['volume'] 
        self._returns = wide_data['close'].pct_change()  # 前一天的收益率
        self._vwap = wide_data['vwap'] # 需要确认 vwap 的定义，使用次级数据进行计算，这里进行了简化
        self._cap = wide_data['cap'] 
        try:
            self._funding = wide_data['funding'] 
        except Exception as e:
            self._funding = None
        # 使用每个时点市值 Top15，按市值占比加权得到 market return
        top_n = 15
        cap_top = self._cap.where(
            self._cap.rank(axis=1, ascending=False, method='first') <= top_n
        )
        cap_sum = cap_top.sum(axis=1).replace(0, np.nan)
        cap_weight = cap_top.div(cap_sum, axis=0)
        market = (self._returns * cap_weight).sum(axis=1, min_count=1)
        # market = self._returns['BTC_USDT_USDT']
        self._market_return = pd.DataFrame(
            np.repeat(market.values[:, None], self._returns.shape[1], axis=1),
            index=self._returns.index,
            columns=self._returns.columns
        )
        # print(self.market_return)
        # print(self.returns)

    @property
    def open(self):
        return self._open
    @property
    def high(self):
        return self._high
    @property
    def low(self):
        return self._low
    @property
    def close(self):
        return self._close
    @property
    def volume(self):
        return self._volume
    @property
    def returns(self):
        return self._returns
    @property
    def vwap(self):
        return self._vwap
    @property
    def market_return(self):
        return self._market_return
    @property
    def cap(self):
        return self._cap
    @property
    def funding(self):
        return self._funding
    def alpha_test_open(self):
        return self.open
    def alpha_test_high(self):
        return self.high
    def alpha_test_low(self):
        return self.low
    def alpha_test_close(self):
        return self.close
    def alpha_test_volume(self):
        return log(self.volume)
    def alpha_test_returns(self):
        return self.returns
    
    # Alpha#1	 (rank(ts_arg_max(SignedPower(((returns < 0) ? ts_std_dev(returns, 20) : close), 2.), 5)) -0.5)
    def alpha001(self):
        # 使用可写副本并避免就地修改底层只读数组
        inner = self.close.copy()
        # 将 returns < 0 的位置替换为 ts_std_dev(self.returns, 20)
        inner = inner.where(self.returns >= 0, ts_std_dev(self.returns, 20))
        return rank(ts_arg_max(inner ** 2, 5))

    # Alpha#2	 (-1 * correlation(rank(ts_delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha002(self):
        df = -1 * correlation(rank(ts_delta(log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#3	 (-1 * correlation(rank(open), rank(volume), 10))
    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    def alpha003_ts(self, window=90):
        df = -1 * correlation(ts_rank(self.open, window), ts_rank(self.volume, window), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    def alpha003_ts_10(self, window=10):
        df = -1 * correlation(ts_rank(self.open, window), ts_rank(self.volume, window), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#4	 (-1 * Ts_Rank(rank(low), 9))
    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)
    
    # Alpha#5	 (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha005(self):
        return  (rank(self.open - (self.vwap.rolling(window=10).mean())) * (-1 * abs(rank((self.close - self.vwap)))))
    def alpha005_ts(self):
        return  -(ts_rank(self.open - (self.vwap.rolling(window=10).mean())) * (-1 * abs(ts_rank((self.close - self.vwap)))))
    
    # Alpha#6	 (-1 * correlation(open, volume, 10))
    def alpha006(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#7	 ((adv20 < volume) ? ((-1 * ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7))) : (-1* 1))
    def alpha007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(abs(ts_delta(self.close, 7)), 60) * sign(ts_delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha
    
    # Alpha#8	 (-1 * rank(((sum(open, 5) * sum(returns, 5)) - ts_delay((sum(open, 5) * sum(returns, 5)),10))))
    def alpha008(self):
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           ts_delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))
    
    # Alpha#9	 ((0 < ts_min(ts_delta(close, 1), 5)) ? ts_delta(close, 1) : ((ts_max(ts_delta(close, 1), 5) < 0) ?ts_delta(close, 1) : (-1 * ts_delta(close, 1))))
    def alpha009(self):
        delta_close = ts_delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha
    
    # Alpha#10	 rank(((0 < ts_min(ts_delta(close, 1), 4)) ? ts_delta(close, 1) : ((ts_max(ts_delta(close, 1), 4) < 0)? ts_delta(close, 1) : (-1 * ts_delta(close, 1)))))
    def alpha010(self):
        delta_close = ts_delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha
    
    # Alpha#11	 ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(ts_delta(volume, 3)))
    # def alpha011(self):
    #     return -((rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))) *rank(ts_delta(self.volume, 3)))
    def alpha011_ts(self):
        return -((ts_rank(ts_max((self.vwap - self.close), 3)) + ts_rank(ts_min((self.vwap - self.close), 3))) *ts_rank(ts_delta(self.volume, 3)))
    
    def alpha011_v1_abs_deviation(self):
        """变体1：使用 |vwap - close| 的绝对偏离，避免 max/min 抵消"""
        dev = (self.vwap - self.close).abs()
        rank_dev = rank(ts_max(dev, 3))  # 过去3天最大绝对偏离
        rank_vol = rank(ts_delta(self.volume, 3))
        return rank_dev * rank_vol


    def alpha011_v2_only_below_vwap(self):
        """变体2：只关注收盘 < VWAP 的情况（做多信号：超卖反弹）"""
        # 当 close < vwap 时，dev 为正；否则设为0
        dev = np.maximum(self.vwap - self.close, 0)
        rank_dev = rank(ts_max(dev, 3))
        rank_vol = rank(ts_delta(self.volume, 3))
        return rank_dev * rank_vol


    def alpha011_v3_volatility_normalized(self):
        """变体3：对偏离做波动率标准化（除以 ATR）"""
        dev = self.vwap - self.close
        # 计算 ATR（简化版）
        tr = self.high - self.low
        atr = sma(tr, 10)
        normalized_dev = dev / (atr + 1e-8)
        
        rank_dev = rank(ts_max(normalized_dev, 3)) + rank(ts_min(normalized_dev, 3))
        rank_vol = rank(ts_delta(self.volume, 3))
        return rank_dev * rank_vol


    def alpha011_v4_use_adv_instead_of_delta_vol(self):
        """变体4：用 ADV（均量）变化替代 volume delta，更平滑"""
        adv10 = sma(self.volume, 10)
        vol_change = ts_delta(adv10, 3)  # 均量的变化
        
        rank_dev = rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))
        rank_vol = rank(vol_change)
        return rank_dev * rank_vol


    def alpha011_v5_pct_rank_output(self):
        """变体5：最终输出做横截面百分位（提升稳定性）"""
        raw = (rank(ts_max((self.vwap - self.close), 3)) + 
            rank(ts_min((self.vwap - self.close), 3))) * rank(ts_delta(self.volume, 3))
        return rank(raw)


    def alpha011_v6_adjust_window(self):
        """变体6：调整窗口（5日更敏感）"""
        rank_dev = rank(ts_max((self.vwap - self.close), 5)) + rank(ts_min((self.vwap - self.close), 5))
        rank_vol = rank(ts_delta(self.volume, 5))
        return rank_dev * rank_vol
    # Alpha#12	 (sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1)))
    def alpha012(self):
        return sign(ts_delta(self.volume, 1)) * (-1 * ts_delta(self.close, 1))

    # Alpha#13	 (-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha013(self):
        return -1 * rank(ts_covariance(rank(self.close), rank(self.volume), 5))
    
    # Alpha#14	 ((-1 * rank(ts_delta(returns, 3))) * correlation(open, volume, 10))
    def alpha014(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * rank(ts_delta(self.returns, 3)) * df
    
    # Alpha#15	 (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)
    
    # Alpha#16	 (-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha016(self):
        return -1 * rank(ts_covariance(rank(self.high), rank(self.volume), 5))
    
    # Alpha#17	 (((-1 * rank(ts_rank(close, 10))) * rank(ts_delta(ts_delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    def alpha017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(ts_delta(ts_delta(self.close, 1), 1)) *
                     rank(ts_rank((self.volume / adv20), 5)))
        
    # Alpha#18	 (-1 * rank(((ts_std_dev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    def alpha018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank((ts_std_dev(abs((self.close - self.open)), 5) + (self.close - self.open)) +
                          df))
    
    # Alpha#19	 ((-1 * sign(((close - ts_delay(close, 7)) + ts_delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    def alpha019(self):
        return ((-1 * sign((self.close - ts_delay(self.close, 7)) + ts_delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250))))
    
    # Alpha#20	 (((-1 * rank((open - ts_delay(high, 1)))) * rank((open - ts_delay(close, 1)))) * rank((open -ts_delay(low, 1))))
    def alpha020(self):
        return -1 * (rank(self.open - ts_delay(self.high, 1)) *
                     rank(self.open - ts_delay(self.close, 1)) *
                     rank(self.open - ts_delay(self.low, 1)))

    # Alpha#21	 ((((sum(close, 8) / 8) + ts_std_dev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - ts_std_dev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha021(self):
        cond_1 = sma(self.close, 8) + ts_std_dev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index
                             )
#        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index,
#                             columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha
    
    # Alpha#22	 (-1 * (ts_delta(correlation(high, volume, 5), 5) * rank(ts_std_dev(close, 20))))
    def alpha022(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_delta(df, 5) * rank(ts_std_dev(self.close, 20))

    # Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * ts_delta(high, 2)) : 0)
    def alpha023(self):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close),index=self.close.index,columns=['close'])
        alpha.at[cond,'close'] = -1 * ts_delta(self.high, 2).fillna(value=0)
        return alpha
    
    # Alpha#24	 ((((ts_delta((sum(close, 100) / 100), 100) / ts_delay(close, 100)) < 0.05) ||((ts_delta((sum(close, 100) / 100), 100) / ts_delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * ts_delta(close, 3)))
    def alpha024(self):
        cond = ts_delta(sma(self.close, 100), 100) / ts_delay(self.close, 100) <= 0.05
        alpha = -1 * ts_delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha
    
    # Alpha#25	 rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    def alpha025(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))
    
    # Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)
    
    # Alpha#27	 ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    ###
    def alpha027(self):
        """
        Alpha027: rank(sma(correlation(rank(volume), rank(close), 6), 2)) 
                - rank(sma(correlation(rank(volume), rank(vwap), 6), 2))
        """
        # 计算volume与close的6日相关性，然后2日平均
        corr_vc = correlation(rank(self.volume), rank(self.close), window=6)
        sma_corr_vc = sma(corr_vc, window=2)
        rank_vc = rank(sma_corr_vc)
        
        # 计算volume与vwap的6日相关性，然后2日平均
        corr_vv = correlation(rank(self.volume), rank(self.vwap), window=6)
        sma_corr_vv = sma(corr_vv, window=2)
        rank_vv = rank(sma_corr_vv)
        
        # 两者的差
        alpha = rank_vc - rank_vv
        return alpha
    
    # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    # Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(ts_delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(ts_delay((-1 * returns), 6), 5))
    def alpha029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(ts_delta((self.close - 1), 5)))), 2))))), 5) +
                ts_rank(ts_delay((-1 * self.returns), 6), 5))

    # Alpha#30	 (((1.0 - rank(((sign((close - ts_delay(close, 1))) + sign((ts_delay(close, 1) - ts_delay(close, 2)))) +sign((ts_delay(close, 2) - ts_delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha030(self):
        delta_close = ts_delta(self.close, 1)
        inner = sign(delta_close) + sign(ts_delay(delta_close, 1)) + sign(ts_delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    # Alpha#31	 ((rank(rank(rank(ts_decay_linear((-1 * rank(rank(ts_delta(close, 10)))), 10)))) + rank((-1 *ts_delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha031(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(value=0)         
        p1=rank(rank(rank(ts_decay_linear((-1 * rank(rank(ts_delta(self.close, 10)))), 10)))) 
        p2=rank((-1 * ts_delta(self.close, 3)))
        p3=sign(scale(df))
        
        return p1+p2+p3

    # Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, ts_delay(close, 5),230))))
    def alpha032(self):
        return scale(((sma(self.close, 7) / 7) - self.close)) + (20 * scale(correlation(self.vwap, ts_delay(self.close, 5),230)))
    
    # Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
    def alpha033(self):
        return rank(-1 + (self.open / self.close))
    def alpha033_v1_only_negative_jump(self):
        """变体1：只关注「低开高走」（强势信号），其余设为0"""
        jump = (self.open / self.close) - 1  # >0: 高开低走；<0: 低开高走
        strong_signal = np.where(jump < 0, -jump, 0)  # 取绝对值作为强度
        return rank(strong_signal)


    def alpha033_v2_momentum_of_jump(self):
        """变体2：过去N天「开盘跳空」的均值（捕捉持续性）"""
        jump = (self.open / self.close) - 1
        avg_jump = sma(jump, 5)  # 5日平均跳空
        return rank(avg_jump)


    def alpha033_v3_volatility_normalized(self):
        """变体3：跳空幅度除以ATR（标准化）"""
        jump = (self.open - self.close) / self.close
        # 计算ATR（14日）
        tr = self.high - self.low
        atr = sma(tr, 14)
        normalized_jump = jump / (atr + 1e-8)
        return rank(normalized_jump)


    def alpha033_v4_reverse_signal(self):
        """变体4：直接取反——聚焦「高开低走」的反转信号（最符合直觉）"""
        jump = (self.open / self.close) - 1
        return rank(-jump)  # 高开低走 → jump>0 → -jump<0 → rank低？不！
        # 实际：rank(-jump) 高 = jump 小（即低开高走）→ 这不是我们想要的
        # 更清晰写法：
        # return rank(self.close / self.open)  # close/open 越大，越强势


    def alpha033_v5_pct_rank_output(self):
        """变体5：输出为横截面百分位（提升稳定性）"""
        jump = (self.open / self.close) - 1
        return rank(jump)

    def alpha033_v6_refined_reversal(self):
        """
        改进版：聚焦「高开低走」的反转信号，并做波动率调整 + 衰减加权
        逻辑：高开低走越极端（且成交量放大），次日反转概率越高
        """
        # 1. 计算跳空幅度（高开低走为正）
        jump = np.maximum((self.open / self.close) - 1, 0)  # 只保留高开部分
        
        # 2. 波动率标准化（可选）
        tr = self.high - self.low
        atr = sma(tr, 10)
        jump_norm = jump / (atr + 1e-8)
        
        # 3. 结合成交量确认（高开低走 + 放量 = 更强信号）
        adv20 = sma(self.volume, 20)
        vol_ratio = self.volume / (adv20 + 1e-8)
        
        # 4. 合成信号：跳空 × 成交量
        signal = jump_norm * vol_ratio
        
        # 5. 时间衰减（强调最近信号）
        signal_df = ts_decay_linear(signal, period=3)
        
        # 6. 横截面百分位
        return rank(signal_df)
    # Alpha#34	 rank(((1 - rank((ts_std_dev(returns, 2) / ts_std_dev(returns, 5)))) + (1 - rank(ts_delta(close, 1)))))
    def alpha034(self):
        inner = ts_std_dev(self.returns, 2) / ts_std_dev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return rank(2 - rank(inner) - rank(ts_delta(self.close, 1)))

    # Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha035(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))
            
    # Alpha#36	 (((((2.21 * rank(correlation((close - open), ts_delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(ts_delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    def alpha036(self):
        adv20 = sma(self.volume, 20)
        return (((((2.21 * rank(correlation((self.close - self.open), ts_delay(self.volume, 1), 15))) + (0.7 * rank((self.open- self.close)))) + (0.73 * rank(ts_rank(ts_delay((-1 * self.returns), 6), 5)))) + rank(abs(correlation(self.vwap,adv20, 6)))) + (0.6 * rank((((sma(self.close, 200) / 200) - self.open) * (self.close - self.open)))))
    
    # Alpha#37	 (rank(correlation(ts_delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha037(self):
        return rank(correlation(ts_delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)
    
    # Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)
    
    # Alpha#39	 ((-1 * rank((ts_delta(close, 7) * (1 - rank(ts_decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    def alpha039(self):
        adv20 = sma(self.volume, 20)
        return ((-1 * rank(ts_delta(self.close, 7) * (1 - rank(ts_decay_linear((self.volume / adv20), 9))))) *
                (1 + rank(sma(self.returns, 250))))
    
    # Alpha#40	 ((-1 * rank(ts_std_dev(high, 10))) * correlation(high, volume, 10))
    def alpha040(self):
        return -1 * rank(ts_std_dev(self.high, 10)) * correlation(self.high, self.volume, 10)

    # Alpha#41	 (((high * low)^0.5) - vwap)
    def alpha041(self):
        return pow((self.high * self.low),0.5) - self.vwap
    
    def alpha041_v1_arith_mean(self):
        return (self.high + self.low) / 2 - self.vwap
    
    # 变体 2: 比值形式（相对偏差）
    def alpha041_v2_ratio(self):
        geo = pow(self.high * self.low, 0.5)
        return (geo - self.vwap) / self.vwap

    # 变体 3: 绝对偏离（波动/反转信号）
    def alpha041_v3_abs(self):
        geo = pow(self.high * self.low, 0.5)
        return (geo - self.vwap).abs()

    # 变体 4: 用典型价 (Typical Price) 替代 VWAP（无量场景）
    def alpha041_v4_typical_price(self):
        typical = (self.high + self.low + self.close) / 3
        geo = pow(self.high * self.low, 0.5)
        return geo - typical

    # 变体 5: 用收盘价替代 VWAP（极简版）
    def alpha041_v5_close(self):
        geo = pow(self.high * self.low, 0.5)
        return geo - self.close

    # 变体 6: 多日移动平均差（5日）
    def alpha041_v6_ma5_diff(self):
        geo = pow(self.high * self.low, 0.5)
        ma_geo = sma(geo, window=5)
        ma_vwap = sma(self.vwap, window=5)
        return ma_geo - ma_vwap

    # 变体 7: 时间序列 Z-Score（20日）
    def alpha041_v7_zscore(self):
        geo = pow(self.high * self.low, 0.5)
        diff = geo - self.vwap
        mean_ = sma(diff, window=20)
        std_ = ts_std_dev(diff, window=20)
        return (diff - mean_) / std_

    # 变体 8: 与 ATR 标准化（需先计算 ATR，这里用近似）
    def alpha041_v8_atr_sma_norm(self):
        # 近似 ATR：用 high-low 代替 TR
        result_atr = atr_sma(self)
        geo = pow(self.high * self.low, 0.5)
        return (geo - self.vwap) / result_atr
    
    def alpha041_v8_atr_norm(self):
        # 近似 ATR：用 high-low 代替 TR
        result_atr = atr(self)
        geo = pow(self.high * self.low, 0.5)
        return (geo - self.vwap) / result_atr

    # 变体 9: 与成交量交互（标准化后相乘）
    def alpha041_v9_vol_interaction(self):
        geo = pow(self.high * self.low, 0.5)
        base = geo - self.vwap
        vol_z = (self.volume - sma(self.volume, 10)) / ts_std_dev(self.volume, 10)
        return base * vol_z

    # 变体 10: 符号方向信号（+1 / -1）
    def alpha041_v10_sign(self):
        geo = pow(self.high * self.low, 0.5)
        return np.sign(geo - self.vwap)

    # 变体 11: 横截面排名（用于多因子合成）
    def alpha041_v11_rank(self):
        geo = pow(self.high * self.low, 0.5)
        raw = geo - self.vwap
        return rank(raw)

    # 变体 12: 线性衰减加权（过去5日 ts_decay_linear）
    def alpha041_v12_decay5(self):
        geo = pow(self.high * self.low, 0.5)
        diff = geo - self.vwap
        # 注意：ts_decay_linear 返回 DataFrame，取第一列或调整
        result = ts_decay_linear(diff, period=5)
        return result.iloc[:, 0]  # 转回 Series

    # 变体 13: 与5日动量结合
    def alpha041_v13_momentum(self):
        geo = pow(self.high * self.low, 0.5)
        base = geo - self.vwap
        mom = self.close / ts_delay(self.close, 5) - 1
        return base * mom

    # 变体 14: 分位数归一化（横截面百分位）
    def alpha041_v14_pct_rank(self):
        geo = pow(self.high * self.low, 0.5)
        raw = geo - self.vwap
        return raw.rank(pct=True)

    # 变体 15: 平方差（强调大偏离）
    def alpha041_v15_squared(self):
        geo = pow(self.high * self.low, 0.5)
        return (geo - self.vwap) ** 2
    # Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
    def alpha042(self):
        return rank((self.vwap - self.close)) / rank((self.vwap + self.close))
        
    # Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * ts_delta(close, 7)), 8))
    def alpha043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * ts_delta(self.close, 7)), 8)

    # Alpha#44	 (-1 * correlation(high, rank(volume), 5))
    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    # Alpha#45	 (-1 * ((rank((sum(ts_delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
    def alpha045(self):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank(sma(ts_delay(self.close, 5), 20)) * df *
                     rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))
    
    # Alpha#46	 ((0.25 < (((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - ts_delay(close, 1)))))
    def alpha046(self):
        inner = ((ts_delay(self.close, 20) - ts_delay(self.close, 10)) / 10) - ((ts_delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * ts_delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    # Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - ts_delay(vwap, 5))))
    def alpha047(self):
        adv20 = sma(self.volume, 20)
        return ((((rank((1 / self.close)) * self.volume) / adv20) * ((self.high * rank((self.high - self.close))) / (sma(self.high, 5) /5))) - rank((self.vwap - ts_delay(self.vwap, 5))))
    
    # Alpha#48	 (indneutralize(((correlation(ts_delta(close, 1), ts_delta(ts_delay(close, 1), 1), 250) *ts_delta(close, 1)) / close), IndClass.subindustry) / sum(((ts_delta(close, 1) / ts_delay(close, 1))^2), 250))
     
    
    # Alpha#49	 (((((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - ts_delay(close, 1))))
    def alpha049(self):
        inner = (((ts_delay(self.close, 20) - ts_delay(self.close, 10)) / 10) - ((ts_delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * ts_delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha
    
    # Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    def alpha050(self):
        return (-1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))
    
    # Alpha#51	 (((((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - ts_delay(close, 1))))
    def alpha051(self):
        inner = (((ts_delay(self.close, 20) - ts_delay(self.close, 10)) / 10) - ((ts_delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * ts_delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha
    
    # Alpha#52	 ((((-1 * ts_min(low, 5)) + ts_delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    def alpha052(self):
        return (((-1 * ts_delta(ts_min(self.low, 5), 5)) *
                 rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume, 5))
        
    # Alpha#53	 (-1 * ts_delta((((close - low) - (high - close)) / (close - low)), 9))
    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * ts_delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    # Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / (divisor)
        df = correlation(rank(inner), rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    # This Alpha uses the Cap, however I have not acquired the data yet
#    def alpha056(self):
#        return (0 - (1 * (rank((sma(self.returns, 10) / sma(sma(self.returns, 2), 3))) * rank((self.returns * self.cap)))))
    
    # Alpha#57	 (0 - (1 * ((close - vwap) / ts_decay_linear(rank(ts_arg_max(close, 30)), 2))))
    def alpha057(self):
        return (0 - (1 * ((self.close - self.vwap) / ts_decay_linear(rank(ts_arg_max(self.close, 30)), 2))))
    
    # Alpha#58	 (-1 * Ts_Rank(ts_decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))
     
    # Alpha#59	 (-1 * Ts_Rank(ts_decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
     
    
    # Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_arg_max(close, 10))))))
    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return - ((2 * scale(rank(inner))) - scale(rank(ts_arg_max(self.close, 10))))
    
	# Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    def alpha061(self):
        adv180 = sma(self.volume, 180)
        return (rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18)))
    
	# Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    def alpha062(self):
        adv20 = sma(self.volume, 20)
        return ((rank(correlation(self.vwap, sma(adv20, 22), 10)) < rank(((rank(self.open) +rank(self.open)) < (rank(((self.high + self.low) / 2)) + rank(self.high))))) * -1)
    
    # Alpha#63	 ((rank(ts_decay_linear(ts_delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(ts_decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)
     
    
    # Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(ts_delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    def alpha064(self):
        adv120 = sma(self.volume, 120)
        return ((rank(correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13),sma(adv120, 13), 17)) < rank(ts_delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 -0.178404))), 4))) * -1)
    
    # Alpha#65	 ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    def alpha065(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), sma(adv60,9), 6)) < rank((self.open - ts_min(self.open, 14)))) * -1)
      
    # Alpha#66	 ((rank(ts_decay_linear(ts_delta(vwap, 3.51013), 7.23052)) + Ts_Rank(ts_decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    def alpha066(self):
        return ((rank(ts_decay_linear(ts_delta(self.vwap, 4), 7)) + ts_rank(ts_decay_linear(((((self.low* 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)
    
    # Alpha#67	 ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
     
    
    # Alpha#68	 ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(ts_delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    def alpha068(self):
        adv15 = sma(self.volume, 15)
        return ((ts_rank(correlation(rank(self.high), rank(adv15), 9), 14) <rank(ts_delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1))) * -1)
    
    # Alpha#69	 ((rank(ts_max(ts_delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)
         
    # Alpha#70	 ((rank(ts_delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
     
    
    # Alpha#71	 max(Ts_Rank(ts_decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(ts_decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
    def alpha071(self):
        adv180 = sma(self.volume, 180)
        p1=ts_rank(ts_decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18), 4), 16)
        p2=ts_rank(ts_decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)), 16), 4)
        df=pd.DataFrame({'p1':p1,'p2':p2})
        df.at[df['p1']>=df['p2'],'max']=df['p1']
        df.at[df['p2']>=df['p1'],'max']=df['p2']
        return df['max']
        #return max(ts_rank(ts_decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18), 4), 16), ts_rank(ts_decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)), 16), 4))
    
    # Alpha#72	 (rank(ts_decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(ts_decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
    def alpha072(self):
        adv40 = sma(self.volume, 40)
        return (rank(ts_decay_linear(correlation(((self.high + self.low) / 2), adv40, 9), 10)) /rank(ts_decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7),3)))
    
    # Alpha#73	 (max(rank(ts_decay_linear(ts_delta(vwap, 4.72775), 2.91864)),Ts_Rank(ts_decay_linear(((ts_delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    def alpha073(self):
        p1 = rank(ts_decay_linear(ts_delta(self.vwap, 5), 3))
        p2 = ts_rank(ts_decay_linear(((ts_delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / 
                                    ((self.open * 0.147155) + (self.low * (1 - 0.147155)))) * -1), 3), 17)
        
        max_values = np.maximum(p1, p2)
        return -1 * max_values
        #return (max(rank(ts_decay_linear(ts_delta(self.vwap, 5), 3)),ts_rank(ts_decay_linear(((ts_delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open *0.147155) + (self.low * (1 - 0.147155)))) * -1), 3), 17)) * -1)
    
    # Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
    def alpha074(self):
        adv30 = sma(self.volume, 30)
        return ((rank(correlation(self.close, sma(adv30, 37), 15)) <rank(correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11)))* -1)
    
    # Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
    def alpha075(self):
        adv50 = sma(self.volume, 50)
        return (rank(correlation(self.vwap, self.volume, 4)) < rank(correlation(rank(self.low), rank(adv50),12)))
    
    # Alpha#76	 (max(rank(ts_decay_linear(ts_delta(vwap, 1.24383), 11.8259)),Ts_Rank(ts_decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)
     

    # Alpha#77	 min(rank(ts_decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(ts_decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    def alpha077(self):
        """变体1：简化 p1，直接用 mid - vwap（去除冗余 high）
        min(
            rank(ts_decay_linear(((high + low) / 2) - vwap, 20)),
            rank(ts_decay_linear(correlation(((high + low) / 2), sma(volume, 40), 3), 6))
        )   
        """
        mid = (self.high + self.low) / 2
        p1 = rank(ts_decay_linear((mid - self.vwap), 20))
        
        adv40 = sma(self.volume, 40)
        p2 = rank(ts_decay_linear(correlation(mid, adv40, 3), 6))
        
        return np.minimum(p1, p2)


    def alpha077_v2_use_pct_rank(self):
        """变体2：将最终结果改为横截面百分位（提升跨股票可比性）"""
        mid = (self.high + self.low) / 2
        p1 = rank(ts_decay_linear((mid - self.vwap), 20))
        
        adv40 = sma(self.volume, 40)
        p2 = rank(ts_decay_linear(correlation(mid, adv40, 3), 6))
        
        min_val = np.minimum(p1, p2)
        # 横截面百分位排名（更稳定）
        return rank(min_val)


    def alpha077_v3_adjust_windows(self):
        """变体3：调整时间窗口（短周期更敏感）"""
        mid = (self.high + self.low) / 2
        p1 = rank(ts_decay_linear((mid - self.vwap), 10))   # 20→10
        
        adv20 = sma(self.volume, 20)  # 40→20
        p2 = rank(ts_decay_linear(correlation(mid, adv20, 2), 4))  # 3→2, 6→4
        
        return np.minimum(p1, p2)


    # Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    def alpha078(self):
        adv40 = sma(self.volume, 40)
        return (rank(correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20),ts_sum(adv40,20), 7)).pow(rank(correlation(rank(self.vwap), rank(self.volume), 6))))
    
    # Alpha#79	 (rank(ts_delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
     
    # Alpha#80	 ((rank(Sign(ts_delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
     
   
    # Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    def alpha081(self):
        adv10 = sma(self.volume, 10)
        return ((rank(log(ts_product(rank((rank(correlation(self.vwap, ts_sum(adv10, 50),8)).pow(4))), 15))) < rank(correlation(rank(self.vwap), rank(self.volume), 5))) * -1)
    
    # Alpha#82	 (min(rank(ts_decay_linear(ts_delta(open, 1.46063), 14.8717)),Ts_Rank(ts_decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
     
    
    # Alpha#83	 ((rank(ts_delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
    def alpha083(self):
        return ((rank(ts_delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) / (((self.high -self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))
    
    # Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), ts_delta(close,4.96796))
    def alpha084(self):
        return pow(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), ts_delta(self.close,5))
    
    # Alpha#85	 (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
    def alpha085(self):
        adv30 = sma(self.volume, 30)
        return (rank(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30,10)).pow(rank(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10),7))))
    
    # Alpha#86	 ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)

    def alpha086(self, binary_output=True):
        """
        Alpha086更精确的实现
        
        Args:
            binary_output: 是否输出二值信号，False时输出连续值
        """
        adv20 = sma(self.volume, 20)
        
        # 计算时间序列rank
        corr_series = correlation(self.close, sma(adv20, 15), window=6)
        ts_rank_val = ts_rank(corr_series, window=20)
        
        # 计算横截面rank（虽然公式里有open，但实际简化为close-vwap）
        cross_rank_val = rank(self.close - self.vwap)
        
        # 计算差异
        diff = ts_rank_val - cross_rank_val
        
        if binary_output:
            # 二值化：diff < 0 时返回-1，否则返回1
            alpha = diff.copy()
            alpha[:] = 1
            return alpha.mask(diff < 0, -1)
        else:
            # 返回连续值
            return -diff  # 负号使得当ts_rank < cross_rank时为正
    # Alpha#87	 (max(rank(ts_decay_linear(ts_delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(ts_decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
     
    
    # Alpha#88	 min(rank(ts_decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(ts_decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
    def alpha088(self):
        adv60 = sma(self.volume, 60)
        
        # 计算p1和p2
        p1 = rank(ts_decay_linear(((rank(self.open) + rank(self.low)) - 
                            (rank(self.high) + rank(self.close))), 8))
        p2 = ts_rank(ts_decay_linear(correlation(ts_rank(self.close, 8), 
                                            ts_rank(adv60, 21), 8), 7), 3)
        
        # 使用np.minimum获取两者最小值 - 最简洁高效
        min_values = np.minimum(p1, p2)
        
        # 保持索引一致性
        return min_values
        #return min(rank(ts_decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))),8)), ts_rank(ts_decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60,20.6966), 8), 7), 3))
    
    def alpha088_v1_price_combo(self):
        """变体1：修改 p1 中的价格组合逻辑（使用几何中心 vs 收盘）"""
        adv60 = sma(self.volume, 60)
        
        # 新 p1: (sqrt(high*low) - close) 的 rank
        geo = (self.high * self.low) ** 0.5
        p1_raw = geo - self.close
        p1 = rank(ts_decay_linear(rank(p1_raw), 8))
        
        # p2 保持不变
        p2 = ts_rank(
            ts_decay_linear(
                correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8), 
                7
            ), 
            3
        )
        
        return np.minimum(p1, p2)


    def alpha088_v2_window_tuning(self):
        """变体2：调整所有时间窗口（短周期更敏感）"""
        adv60 = sma(self.volume, 30)  # 更短期流动性
        
        p1 = rank(ts_decay_linear(
            ((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 
            5  # 原8 → 5
        ))
        
        p2 = ts_rank(
            ts_decay_linear(
                correlation(ts_rank(self.close, 5), ts_rank(adv60, 10), window=5), 
                5  # 衰减窗口也缩短
            ), 
            2  # ts_rank 窗口变小
        )
        
        return np.minimum(p1, p2)


    def alpha088_v3_use_max_instead_of_min(self):
        """变体3：取 max 而非 min（激进信号）"""
        adv60 = sma(self.volume, 60)
        
        p1 = rank(ts_decay_linear(((rank(self.open) + rank(self.low)) - 
                                (rank(self.high) + rank(self.close))), 8))
        p2 = ts_rank(ts_decay_linear(correlation(ts_rank(self.close, 8), 
                                            ts_rank(adv60, 21), 8), 7), 3)
        
        return np.maximum(p1, p2)


    def alpha088_v4_replace_adv_with_turnover(self):
        """变体4：用换手率替代成交量（如果数据中有流通股本）"""
        # 假设 self.turnover = self.volume / self.shares_float （若无 shares_float，可用 volume / close 近似）
        # 这里用 volume / close 作为流动性代理（金额）
        turnover_proxy = self.volume / self.close
        adv60 = sma(turnover_proxy, 60)
        
        p1 = rank(ts_decay_linear(((rank(self.open) + rank(self.low)) - 
                                (rank(self.high) + rank(self.close))), 8))
        p2 = ts_rank(ts_decay_linear(correlation(ts_rank(self.close, 8), 
                                            ts_rank(adv60, 21), 8), 7), 3)
        
        return np.minimum(p1, p2)


    def alpha088_v5_add_volatility_weight(self):
        """变体5：用 ATR 对 p1/p2 加权，高波动时降低信号强度"""
        # 先计算 ATR（简化版）
        tr = self.high - self.low
        atr = sma(tr, 14)
        vol_norm = 1.0 / (1 + atr)  # 波动越大，权重越小
        
        adv60 = sma(self.volume, 60)
        p1_raw = rank(ts_decay_linear(((rank(self.open) + rank(self.low)) - 
                                    (rank(self.high) + rank(self.close))), 8))
        p2_raw = ts_rank(ts_decay_linear(correlation(ts_rank(self.close, 8), 
                                                ts_rank(adv60, 21), 8), 7), 3)
        
        p1 = p1_raw * vol_norm
        p2 = p2_raw * vol_norm
        
        return np.minimum(p1, p2)


    def alpha088_v6_mean_instead_of_min(self):
        """变体6：取平均值（平衡信号）"""
        adv60 = sma(self.volume, 60)
        
        p1 = rank(ts_decay_linear(((rank(self.open) + rank(self.low)) - 
                                (rank(self.high) + rank(self.close))), 8))
        p2 = ts_rank(ts_decay_linear(correlation(ts_rank(self.close, 8), 
                                            ts_rank(adv60, 21), 8), 7), 3)
        
        return (p1 + p2) / 2


    # Alpha#89	 (Ts_Rank(ts_decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(ts_decay_linear(ts_delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
     
    # Alpha#90	 ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
     
    # Alpha#91	 ((Ts_Rank(ts_decay_linear(ts_decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(ts_decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
     

    # Alpha#92	 min(Ts_Rank(ts_decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(ts_decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    def alpha092(self):
        adv30 = sma(self.volume, 30)
        p1=ts_rank(ts_decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)), 15),19)
        p2=ts_rank(ts_decay_linear(correlation(rank(self.low), rank(adv30), 8), 7),7)
        # df=pd.DataFrame({'p1':p1,'p2':p2})
        # df.at[df['p1']>=df['p2'],'min']=df['p2']
        # df.at[df['p2']>=df['p1'],'min']=df['p1']
        min_values = np.minimum(p1, p2)
        return min_values
        #return  min(ts_rank(ts_decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)), 15),19), ts_rank(ts_decay_linear(correlation(rank(self.low), rank(adv30), 8), 7),7))
    
    # Alpha#93	 (Ts_Rank(ts_decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(ts_decay_linear(ts_delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))
     
    
    # Alpha#94	 ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    def alpha094(self):
        adv60 = sma(self.volume, 60)
        return ((rank((self.vwap - ts_min(self.vwap, 12))).pow(ts_rank(correlation(ts_rank(self.vwap,20), ts_rank(adv60, 4), 18), 3)) * -1))
    
    # Alpha#95	 (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    def alpha095(self):
        adv40 = sma(self.volume, 40)
        return (rank((self.open - ts_min(self.open, 12))) < ts_rank((rank(correlation(sma(((self.high + self.low)/ 2), 19), sma(adv40, 19), 13)).pow(5)), 12))
    
    # Alpha#96	 (max(Ts_Rank(ts_decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(ts_decay_linear(ts_arg_max(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    def alpha096(self):
        adv60 = sma(self.volume, 60)
        p1=ts_rank(ts_decay_linear(correlation(rank(self.vwap), rank(self.volume), 4),4), 8)
        p2=ts_rank(ts_decay_linear(ts_arg_max(correlation(ts_rank(self.close, 7),ts_rank(adv60, 4), 4), 13), 14), 13)
        return -1*np.maximum(p1, p2)
        #return (max(ts_rank(ts_decay_linear(correlation(rank(self.vwap), rank(self.volume), 4),4), 8), ts_rank(ts_decay_linear(ts_arg_max(correlation(ts_rank(self.close, 7),ts_rank(adv60, 4), 4), 13), 14), 13)) * -1)
    
    # Alpha#97	 ((rank(ts_decay_linear(ts_delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(ts_decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
     
    
    # Alpha#98	 (rank(ts_decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(ts_decay_linear(Ts_Rank(ts_arg_min(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    def alpha098(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)
        return (rank(ts_decay_linear(correlation(self.vwap, sma(adv5, 26), 5), 7)) -rank(ts_decay_linear(ts_rank(ts_arg_min(correlation(rank(self.open), rank(adv15), 21), 9),7), 8)))
    
    # Alpha#99	 ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(low, volume, 6.28259))) * -1)
    def alpha099(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)) <rank(correlation(self.low, self.volume, 6))) * -1)
    
    # Alpha#100	 (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_arg_min(close, 30))),IndClass.subindustry))) * (volume / adv20))))
     

    # Alpha#101	 ((close - open) / ((high - low) + .001))
    def alpha101(self):
        return (self.close - self.open) /((self.high - self.low) + 0.001)
    def alpha_cr12(self):
        cr12 = pow(1 + self.returns, 12) - 1
        jackpot_s = log(10 + cr12)
        vol_z = zscore(sma(self.volume,6))
        hit = (cr12 > 1).astype(int)

        raw = sma(jackpot_s,7) * sma(hit,20) * (1 - rank(vol_z)) 
        return rank(raw)
    # def alpha_cr3(self):
    #     cr12 = pow(1 + self.close.pct_change(), 3) - 1
    #     jackpot_s = np.log(np.maximum(0.00001, cr12)+1)
    #     vol_z = zscore(sma(self.volume, 7), 7)
    #     hit = (cr12 > 0.6).astype(int)
    #     raw = sma(jackpot_s, 7) * sma(hit, 7) * (1 - rank(vol_z))
    #     return -rank(raw)
# ts_mean(
#     -((close - low) - (high - close)) / (0.001 + high - low) * group_rank(volume, sector),
#     2
# )

    def alpha041_momentum_price(self):
        '''
        ts_rank(-vwap, 3) * rank(sma(volume, 3) / sma(volume, 20))
        '''
        vol_fast = sma(self.volume, 3)
        vol_slow = sma(self.volume, 20)
        vol_confirm = rank(vol_fast / vol_slow)
        return ts_rank(-self.vwap,3) * vol_confirm
    
    def alpha_vr(self):
        vr_raw = -((self.close - self.low) - (self.high - self.close)) / (0.001 + self.high - self.low)
        vr_ranked = vr_raw * rank(self.volume)
        return sma(vr_ranked, 2)
    def alpha_vcm_10d(self):
        """
        VCM_10D因子: 量价结合因子
        公式: VCM_10D = (TS_MEAN(volume, 10) / TS_MEAN(ts_delay(volume, 1), 10)) * TS_SUM(return, 3)
        
        逻辑: 用成交量比率（当前量能vs历史量能）来调整短期动量强度
        """
        # 计算10日平均成交量
        volume_ma10 = sma(self.volume, 10)
        
        # 计算昨日成交量的10日平均 (ts_delay(volume, 1)的10日均线)
        # 首先获取昨日成交量，然后计算其10日均线
        volume_delay1 = ts_delay(self.volume, 1)
        volume_delay1_ma10 = sma(volume_delay1, 10)
        
        # 计算成交量比率，添加小常数避免除零
        volume_ratio = volume_ma10 / (volume_delay1_ma10 + 0.001)
        
        # 计算3日累计收益
        returns_3d_sum = ts_sum(self.returns, 3)
        
        # 因子值 = 成交量比率 * 3日累计收益
        factor_value = volume_ratio * returns_3d_sum
        
        return factor_value
    def alpha_volume_price_correlation(self):
        """
        成交量-价格相关性因子
        公式: TS_CORR((($volume - TS_MEAN($volume, 5)) / (TS_STD($volume, 5) + 1e-8)), TS_PCTCHANGE($close, 5), 5)
        
        逻辑: 计算标准化成交量与5日收益率的相关性
        - 第一部分: 成交量的Z-score标准化 (去均值除标准差)
        - 第二部分: 5日价格变化率
        - 计算两者的5日滚动相关性
        """
    # 1. 计算成交量的5日均值
        volume_ma5 = ts_mean(self.volume, 5)
        
        # 2. 计算成交量的5日标准差（添加小常数避免除零）
        volume_std5 = ts_std(self.volume, 5)
        volume_std5_safe = volume_std5 + 1e-8
        
        # 3. 成交量标准化 (相当于Z-score)
        # (当前成交量 - 5日均值) / 5日标准差
        volume_normalized = (self.volume - volume_ma5) / volume_std5_safe
        
        # 4. 计算5日价格变化率
        price_pct_change = ts_pctchange(self.close, 5)
        
        # 5. 计算两者的5日滚动相关性
        factor_value = ts_corr(volume_normalized, price_pct_change, 5)
        
        return factor_value
    def alpha_volume_spike_momentum_factor_10d(self):
        """
        成交量突增动量因子（10日）
        公式: (($volume - TS_MEAN($volume, 10)) / (TS_STD($volume, 10) + 1e-8))
        
        逻辑: 计算成交量的标准化值，用于捕捉成交量的异常波动
        """
        volume_ma10 = ts_mean(self.volume, 10)
        volume_std10 = ts_std(self.volume, 10)
        volume_std10_safe = volume_std10 + 1e-8
        factor_value = (self.volume - volume_ma10) / volume_std10_safe
        return factor_value
    def alpha_Uncertainty_Volume_Factor_10D(self):
        return -ts_std(self.volume, 10) / (ts_std(ts_pctchange(self.close, 1), 10) + 1e-8)
    def alpha_momentum_confirmation(self):
        return ts_mean(self.returns, 7) + ts_mean(self.returns, 14)
    # def run_fast(self, fast_expression: str):
    #     engine = FastExpressionEngine(self)
    #     return engine.evaluate(fast_expression)