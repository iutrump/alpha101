import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
import os
from typing import Callable
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
    Alpha101-style cross-sectional scaling.

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

        # +1 follows the original Alpha101 convention.
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

        # +1 follows the original Alpha101 convention.
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

    