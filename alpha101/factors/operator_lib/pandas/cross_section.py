from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "rank",
    "scale",
    "winsorize_group",
    "winsorize_mad",
    "zscore",
    "winsorize",
    "neutralize",
]

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
