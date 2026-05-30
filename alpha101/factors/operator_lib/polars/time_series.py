from __future__ import annotations

import numpy as np

from alpha101.factors.operator_lib.polars.utils import from_numpy_like, require_polars, rolling_apply_numpy, to_numpy

__all__ = [
    "correlation",
    "ts_alpha",
    "ts_arg_max",
    "ts_arg_min",
    "ts_beta",
    "ts_corr",
    "ts_covariance",
    "ts_decay_linear",
    "ts_delay",
    "ts_delta",
    "ts_drawdown",
    "ts_ema",
    "ts_kurtosis",
    "ts_max",
    "ts_mean",
    "ts_min",
    "ts_pos",
    "ts_product",
    "ts_r2",
    "ts_rank",
    "ts_resid",
    "ts_skewness",
    "ts_slope",
    "ts_std",
    "ts_std_dev",
    "ts_sum",
    "ts_zscore",
]


def _select_all(df, method: str, *args, **kwargs):
    pl = require_polars()
    return df.select([getattr(pl.col(col), method)(*args, **kwargs).alias(col) for col in df.columns])


def ts_sum(df, window=10):
    return _select_all(df, "rolling_sum", window_size=window, min_samples=window)


def ts_mean(df, window=10):
    return _select_all(df, "rolling_mean", window_size=window, min_samples=window)


def ts_min(df, window=10):
    return _select_all(df, "rolling_min", window_size=window, min_samples=window)


def ts_max(df, window=10):
    return _select_all(df, "rolling_max", window_size=window, min_samples=window)


def ts_std_dev(df, window=10):
    return _select_all(df, "rolling_std", window_size=window, min_samples=window)


def ts_delay(df, period=1):
    return df.select([require_polars().col(col).shift(period).alias(col) for col in df.columns])


def ts_delta(df, period=1):
    return df - ts_delay(df, period)


def ts_ema(df, window=10):
    return df.select([require_polars().col(col).ewm_mean(span=window, min_samples=window).alias(col) for col in df.columns])


def ts_skewness(df, window=10):
    def _skew(arr):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        centered = arr - mean
        return np.nanmean(centered ** 3, axis=0) / (std ** 3)

    return rolling_apply_numpy(df, window, _skew)


def ts_kurtosis(df, window=10):
    def _kurt(arr):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        centered = arr - mean
        return np.nanmean(centered ** 4, axis=0) / (std ** 4) - 3.0

    return rolling_apply_numpy(df, window, _kurt)


def ts_rank(df, window=10):
    return rolling_apply_numpy(df, window, lambda arr: (arr <= arr[-1]).sum(axis=0) / window)


def ts_product(df, window=10):
    return rolling_apply_numpy(df, window, lambda arr: np.prod(arr, axis=0))


def ts_arg_max(df, window=10):
    return rolling_apply_numpy(df, window, lambda arr: np.argmax(arr, axis=0) + 1)


def ts_arg_min(df, window=10):
    return rolling_apply_numpy(df, window, lambda arr: np.argmin(arr, axis=0) + 1)


def ts_decay_linear(df, period: int):
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    return rolling_apply_numpy(df, period, lambda arr: np.sum(arr * weights[:, None], axis=0))


def ts_zscore(df, window=10, eps=1e-8):
    mean = ts_mean(df, window)
    std = ts_std_dev(df, window)
    return (df - mean) / (std + eps)


def ts_pos(df, window=10):
    roll_min = ts_min(df, window)
    roll_max = ts_max(df, window)
    denom = roll_max - roll_min
    return (df - roll_min) / denom


def ts_drawdown(df, window=10):
    roll_max = ts_max(df, window)
    return df / roll_max - 1.0


def ts_slope(df, window=10):
    if window < 2:
        raise ValueError("window must be >= 2")
    t = np.arange(window, dtype=float)
    t = t - t.mean()
    denom = np.sum(t * t)
    return rolling_apply_numpy(df, window, lambda arr: np.sum(t[:, None] * (arr - arr.mean(axis=0)), axis=0) / denom)


def ts_covariance(x, y, window=10):
    return ts_mean(x * y, window) - ts_mean(x, window) * ts_mean(y, window)


def correlation(x, y, window=10):
    cov = ts_covariance(x, y, window)
    return cov / (ts_std_dev(x, window) * ts_std_dev(y, window))


def _ts_regression(y, x, window: int, rettype: int, lag: int = 0):
    if lag > 0:
        y = ts_delay(y, lag)
        x = ts_delay(x, lag)
    mean_x = ts_mean(x, window)
    mean_y = ts_mean(y, window)
    cov_xy = ts_covariance(x, y, window)
    var_x = ts_covariance(x, x, window)
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    if rettype == 0:
        return beta
    if rettype == 1:
        return alpha
    if rettype == 3:
        y_pred = alpha + beta * x
        ss_res = ts_sum((y - y_pred) * (y - y_pred), window)
        ss_tot = ts_sum((y - mean_y) * (y - mean_y), window)
        return 1.0 - ss_res / ss_tot
    return y - (alpha + beta * x)


def ts_beta(y, x, d, lag=0):
    return _ts_regression(y, x, d, 0, lag)


def ts_alpha(y, x, d, lag=0):
    return _ts_regression(y, x, d, 1, lag)


def ts_resid(y, x, d, lag=0):
    return _ts_regression(y, x, d, 2, lag)


def ts_r2(y, x, d, lag=0):
    return _ts_regression(y, x, d, 3, lag)


ts_std = ts_std_dev
ts_corr = correlation
