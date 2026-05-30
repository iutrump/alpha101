from __future__ import annotations

from alpha101.data.views import PolarsFactor
from alpha101.factors.operator_lib.polars.utils import ensure_factor, require_polars

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


def _rolling(factor: PolarsFactor, method: str, window: int):
    factor = ensure_factor(factor)
    value = require_polars().col("value")
    expr = getattr(value, method)(window_size=window, min_samples=window).over("symbol")
    return factor.map_value(expr)


def _rolling_map(factor: PolarsFactor, window: int, function):
    factor = ensure_factor(factor)
    expr = (
        require_polars()
        .col("value")
        .rolling_map(function, window_size=window, min_samples=window)
        .over("symbol")
    )
    return factor.map_value(expr)


def ts_sum(factor: PolarsFactor, window=10):
    return _rolling(factor, "rolling_sum", window)


def ts_mean(factor: PolarsFactor, window=10):
    return _rolling(factor, "rolling_mean", window)


def ts_min(factor: PolarsFactor, window=10):
    return _rolling(factor, "rolling_min", window)


def ts_max(factor: PolarsFactor, window=10):
    return _rolling(factor, "rolling_max", window)


def ts_std_dev(factor: PolarsFactor, window=10):
    return _rolling(factor, "rolling_std", window)


def ts_delay(factor: PolarsFactor, period=1):
    factor = ensure_factor(factor)
    return factor.map_value(require_polars().col("value").shift(period).over("symbol"))


def ts_delta(factor: PolarsFactor, period=1):
    return factor - ts_delay(factor, period)


def ts_ema(factor: PolarsFactor, window=10):
    factor = ensure_factor(factor)
    return factor.map_value(
        require_polars().col("value").ewm_mean(span=window, adjust=False, min_samples=window).over("symbol")
    )


def ts_skewness(factor: PolarsFactor, window=10):
    factor = ensure_factor(factor)
    return factor.map_value(
        require_polars().col("value").rolling_skew(window_size=window, min_samples=window, bias=False).over("symbol")
    )


def ts_kurtosis(factor: PolarsFactor, window=10):
    factor = ensure_factor(factor)
    return factor.map_value(
        require_polars()
        .col("value")
        .rolling_kurtosis(window_size=window, min_samples=window, fisher=True, bias=False)
        .over("symbol")
    )


def ts_rank(factor: PolarsFactor, window=10):
    factor = ensure_factor(factor)
    return factor.map_value(
        require_polars().col("value").rolling_rank(window_size=window, method="max", min_samples=window).over("symbol")
        / window
    )


def ts_product(factor: PolarsFactor, window=10):
    return _rolling_map(factor, window, lambda values: values.product())


def ts_arg_max(factor: PolarsFactor, window=10):
    return _rolling_map(factor, window, lambda values: values.arg_max() + 1)


def ts_arg_min(factor: PolarsFactor, window=10):
    return _rolling_map(factor, window, lambda values: values.arg_min() + 1)


def ts_decay_linear(factor: PolarsFactor, period: int):
    if period <= 0:
        raise ValueError("period must be a positive integer")
    factor = ensure_factor(factor)
    pl = require_polars()
    # Match the existing pandas implementation, which uses np.convolve and
    # therefore applies the largest weight to the oldest value in the window.
    denom = period * (period + 1) / 2.0
    weighted_sum = sum(
        pl.col("value").shift(lag).over("symbol") * float(lag + 1)
        for lag in range(period)
    )
    valid_count = (
        pl.col("value")
        .is_finite()
        .cast(pl.Int32)
        .rolling_sum(window_size=period, min_samples=period)
        .over("symbol")
    )
    return factor.map_value(
        pl.when(valid_count == period).then(weighted_sum / denom).otherwise(None)
    )


def ts_zscore(factor: PolarsFactor, window=10, eps=1e-8):
    mean = ts_mean(factor, window)
    std = ts_std_dev(factor, window)
    return (factor - mean) / (std + eps)


def ts_pos(factor: PolarsFactor, window=10):
    roll_min = ts_min(factor, window)
    roll_max = ts_max(factor, window)
    return (factor - roll_min) / (roll_max - roll_min)


def ts_drawdown(factor: PolarsFactor, window=10):
    return factor / ts_max(factor, window) - 1.0


def ts_slope(factor: PolarsFactor, window=10):
    if window < 2:
        raise ValueError("window must be >= 2")
    factor = ensure_factor(factor)
    pl = require_polars()
    time_index = [float(idx) for idx in range(window)]
    time_mean = sum(time_index) / window
    centered_time = [value - time_mean for value in time_index]
    denom = sum(value * value for value in centered_time)
    weighted_sum = sum(
        pl.col("value").shift(lag).over("symbol") * centered_time[window - lag - 1]
        for lag in range(window)
    )
    valid_count = (
        pl.col("value")
        .is_finite()
        .cast(pl.Int32)
        .rolling_sum(window_size=window, min_samples=window)
        .over("symbol")
    )
    return factor.map_value(
        pl.when(valid_count == window).then(weighted_sum / denom).otherwise(None)
    )


def ts_covariance(x: PolarsFactor, y: PolarsFactor, window=10):
    if window < 2:
        raise ValueError("window must be >= 2")
    sum_xy = ts_sum(x * y, window)
    sum_x = ts_sum(x, window)
    sum_y = ts_sum(y, window)
    return (sum_xy - (sum_x * sum_y) / window) / (window - 1)


def correlation(x: PolarsFactor, y: PolarsFactor, window=10):
    cov = ts_covariance(x, y, window)
    return cov / (ts_std_dev(x, window) * ts_std_dev(y, window))


def _ts_regression(y: PolarsFactor, x: PolarsFactor, window: int, rettype: int, lag: int = 0):
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


def ts_beta(y: PolarsFactor, x: PolarsFactor, d, lag=0):
    return _ts_regression(y, x, d, 0, lag)


def ts_alpha(y: PolarsFactor, x: PolarsFactor, d, lag=0):
    return _ts_regression(y, x, d, 1, lag)


def ts_resid(y: PolarsFactor, x: PolarsFactor, d, lag=0):
    return _ts_regression(y, x, d, 2, lag)


def ts_r2(y: PolarsFactor, x: PolarsFactor, d, lag=0):
    return _ts_regression(y, x, d, 3, lag)


ts_std = ts_std_dev
ts_corr = correlation
