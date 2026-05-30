from __future__ import annotations


DATA_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "vwap",
    "cap",
    "market_return",
    "funding",
]

TS_OPERATORS = {
    "ts_mean": [3, 12, 20, 30, 60],
    "ts_rank": [3, 5, 7, 10, 14, 20, 30],
    "ts_min": [3, 5, 7, 10, 14, 20, 30],
    "ts_max": [3, 5, 7, 10, 14, 20, 30],
    "ts_std_dev": [7, 14, 21, 28, 54],
    "ts_arg_max": [7, 14, 21, 28],
    "ts_arg_min": [7, 14, 21, 28],
    "ts_sum": [5, 10, 20, 30, 60],
    "ts_product": [5, 10, 20],
    "ts_skewness": [10, 20, 30, 60],
    "ts_kurtosis": [10, 20, 30, 60],
    "ts_decay_linear": [7, 14, 21, 28],
    "ts_drawdown": [7, 14, 21, 28],
    "ts_pos": [7, 14, 21, 28],
    "ts_zscore": [7, 14, 21, 28],
    "ts_ema": [6, 12, 24, 48],
    "ts_slope": [6, 12, 24, 48],
}

TS_DUAL_OPERATORS = {
    "ts_corr": [5, 10, 20, 30, 60],
    "ts_covariance": [5, 10, 20, 30, 60],
    "ts_alpha": [7, 14, 21, 28],
    "ts_r2": [7, 14, 21, 28],
    "ts_beta": [7, 14, 21, 28],
    "ts_resid": [6, 12, 24, 48],
}

DELAY_PARAMS = [1, 3, 5, 7, 10, 14, 20]
DELTA_PARAMS = [1, 3, 5, 7, 10]
CROSS_OPERATORS = ["rank", "scale", "zscore", "winsorize"]
BINARY_OPS = ["+", "-", "*", "/"]
UNARY_OPS = ["log", "-", "abs", "sqrt", "sign"]
MAX_DEPTH = 4
MAX_OPERATORS = 8

FORBIDDEN_PATTERNS = [
    (r"ts_mean\(ts_mean", "nested ts_mean"),
    (r"ts_rank\(ts_rank", "nested ts_rank"),
    (r"rank\(rank", "nested rank"),
    (r"scale\(scale", "nested scale"),
    (r"abs\(abs", "nested abs"),
    (r"ts_std_dev\(ts_std_dev", "nested ts_std_dev"),
    (r"ts_arg_max\([^)]*\)[^)]*ts_mean", "ts_arg_max with ts_mean"),
    (r"ts_arg_min\([^)]*\)[^)]*ts_mean", "ts_arg_min with ts_mean"),
    (r"ts_corr\(ts_corr", "nested ts_corr"),
    (r"ts_covariance\(ts_covariance", "nested ts_covariance"),
]
