from __future__ import annotations

from alpha101.factors.operator_lib import operator_names_by_category, operator_params_by_category


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

TS_OPERATORS = operator_params_by_category("time_series")
TS_DUAL_OPERATORS = operator_params_by_category("time_series_dual")
DELAY_PARAMS = operator_params_by_category("lag")["ts_delay"]
DELTA_PARAMS = operator_params_by_category("lag")["ts_delta"]
CROSS_OPERATORS = operator_names_by_category("cross_section")
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
