from __future__ import annotations

import numpy as np

from alpha101.factors.operator_lib.polars.cross_section import winsorize_mad, zscore
from alpha101.factors.operator_lib.polars.time_series import ts_decay_linear
from alpha101.factors.operator_lib.polars.utils import from_numpy_like, to_numpy

__all__ = [
    "if_else",
    "process_factor_wide_format",
    "trade_when",
]


def process_factor_wide_format(df, delay_days=0, decay_period=0):
    df_win = winsorize_mad(df, n=3.0)
    df_std = zscore(df_win)
    if delay_days:
        from alpha101.factors.operator_lib.polars.time_series import ts_delay

        df_std = ts_delay(df_std, delay_days)
    if decay_period > 1:
        return ts_decay_linear(df_std, period=decay_period)
    return df_std


def if_else(condition, true_val, false_val):
    cond = to_numpy(condition).astype(bool)
    true_arr = to_numpy(true_val) if hasattr(true_val, "to_numpy") else true_val
    false_arr = to_numpy(false_val) if hasattr(false_val, "to_numpy") else false_val
    return from_numpy_like(np.where(cond, true_arr, false_arr), condition)


def trade_when(condition, alpha, exit=np.nan):
    return if_else(condition, alpha, exit)
