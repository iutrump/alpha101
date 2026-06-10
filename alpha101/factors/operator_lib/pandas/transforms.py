from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.operator_lib.pandas.cross_section import winsorize_mad, zscore
from alpha101.factors.operator_lib.pandas.time_series import ts_decay_linear

__all__ = [
    "process_factor_wide_format",
    "neg",
    "add",
    "sub",
    "mul",
    "div",
    "lt",
    "gt",
    "eq",
    "logical_or",
    "if_else",
    "trade_when",
]

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


def neg(x):
    return -x


def add(left, right):
    return left + right


def sub(left, right):
    return left - right


def mul(left, right):
    return left * right


def div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        return left / right


def lt(left, right):
    return left < right


def gt(left, right):
    return left > right


def eq(left, right):
    return left == right


def logical_or(left, right):
    return np.logical_or(left, right)


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
