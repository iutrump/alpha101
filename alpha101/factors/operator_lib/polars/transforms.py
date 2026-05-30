from __future__ import annotations

import numpy as np

from alpha101.data.views import PolarsFactor
from alpha101.factors.operator_lib.polars.cross_section import winsorize_mad, zscore
from alpha101.factors.operator_lib.polars.time_series import ts_decay_linear
from alpha101.factors.operator_lib.polars.utils import ensure_factor, require_polars

__all__ = [
    "if_else",
    "process_factor_wide_format",
    "trade_when",
]


def process_factor_wide_format(factor: PolarsFactor, delay_days=0, decay_period=0):
    factor_win = winsorize_mad(factor, n=3.0)
    factor_std = zscore(factor_win)
    if delay_days:
        from alpha101.factors.operator_lib.polars.time_series import ts_delay

        factor_std = ts_delay(factor_std, delay_days)
    if decay_period > 1:
        return ts_decay_linear(factor_std, period=decay_period)
    return factor_std


def if_else(condition: PolarsFactor, true_val, false_val):
    condition = ensure_factor(condition)
    pl = require_polars()
    if isinstance(true_val, PolarsFactor) and isinstance(false_val, PolarsFactor):
        joined = condition.frame.rename({"value": "condition"}).join(
            true_val.frame.rename({"value": "true_value"}),
            on=["date", "symbol"],
            how="left",
        ).join(
            false_val.frame.rename({"value": "false_value"}),
            on=["date", "symbol"],
            how="left",
        )
        return condition.with_frame(
            joined.with_columns(
                pl.when(pl.col("condition")).then(pl.col("true_value")).otherwise(pl.col("false_value")).alias("value")
            ).select(["date", "symbol", "value"])
        )
    if isinstance(true_val, PolarsFactor):
        joined = condition.frame.rename({"value": "condition"}).join(
            true_val.frame.rename({"value": "true_value"}),
            on=["date", "symbol"],
            how="left",
        )
        return condition.with_frame(
            joined.with_columns(
                pl.when(pl.col("condition")).then(pl.col("true_value")).otherwise(false_val).alias("value")
            ).select(["date", "symbol", "value"])
        )
    if isinstance(false_val, PolarsFactor):
        joined = condition.frame.rename({"value": "condition"}).join(
            false_val.frame.rename({"value": "false_value"}),
            on=["date", "symbol"],
            how="left",
        )
        return condition.with_frame(
            joined.with_columns(
                pl.when(pl.col("condition")).then(true_val).otherwise(pl.col("false_value")).alias("value")
            ).select(["date", "symbol", "value"])
        )
    return condition.map_value(pl.when(pl.col("value")).then(true_val).otherwise(false_val))


def trade_when(condition, alpha, exit=np.nan):
    return if_else(condition, alpha, exit)
