from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

__all__ = [
    "make_ts_regression",
    "ts_beta",
    "ts_alpha",
    "ts_resid",
    "ts_r2",
]

def make_ts_regression(rettype: int) -> Callable[[pd.DataFrame, pd.DataFrame, int, int], pd.DataFrame]:
    def ts_regression(y: pd.DataFrame, x: pd.DataFrame, d: int, lag: int = 0) -> pd.DataFrame:
        if not y.shape == x.shape:
            raise ValueError("y 和 x 必须 shape 一致")
        if not y.index.equals(x.index) or not y.columns.equals(x.columns):
            raise ValueError("y 和 x 的 index/columns 必须一致")

        if lag > 0:
            y = y.shift(lag)
            x = x.shift(lag)

        sum_x = x.rolling(d, min_periods=d).sum()
        sum_y = y.rolling(d, min_periods=d).sum()
        sum_xy = (x * y).rolling(d, min_periods=d).sum()
        sum_x2 = (x * x).rolling(d, min_periods=d).sum()

        cov_xy = sum_xy - (sum_x * sum_y) / d
        var_x = sum_x2 - (sum_x * sum_x) / d

        beta = cov_xy / var_x
        beta[var_x == 0] = np.nan

        if rettype == 0:
            return beta

        mean_x = sum_x / d
        mean_y = sum_y / d
        alpha = mean_y - beta * mean_x

        if rettype == 1:
            return alpha
        if rettype == 2:
            return y - (alpha + beta * x)
        if rettype == 3:
            sum_y2 = (y * y).rolling(d, min_periods=d).sum()
            var_y = sum_y2 - (sum_y * sum_y) / d
            r2 = (cov_xy * cov_xy) / (var_x * var_y)
            r2[(var_x <= 0) | (var_y <= 0)] = np.nan
            return r2
        return y - (alpha + beta * x)

    return ts_regression


ts_beta = make_ts_regression(rettype=0)
ts_alpha = make_ts_regression(rettype=1)
ts_resid = make_ts_regression(rettype=2)
ts_r2 = make_ts_regression(rettype=3)
