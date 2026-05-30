from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

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


ts_beta = make_ts_regression(rettype=0)
ts_alpha = make_ts_regression(rettype=1)
ts_resid = make_ts_regression(rettype=2)
ts_r2 = make_ts_regression(rettype=3)
