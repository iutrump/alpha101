from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_curve_df(eval_dates, daily: dict[str, Any], n_quantiles: int) -> pd.DataFrame:
    daily_pnl = daily["daily_pnl"]
    daily_pnl_net = daily["daily_pnl_net"]
    quantile_daily = daily["quantile_daily"]
    base_curve_df = pd.DataFrame(
        {
            "date": pd.to_datetime(eval_dates).astype(str),
            "pnl": daily_pnl.astype(float),
            "cum_pnl": np.cumprod(1.0 + daily_pnl).astype(float),
            "pnl_net": daily_pnl_net.astype(float),
            "cum_pnl_net": np.cumprod(1.0 + daily_pnl_net).astype(float),
            "trading_cost": daily["trading_cost_daily"].astype(float),
            "funding_cost": daily["funding_cost_daily"].astype(float),
        }
    )
    quantile_cum = np.cumprod(1.0 + quantile_daily, axis=0)
    quantile_cols = {}
    for q in range(n_quantiles):
        quantile_cols[f"q{q + 1}_pnl"] = quantile_daily[:, q].astype(float)
        quantile_cols[f"q{q + 1}_cum"] = quantile_cum[:, q].astype(float)
    return pd.concat([base_curve_df, pd.DataFrame(quantile_cols, index=base_curve_df.index)], axis=1)
