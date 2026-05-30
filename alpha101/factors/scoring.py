from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.metrics import information_ratio, max_drawdown, safe_float, sharpe_ratio, win_rate
from alpha101.factors.operator_lib import process_factor_wide_format


def forward_returns(close: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return close.pct_change(periods, fill_method=None).shift(-periods)


def score_factor_cross_section(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    forward_periods: int = 1,
    preprocess: bool = True,
) -> dict:
    if preprocess:
        factor = process_factor_wide_format(factor)
    target = forward_returns(close.reindex(columns=factor.columns), forward_periods)

    ic_by_date = factor.corrwith(target, axis=1).replace([np.inf, -np.inf], np.nan)
    ic_mean, ic_std, ic_ir = information_ratio(ic_by_date)

    long_short = []
    valid_dates = 0
    for date in factor.index.intersection(target.index):
        row = pd.DataFrame({"factor": factor.loc[date], "target": target.loc[date]}).dropna()
        if len(row) < n_quantiles:
            continue
        valid_dates += 1
        ranked = row.sort_values("factor", kind="mergesort")
        groups = np.array_split(np.arange(len(ranked)), n_quantiles)
        long_short.append(
            ranked.iloc[groups[-1]]["target"].mean()
            - ranked.iloc[groups[0]]["target"].mean()
        )

    pnl = pd.Series(long_short, dtype=float)
    equity = (1.0 + pnl.fillna(0.0)).cumprod()
    sharpe = sharpe_ratio(pnl, scale=np.sqrt(max(len(pnl), 1)))
    total_return = safe_float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0

    return {
        "fitness": float(ic_ir * 100.0 + sharpe),
        "returns": float(total_return),
        "sharpe_ratio": float(sharpe),
        "ic_ir": float(ic_ir),
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "drawdown": float(max_drawdown(equity)),
        "win_rate": float(win_rate(pnl)),
        "obs_count": int(valid_dates),
        "nan_ratio": float(factor.isna().sum().sum() / max(factor.size, 1)),
    }
