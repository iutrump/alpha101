from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.operators import process_factor_wide_format


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax().replace(0, np.nan)
    drawdown = equity.div(running_max).sub(1.0)
    return safe_float(drawdown.min())


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
    ic_mean = safe_float(ic_by_date.mean())
    ic_std = safe_float(ic_by_date.std(ddof=1))
    ic_ir = ic_mean / (ic_std + 1e-8)

    long_short = []
    valid_dates = 0
    for date in factor.index.intersection(target.index):
        row = pd.DataFrame({"factor": factor.loc[date], "target": target.loc[date]}).dropna()
        if len(row) < n_quantiles:
            continue
        valid_dates += 1
        ranked = row.sort_values("factor", kind="mergesort")
        groups = np.array_split(ranked, n_quantiles)
        long_short.append(groups[-1]["target"].mean() - groups[0]["target"].mean())

    pnl = pd.Series(long_short, dtype=float)
    equity = (1.0 + pnl.fillna(0.0)).cumprod()
    pnl_mean = safe_float(pnl.mean())
    pnl_std = safe_float(pnl.std(ddof=1))
    sharpe = pnl_mean / (pnl_std + 1e-8) * np.sqrt(max(len(pnl), 1))
    total_return = safe_float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0

    return {
        "fitness": float(ic_ir * 100.0 + sharpe),
        "returns": float(total_return),
        "sharpe_ratio": float(sharpe),
        "ic_ir": float(ic_ir),
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "drawdown": float(max_drawdown(equity)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "obs_count": int(valid_dates),
        "nan_ratio": float(factor.isna().sum().sum() / max(factor.size, 1)),
    }
