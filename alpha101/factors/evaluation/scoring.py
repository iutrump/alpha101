from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.evaluation.metrics import (
    information_ratio,
    max_drawdown,
    safe_float,
    sharpe_ratio,
    win_rate,
)
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


def forward_returns_array(close: np.ndarray, periods: int = 1) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    out = np.full_like(close, np.nan, dtype=float)
    if periods <= 0:
        return out
    if close.shape[0] <= periods:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:-periods] = close[periods:] / close[:-periods] - 1.0
    return out


def score_factor_cross_section_array(
    factor: np.ndarray,
    target: np.ndarray,
    *,
    n_quantiles: int = 5,
) -> dict:
    factor = np.asarray(factor, dtype=float)
    target = np.asarray(target, dtype=float)
    if factor.shape != target.shape:
        raise ValueError(f"factor and target shape mismatch: {factor.shape} != {target.shape}")

    n_dates = factor.shape[0]
    ic_by_date = np.full(n_dates, np.nan, dtype=np.float32)
    long_short: list[float] = []
    valid_dates = 0
    nan_ratio = float(np.isnan(factor).sum() / max(factor.size, 1))

    for date_idx in range(n_dates):
        factor_row = factor[date_idx]
        target_row = target[date_idx]
        mask = np.isfinite(factor_row) & np.isfinite(target_row)
        n_valid = int(mask.sum())
        if n_valid >= 3:
            x = factor_row[mask].astype(np.float64, copy=False)
            y = target_row[mask].astype(np.float64, copy=False)
            x_std = x.std()
            y_std = y.std()
            if x_std > 0 and y_std > 0:
                ic_by_date[date_idx] = float(np.mean((x - x.mean()) * (y - y.mean())) / (x_std * y_std))

        if n_valid < n_quantiles:
            continue

        valid_dates += 1
        valid_factor = factor_row[mask]
        valid_target = target_row[mask]
        base_group_size, remainder = divmod(n_valid, n_quantiles)
        short_count = base_group_size + (1 if remainder else 0)
        long_count = base_group_size
        short_idx = np.argpartition(valid_factor, short_count - 1)[:short_count]
        long_start = n_valid - long_count
        long_idx = np.argpartition(valid_factor, long_start)[long_start:]
        long_short.append(float(valid_target[long_idx].mean() - valid_target[short_idx].mean()))

    ic_mean, ic_std, ic_ir = information_ratio(ic_by_date)
    pnl = np.asarray(long_short, dtype=np.float64)
    equity = pd.Series((1.0 + np.nan_to_num(pnl, nan=0.0)).cumprod(), dtype=float)
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
        "nan_ratio": float(nan_ratio),
    }
