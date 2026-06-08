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
    return score_factor_cross_section_array(
        factor.to_numpy(dtype=float, copy=False),
        target.to_numpy(dtype=float, copy=False),
        n_quantiles=n_quantiles,
    )


def score_factor_search(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    forward_periods: int = 1,
    preprocess: bool = True,
    min_segment_obs: int = 30,
    segment_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    transaction_cost: float = 0.001,
    mode: str = "search",
) -> dict:
    if preprocess:
        factor = process_factor_wide_format(factor)
    target = forward_returns(close.reindex(columns=factor.columns), forward_periods)
    return score_factor_search_array(
        factor.to_numpy(dtype=float, copy=False),
        target.to_numpy(dtype=float, copy=False),
        n_quantiles=n_quantiles,
        min_segment_obs=min_segment_obs,
        segment_ratios=segment_ratios,
        transaction_cost=transaction_cost,
        mode=mode,
    )


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
    transaction_cost: float = 0.001,
) -> dict:
    factor = np.asarray(factor, dtype=float)
    target = np.asarray(target, dtype=float)
    if factor.shape != target.shape:
        raise ValueError(f"factor and target shape mismatch: {factor.shape} != {target.shape}")

    return _score_factor_segment(
        factor,
        target,
        n_quantiles=n_quantiles,
        transaction_cost=transaction_cost,
    )


def factor_pnl_series_array(
    factor: np.ndarray,
    target: np.ndarray,
    *,
    n_quantiles: int = 5,
    transaction_cost: float = 0.001,
) -> np.ndarray:
    factor = np.asarray(factor, dtype=float)
    target = np.asarray(target, dtype=float)
    if factor.shape != target.shape:
        raise ValueError(f"factor and target shape mismatch: {factor.shape} != {target.shape}")

    valid = np.isfinite(factor) & np.isfinite(target)
    pnl = np.full(factor.shape[0], np.nan, dtype=np.float64)
    prev_long_idx: np.ndarray | None = None
    prev_short_idx: np.ndarray | None = None

    for date_idx in range(factor.shape[0]):
        mask = valid[date_idx]
        n_valid = int(mask.sum())
        if n_valid < n_quantiles:
            continue

        valid_factor = factor[date_idx][mask]
        valid_target = target[date_idx][mask]
        valid_idx = np.flatnonzero(mask)
        base_group_size, remainder = divmod(n_valid, n_quantiles)
        short_count = base_group_size + (1 if remainder else 0)
        long_count = base_group_size
        short_idx = np.argpartition(valid_factor, short_count - 1)[:short_count]
        long_start = n_valid - long_count
        long_idx = np.argpartition(valid_factor, long_start)[long_start:]
        long_global_idx = valid_idx[long_idx]
        short_global_idx = valid_idx[short_idx]

        gross = float(valid_target[long_idx].mean() - valid_target[short_idx].mean())
        if prev_long_idx is None or prev_short_idx is None:
            turnover = 0.0
        else:
            long_turnover = _turnover_ratio(long_global_idx, prev_long_idx)
            short_turnover = _turnover_ratio(short_global_idx, prev_short_idx)
            turnover = float((long_turnover + short_turnover) * 0.5)
        pnl[date_idx] = gross - turnover * float(transaction_cost)
        prev_long_idx = long_global_idx
        prev_short_idx = short_global_idx

    return pnl


def score_factor_search_array(
    factor: np.ndarray,
    target: np.ndarray,
    *,
    n_quantiles: int = 5,
    segment_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    min_segment_obs: int = 30,
    transaction_cost: float = 0.001,
    mode: str = "search",
) -> dict:
    factor = np.asarray(factor, dtype=float)
    target = np.asarray(target, dtype=float)
    if factor.shape != target.shape:
        raise ValueError(f"factor and target shape mismatch: {factor.shape} != {target.shape}")

    slices = _time_segment_slices(factor.shape[0], segment_ratios)
    if mode not in {"search", "final"}:
        raise ValueError("mode must be 'search' or 'final'")

    segment_names = ("train", "valid") if mode == "search" else ("train", "valid", "test")
    segment_metrics = {}
    for name in segment_names:
        seg_slice = slices[name]
        segment_metrics[name] = _score_factor_segment(
            factor[seg_slice],
            target[seg_slice],
            n_quantiles=n_quantiles,
            transaction_cost=transaction_cost,
        )

    train = segment_metrics["train"]
    valid = segment_metrics["valid"]
    if train["obs_count"] < min_segment_obs:
        raise ValueError(f"Insufficient train observations: {train['obs_count']} < {min_segment_obs}")
    if valid["obs_count"] < min_segment_obs:
        raise ValueError(f"Insufficient valid observations: {valid['obs_count']} < {min_segment_obs}")

    robust_fitness = min(train["fitness"], valid["fitness"])
    max_abs = max(abs(train["fitness"]), abs(valid["fitness"]))
    min_abs = min(abs(train["fitness"]), abs(valid["fitness"]))
    stability_score = float(min_abs / max_abs) if max_abs > 0 else 0.0
    fitness = float(robust_fitness * stability_score)

    out = {
        **valid,
        "fitness": fitness,
        "robust_fitness": float(robust_fitness),
        "stability_score": stability_score,
        "scoring_mode": mode,
        "train_fitness": float(train["fitness"]),
        "valid_fitness": float(valid["fitness"]),
        "train_sharpe": float(train["sharpe_ratio"]),
        "valid_sharpe": float(valid["sharpe_ratio"]),
        "train_returns": float(train["returns"]),
        "valid_returns": float(valid["returns"]),
        "train_ic_ir": float(train["ic_ir"]),
        "valid_ic_ir": float(valid["ic_ir"]),
        "train_obs_count": int(train["obs_count"]),
        "valid_obs_count": int(valid["obs_count"]),
        "segment_metrics": segment_metrics,
    }
    if mode == "final":
        test = segment_metrics["test"]
        out.update(
            {
                "test_fitness": float(test["fitness"]),
                "test_sharpe": float(test["sharpe_ratio"]),
                "test_returns": float(test["returns"]),
                "test_ic_ir": float(test["ic_ir"]),
                "test_obs_count": int(test["obs_count"]),
            }
        )
    return out


def _score_factor_segment(
    factor: np.ndarray,
    target: np.ndarray,
    *,
    n_quantiles: int,
    transaction_cost: float,
) -> dict:
    n_dates = factor.shape[0]
    valid = np.isfinite(factor) & np.isfinite(target)
    valid_counts = valid.sum(axis=1)
    masked_factor = np.where(valid, factor, 0.0)
    masked_target = np.where(valid, target, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor_mean = masked_factor.sum(axis=1) / valid_counts
        target_mean = masked_target.sum(axis=1) / valid_counts
        factor_var = (masked_factor * masked_factor).sum(axis=1) / valid_counts - factor_mean * factor_mean
        target_var = (masked_target * masked_target).sum(axis=1) / valid_counts - target_mean * target_mean
        covariance = (masked_factor * masked_target).sum(axis=1) / valid_counts - factor_mean * target_mean
        ic_by_date = covariance / np.sqrt(factor_var * target_var)
    ic_by_date = ic_by_date.astype(np.float32, copy=False)
    ic_by_date[(valid_counts < 3) | (factor_var <= 0) | (target_var <= 0)] = np.nan

    gross_pnl: list[float] = []
    turnover_values: list[float] = []
    valid_dates = 0
    nan_ratio = float(np.isnan(factor).sum() / max(factor.size, 1))
    prev_long_idx: np.ndarray | None = None
    prev_short_idx: np.ndarray | None = None

    for date_idx in range(n_dates):
        factor_row = factor[date_idx]
        target_row = target[date_idx]
        mask = valid[date_idx]
        n_valid = int(valid_counts[date_idx])

        if n_valid < n_quantiles:
            continue

        valid_dates += 1
        valid_factor = factor_row[mask]
        valid_target = target_row[mask]
        valid_idx = np.flatnonzero(mask)
        base_group_size, remainder = divmod(n_valid, n_quantiles)
        short_count = base_group_size + (1 if remainder else 0)
        long_count = base_group_size
        short_idx = np.argpartition(valid_factor, short_count - 1)[:short_count]
        long_start = n_valid - long_count
        long_idx = np.argpartition(valid_factor, long_start)[long_start:]
        long_global_idx = valid_idx[long_idx]
        short_global_idx = valid_idx[short_idx]
        gross_pnl.append(float(valid_target[long_idx].mean() - valid_target[short_idx].mean()))

        if prev_long_idx is None or prev_short_idx is None:
            turnover_values.append(0.0)
        else:
            long_turnover = _turnover_ratio(long_global_idx, prev_long_idx)
            short_turnover = _turnover_ratio(short_global_idx, prev_short_idx)
            turnover_values.append(float((long_turnover + short_turnover) * 0.5))
        prev_long_idx = long_global_idx
        prev_short_idx = short_global_idx

    ic_mean, ic_std, ic_ir = information_ratio(ic_by_date)
    pnl_gross = np.asarray(gross_pnl, dtype=np.float64)
    turnover_arr = np.asarray(turnover_values, dtype=np.float64)
    cost = turnover_arr * float(transaction_cost)
    pnl_net = pnl_gross - cost

    # Search metrics use simple-interest accumulation. This avoids letting path
    # compounding dominate factor selection during genetic search.
    equity = pd.Series(1.0 + np.nan_to_num(pnl_net, nan=0.0).cumsum(), dtype=float)
    sharpe_gross = sharpe_ratio(pnl_gross, scale=np.sqrt(max(len(pnl_gross), 1)))
    sharpe_net = sharpe_ratio(pnl_net, scale=np.sqrt(max(len(pnl_net), 1)))
    total_return_gross = safe_float(np.nansum(pnl_gross))
    total_return_net = safe_float(np.nansum(pnl_net))
    avg_turnover = safe_float(np.nanmean(turnover_arr)) if turnover_arr.size else 0.0
    base = _fitness_base(sharpe_net, total_return_net, avg_turnover)

    return {
        "fitness": float(base),
        "returns": float(total_return_net),
        "returns_before_cost": float(total_return_gross),
        "returns_after_cost": float(total_return_net),
        "sharpe_ratio": float(sharpe_net),
        "sharpe_before_cost": float(sharpe_gross),
        "sharpe_after_cost": float(sharpe_net),
        "ic_ir": float(ic_ir),
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "drawdown": float(max_drawdown(equity)),
        "turnover": float(avg_turnover),
        "transaction_cost": float(transaction_cost),
        "win_rate": float(win_rate(pnl_net)),
        "obs_count": int(valid_dates),
        "nan_ratio": float(nan_ratio),
    }


def _fitness_base(sharpe: float, returns: float, turnover: float) -> float:
    positive_returns = max(float(returns), 0.0)
    turnover_floor = max(float(turnover), 0.15)
    if positive_returns <= 0.0:
        return 0.0
    return float(sharpe * np.sqrt(positive_returns / turnover_floor))


def _turnover_ratio(current_idx: np.ndarray, prev_idx: np.ndarray) -> float:
    if current_idx.size == 0:
        return 0.0
    changed = np.setdiff1d(current_idx, prev_idx, assume_unique=False).size
    return float(changed / current_idx.size)


def _time_segment_slices(
    n_rows: int,
    ratios: tuple[float, float, float],
) -> dict[str, slice]:
    if n_rows <= 0:
        return {
            "train": slice(0, 0),
            "valid": slice(0, 0),
            "test": slice(0, 0),
        }
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("segment ratios must sum to a positive value")
    train_ratio, valid_ratio, _ = (float(r) / total for r in ratios)
    train_end = int(n_rows * train_ratio)
    valid_end = train_end + int(n_rows * valid_ratio)
    train_end = min(max(train_end, 1), max(n_rows - 2, 1))
    valid_end = min(max(valid_end, train_end + 1), max(n_rows - 1, train_end + 1))
    return {
        "train": slice(0, train_end),
        "valid": slice(train_end, valid_end),
        "test": slice(valid_end, n_rows),
    }
