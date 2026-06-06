from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def aggregate_factor(factor_raw: np.ndarray, k_bars: int, factor_agg: str) -> np.ndarray:
    if k_bars <= 1:
        return factor_raw
    factor_df = pd.DataFrame(factor_raw, copy=False)
    if factor_agg == "mean":
        return factor_df.rolling(window=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)
    if factor_agg == "std":
        return factor_df.rolling(window=k_bars, min_periods=k_bars).std().to_numpy(dtype=np.float32)
    return factor_df.ewm(span=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)


def forward_compound_returns(target_raw_full: np.ndarray, k_bars: int) -> np.ndarray:
    if k_bars <= 1:
        return target_raw_full
    target_forward_full = np.ones_like(target_raw_full, dtype=np.float32)
    for step in range(k_bars):
        valid_len = target_raw_full.shape[0] - step
        if valid_len > 0:
            target_forward_full[:valid_len] *= 1.0 + target_raw_full[step:, :]
        target_forward_full[valid_len:, :] = np.nan
    return target_forward_full - 1.0


def funding_time_mask(eval_dates) -> np.ndarray:
    eval_ts = pd.DatetimeIndex(eval_dates)
    funding_period_ns = pd.Timedelta(hours=8).value
    return (eval_ts.view("int64") % funding_period_ns) == 0


def compute_daily_arrays(
    *,
    factor_data: np.ndarray,
    target_forward_eval: np.ndarray,
    funding_eval_raw: np.ndarray,
    n_quantiles: int,
    long_group: int,
    short_group: int,
    leverage: float,
    is_funding_time: np.ndarray,
    round_trip_fee: float,
) -> dict[str, Any]:
    valid_mask = np.isfinite(factor_data) & np.isfinite(target_forward_eval)
    n_dates = factor_data.shape[0]
    daily_pnl = np.zeros(n_dates, dtype=np.float32)
    quantile_daily = np.zeros((n_dates, n_quantiles), dtype=np.float32)
    turnover_daily = np.zeros(n_dates, dtype=np.float32)
    funding_cost_daily = np.zeros(n_dates, dtype=np.float32)
    turnover_sum = 0.0
    turnover_count = 0
    avg_long_funding_sum = 0.0
    avg_short_funding_sum = 0.0
    funding_count = 0
    prev_long_idx = None
    prev_short_idx = None

    for date_idx in np.flatnonzero(valid_mask.sum(axis=1) >= n_quantiles):
        valid_idx = np.where(valid_mask[date_idx])[0]
        factor_vals = factor_data[date_idx, valid_idx]
        target_vals = target_forward_eval[date_idx, valid_idx]
        sorted_local_idx = np.argsort(factor_vals, kind="mergesort")
        groups = np.array_split(sorted_local_idx, n_quantiles)

        for q in range(n_quantiles):
            q_local_idx = groups[q]
            if q_local_idx.size > 0:
                quantile_daily[date_idx, q] = float(target_vals[q_local_idx].mean())

        short_local_idx = groups[short_group - 1]
        long_local_idx = groups[long_group - 1]
        short_idx = valid_idx[short_local_idx]
        long_idx = valid_idx[long_local_idx]
        daily_pnl[date_idx] = (
            float(target_vals[long_local_idx].mean()) - float(target_vals[short_local_idx].mean())
        ) * leverage

        funding_vals = funding_eval_raw[date_idx, valid_idx]
        long_funding = float(funding_vals[long_local_idx].mean()) if long_local_idx.size > 0 else 0.0
        short_funding = float(funding_vals[short_local_idx].mean()) if short_local_idx.size > 0 else 0.0
        avg_long_funding_sum += long_funding
        avg_short_funding_sum += short_funding
        funding_count += 1
        if bool(is_funding_time[date_idx]):
            funding_cost_daily[date_idx] = (long_funding - short_funding) * leverage

        if prev_long_idx is not None:
            long_turnover = len(np.setdiff1d(long_idx, prev_long_idx)) / max(len(long_idx), 1)
            short_turnover = len(np.setdiff1d(short_idx, prev_short_idx)) / max(len(short_idx), 1)
            turnover_now = (long_turnover + short_turnover) * 0.5
            turnover_daily[date_idx] = turnover_now
            turnover_sum += turnover_now
            turnover_count += 1
        prev_long_idx = long_idx
        prev_short_idx = short_idx

    trading_cost_daily = turnover_daily * float(round_trip_fee) * leverage
    daily_pnl_net = daily_pnl - trading_cost_daily - funding_cost_daily
    return {
        "daily_pnl": daily_pnl,
        "daily_pnl_net": daily_pnl_net,
        "quantile_daily": quantile_daily,
        "trading_cost_daily": trading_cost_daily,
        "funding_cost_daily": funding_cost_daily,
        "turnover": float((turnover_sum / turnover_count) if turnover_count > 0 else 0.0),
        "avg_long_funding": float(avg_long_funding_sum / funding_count) if funding_count > 0 else 0.0,
        "avg_short_funding": float(avg_short_funding_sum / funding_count) if funding_count > 0 else 0.0,
    }
