from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alpha101.data.panel import nan_rowwise_corr, sample_indices_after_agg


SINGLE_SIDE_FEE = 0.0005
ROUND_TRIP_FEE = SINGLE_SIDE_FEE * 2


@dataclass(frozen=True)
class LongShortBacktestConfig:
    n_quantiles: int = 5
    long_group: int = 5
    short_group: int = 1
    leverage: float = 1.0
    k_bars: int = 1
    freq: str = "1d"
    factor_agg: str = "ewma"


def backtest_long_short(
    panel_df: pd.DataFrame,
    *,
    factor_name: str,
    target_col: str,
    config: LongShortBacktestConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if factor_name not in panel_df.columns.get_level_values(0):
        raise ValueError(f"Factor {factor_name} not found in panel")
    if target_col not in panel_df.columns.get_level_values(0):
        raise ValueError(f"Target {target_col} not found in panel")

    n_quantiles = config.n_quantiles
    target_symbols = list(panel_df[target_col].columns)
    target_symbol_set = set(target_symbols)
    target_symbol_to_idx = {sym: idx for idx, sym in enumerate(target_symbols)}

    factor_symbols = list(panel_df[factor_name].columns)
    common_symbols = [sym for sym in factor_symbols if sym in target_symbol_set]
    if len(common_symbols) < n_quantiles:
        raise ValueError(f"Not enough valid symbols ({len(common_symbols)}) for n_quantiles={n_quantiles}")

    factor_cols = [(factor_name, sym) for sym in common_symbols]
    funding_cols = [("funding", sym) for sym in common_symbols]
    target_indices = np.fromiter((target_symbol_to_idx[sym] for sym in common_symbols), dtype=np.int32)

    factor_raw = panel_df[factor_cols].to_numpy(dtype=np.float32, copy=False)
    factor_data_full = _aggregate_factor(factor_raw, config.k_bars, config.factor_agg)

    target_raw_full = panel_df[target_col].to_numpy(dtype=np.float32, copy=False)
    funding_raw_full = panel_df[funding_cols].to_numpy(dtype=np.float32, copy=False)
    target_forward_full = _forward_compound_returns(target_raw_full, config.k_bars)

    sampled_idx = sample_indices_after_agg(len(panel_df), config.k_bars)
    if sampled_idx.size == 0:
        raise ValueError("No sampled bars available after aggregation")

    eval_dates = panel_df.index[sampled_idx]
    factor_data = factor_data_full[sampled_idx]
    target_forward_eval = target_forward_full[sampled_idx][:, target_indices]
    target_eval_raw = target_raw_full[sampled_idx][:, target_indices]
    factor_eval_raw = factor_raw[sampled_idx]
    funding_eval_raw = funding_raw_full[sampled_idx]
    is_funding_time = _funding_time_mask(eval_dates)

    daily = _compute_daily_arrays(
        factor_data=factor_data,
        target_forward_eval=target_forward_eval,
        funding_eval_raw=funding_eval_raw,
        n_quantiles=n_quantiles,
        long_group=config.long_group,
        short_group=config.short_group,
        leverage=config.leverage,
        is_funding_time=is_funding_time,
    )

    metrics = _compute_metrics(
        daily=daily,
        factor_eval_raw=factor_eval_raw,
        target_eval_raw=target_eval_raw,
        n_symbols=len(common_symbols),
        config=config,
    )
    curve_df = _build_curve_df(eval_dates, daily, n_quantiles)
    return curve_df, metrics


def _aggregate_factor(factor_raw: np.ndarray, k_bars: int, factor_agg: str) -> np.ndarray:
    if k_bars <= 1:
        return factor_raw
    factor_df = pd.DataFrame(factor_raw, copy=False)
    if factor_agg == "mean":
        return factor_df.rolling(window=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)
    if factor_agg == "std":
        return factor_df.rolling(window=k_bars, min_periods=k_bars).std().to_numpy(dtype=np.float32)
    return factor_df.ewm(span=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)


def _forward_compound_returns(target_raw_full: np.ndarray, k_bars: int) -> np.ndarray:
    if k_bars <= 1:
        return target_raw_full
    target_forward_full = np.ones_like(target_raw_full, dtype=np.float32)
    for step in range(1, k_bars + 1):
        valid_len = target_raw_full.shape[0] - step
        if valid_len > 0:
            target_forward_full[:valid_len] *= 1.0 + target_raw_full[step:, :]
        target_forward_full[valid_len:, :] = np.nan
    return target_forward_full - 1.0


def _funding_time_mask(eval_dates) -> np.ndarray:
    eval_ts = pd.DatetimeIndex(eval_dates)
    funding_period_ns = pd.Timedelta(hours=8).value
    return (eval_ts.view("int64") % funding_period_ns) == 0


def _compute_daily_arrays(
    *,
    factor_data: np.ndarray,
    target_forward_eval: np.ndarray,
    funding_eval_raw: np.ndarray,
    n_quantiles: int,
    long_group: int,
    short_group: int,
    leverage: float,
    is_funding_time: np.ndarray,
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

    trading_cost_daily = turnover_daily * ROUND_TRIP_FEE * leverage
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


def _compute_metrics(
    *,
    daily: dict[str, Any],
    factor_eval_raw: np.ndarray,
    target_eval_raw: np.ndarray,
    n_symbols: int,
    config: LongShortBacktestConfig,
) -> dict[str, Any]:
    daily_pnl = daily["daily_pnl"]
    daily_pnl_net = daily["daily_pnl_net"]
    cum_pnl = np.cumprod(1.0 + daily_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    max_drawdown = float(np.min((cum_pnl - running_max) / np.maximum(running_max, 1e-8)))

    returns_mean = float(np.mean(daily_pnl))
    returns_std = float(np.std(daily_pnl, ddof=1))
    sharpe = (returns_mean / returns_std) * np.sqrt(len(daily_pnl)) if returns_std > 0 else 0.0
    returns_mean_net = float(np.mean(daily_pnl_net))
    returns_std_net = float(np.std(daily_pnl_net, ddof=1))
    sharpe_net = (returns_mean_net / returns_std_net) * np.sqrt(len(daily_pnl_net)) if returns_std_net > 0 else 0.0
    total_ret = float(np.prod(1.0 + daily_pnl) - 1.0)
    total_ret_net = float(np.prod(1.0 + daily_pnl_net) - 1.0)
    n_years = len(daily_pnl) * pd.to_timedelta(config.freq).total_seconds() * config.k_bars / (365.25 * 24 * 3600)
    returns_annual = float(np.sum(daily_pnl) / max(n_years, 0.01))
    returns_annual_net = float(np.sum(daily_pnl_net) / max(n_years, 0.01))
    funding_annual = float(np.sum(daily["funding_cost_daily"]) / max(n_years, 0.01))
    cagr = float((1 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1)
    cagr_net = float((1 + total_ret_net) ** (1.0 / max(n_years, 0.01)) - 1)
    win_rate = float(np.sum(daily_pnl > 0) / len(daily_pnl))

    ic_series = nan_rowwise_corr(factor_eval_raw, target_eval_raw)
    ic_mean = float(np.nanmean(ic_series)) if ic_series.size > 0 else 0.0
    ic_std = float(np.nanstd(ic_series, ddof=1)) if np.isfinite(ic_series).sum() > 1 else 0.0
    ic_ir = ic_mean / (ic_std + 1e-8)

    return {
        "sharpe": float(sharpe),
        "sharpe_after_cost": float(sharpe_net),
        "cagr": float(cagr),
        "cagr_after_cost": float(cagr_net),
        "returns": float(returns_annual),
        "returns_after_cost": float(returns_annual_net),
        "turnover": float(daily["turnover"]),
        "win_rate": float(win_rate),
        "drawdown": float(max_drawdown),
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "ic_ir": float(ic_ir),
        "obs_count": int(len(daily_pnl)),
        "symbols": int(n_symbols),
        "margin": float(returns_mean * 1000.0),
        "margin_after_cost": float(returns_mean_net * 1000.0),
        "single_side_fee": float(SINGLE_SIDE_FEE),
        "round_trip_fee": float(ROUND_TRIP_FEE),
        "avg_long_funding": float(daily["avg_long_funding"]),
        "avg_short_funding": float(daily["avg_short_funding"]),
        "funding_annual": float(funding_annual),
        "annual_cost_drag": float(returns_annual - returns_annual_net),
        "leverage": float(config.leverage),
        "long_group": int(config.long_group),
        "short_group": int(config.short_group),
    }


def _build_curve_df(eval_dates, daily: dict[str, Any], n_quantiles: int) -> pd.DataFrame:
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
