from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alpha101.data import sample_indices_after_agg
from alpha101.factors.backtesting.simulation import (
    aggregate_factor,
    compute_daily_arrays,
    forward_compound_returns,
    funding_time_mask,
)
from alpha101.factors.backtesting.config import LongShortBacktestConfig
from alpha101.factors.backtesting.curves import build_curve_df
from alpha101.factors.backtesting.metrics import compute_metrics


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
    factor_data_full = aggregate_factor(factor_raw, config.k_bars, config.factor_agg)

    target_raw_full = panel_df[target_col].to_numpy(dtype=np.float32, copy=False)
    funding_raw_full = panel_df[funding_cols].to_numpy(dtype=np.float32, copy=False)
    target_forward_full = forward_compound_returns(target_raw_full, config.k_bars)

    sampled_idx = sample_indices_after_agg(len(panel_df), config.k_bars)
    if sampled_idx.size == 0:
        raise ValueError("No sampled bars available after aggregation")

    eval_dates = panel_df.index[sampled_idx]
    factor_data = factor_data_full[sampled_idx]
    target_forward_eval = target_forward_full[sampled_idx][:, target_indices]
    target_eval_raw = target_raw_full[sampled_idx][:, target_indices]
    factor_eval_raw = factor_raw[sampled_idx]
    funding_eval_raw = np.nan_to_num(funding_raw_full[sampled_idx], nan=0.0)
    is_funding_time = funding_time_mask(eval_dates)

    daily = compute_daily_arrays(
        factor_data=factor_data,
        target_forward_eval=target_forward_eval,
        funding_eval_raw=funding_eval_raw,
        n_quantiles=n_quantiles,
        long_group=config.long_group,
        short_group=config.short_group,
        leverage=config.leverage,
        is_funding_time=is_funding_time,
        round_trip_fee=config.round_trip_fee,
    )

    metrics = compute_metrics(
        daily=daily,
        factor_eval_raw=factor_eval_raw,
        target_eval_raw=target_eval_raw,
        n_symbols=len(common_symbols),
        config=config,
    )
    curve_df = build_curve_df(eval_dates, daily, n_quantiles)
    return curve_df, metrics
