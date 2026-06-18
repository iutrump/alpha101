from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from alpha101.factors.evaluation.metrics import max_drawdown
from alpha101.factors.evaluation.scoring import (
    factor_pnl_series_array,
    score_factor_search_array,
)


RESERVED_EXTERNAL_FIELDS = {"target", "open", "high", "low", "close", "volume", "cap", "funding", "vwap"}


def external_factor_names(wide: pd.DataFrame, *, target_col: str = "target") -> list[str]:
    """Return factor fields available in an external-factor wide frame."""
    fields = list(dict.fromkeys(str(field) for field in wide.columns.get_level_values(0)))
    reserved = set(RESERVED_EXTERNAL_FIELDS)
    reserved.add(target_col)
    return [field for field in fields if field not in reserved]


def score_external_factors(
    wide: pd.DataFrame,
    *,
    factor_names: Iterable[str] | None = None,
    target_col: str = "target",
    n_quantiles: int = 5,
    min_segment_obs: int = 30,
    transaction_cost: float = 0.001,
    mode: str = "final",
    annualization: float = 52.0,
    directions: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Score precomputed factors against a precomputed target matrix."""
    target = _field_frame(wide, target_col)
    names = list(factor_names) if factor_names is not None else external_factor_names(wide, target_col=target_col)
    rows = []
    for factor_name in names:
        factor = _field_frame(wide, factor_name).reindex(index=target.index, columns=target.columns)
        factor_arr = factor.to_numpy(dtype=float, copy=False)
        target_arr = target.to_numpy(dtype=float, copy=False)
        metrics = score_factor_search_array(
            factor_arr,
            target_arr,
            n_quantiles=n_quantiles,
            min_segment_obs=min_segment_obs,
            transaction_cost=transaction_cost,
            mode=mode,
            annualization=annualization,
        )
        row = {"factor": factor_name}
        for key, value in metrics.items():
            if key != "segment_metrics":
                row[key] = value
        direction = _direction_for_factor(factor_name, metrics, directions)
        row.update(
            _long_only_metrics(
                factor_arr,
                target_arr,
                n_quantiles=n_quantiles,
                direction=direction,
                transaction_cost=transaction_cost,
            )
        )
        row["full_obs_count"] = int(_valid_date_count(factor, target, n_quantiles=n_quantiles))
        row["start_date"] = str(target.index[0]) if len(target.index) else None
        row["end_date"] = str(target.index[-1]) if len(target.index) else None
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["factor"])
    return pd.DataFrame(rows).sort_values(["fitness", "ic_ir"], ascending=False).reset_index(drop=True)


def build_external_factor_curve(
    wide: pd.DataFrame,
    factor_name: str,
    *,
    target_col: str = "target",
    n_quantiles: int = 5,
    transaction_cost: float = 0.001,
    direction: int = 1,
) -> pd.DataFrame:
    """Build a simple long-short curve for a precomputed external factor."""
    target = _field_frame(wide, target_col)
    factor = _field_frame(wide, factor_name).reindex(index=target.index, columns=target.columns)
    factor_arr = factor.to_numpy(dtype=float, copy=False)
    target_arr = target.to_numpy(dtype=float, copy=False)
    signed_factor_arr = factor_arr if int(direction) >= 0 else -factor_arr
    pnl_gross = factor_pnl_series_array(
        signed_factor_arr,
        target_arr,
        n_quantiles=n_quantiles,
        transaction_cost=0.0,
    )
    pnl_net = factor_pnl_series_array(
        signed_factor_arr,
        target_arr,
        n_quantiles=n_quantiles,
        transaction_cost=transaction_cost,
    )
    quantile_returns = _quantile_returns_array(factor_arr, target_arr, n_quantiles=n_quantiles)
    benchmark_returns = _benchmark_returns_array(target_arr)
    buy_group = _buy_group(n_quantiles, direction)
    selected_returns = quantile_returns[:, buy_group - 1]
    selected_turnover = _selected_turnover_array(factor_arr, target_arr, n_quantiles=n_quantiles, direction=direction)
    selected_cost = selected_turnover * float(transaction_cost)
    selected_returns_net = selected_returns - selected_cost
    excess_returns = selected_returns - benchmark_returns
    excess_returns_net = selected_returns_net - benchmark_returns

    curve = pd.DataFrame(
        {
            "date": [str(dt) for dt in target.index],
            "pnl": pnl_gross,
            "pnl_net": pnl_net,
            "cum_pnl": 1.0 + np.nan_to_num(pnl_gross, nan=0.0).cumsum(),
            "cum_pnl_net": 1.0 + np.nan_to_num(pnl_net, nan=0.0).cumsum(),
            "selected_ret_before_cost": selected_returns,
            "selected_ret": selected_returns_net,
            "selected_turnover": selected_turnover,
            "selected_cost": selected_cost,
            "benchmark_ret": benchmark_returns,
            "excess_ret_before_cost": excess_returns,
            "excess_ret": excess_returns_net,
            "selected_cum_before_cost": 1.0 + np.nan_to_num(selected_returns, nan=0.0).cumsum(),
            "selected_cum": 1.0 + np.nan_to_num(selected_returns_net, nan=0.0).cumsum(),
            "benchmark_cum": 1.0 + np.nan_to_num(benchmark_returns, nan=0.0).cumsum(),
            "excess_cum_before_cost": 1.0 + np.nan_to_num(excess_returns, nan=0.0).cumsum(),
            "excess_cum": 1.0 + np.nan_to_num(excess_returns_net, nan=0.0).cumsum(),
        }
    )
    for idx in range(n_quantiles):
        values = quantile_returns[:, idx]
        curve[f"q{idx + 1}_cum"] = 1.0 + np.nan_to_num(values, nan=0.0).cumsum()
    return curve


def _direction_for_factor(
    factor_name: str,
    metrics: dict,
    directions: dict[str, int] | None,
) -> int:
    if directions and factor_name in directions:
        return -1 if int(directions[factor_name]) < 0 else 1
    return -1 if float(metrics.get("ic_ir", 0.0)) < 0 else 1


def _buy_group(n_quantiles: int, direction: int) -> int:
    return int(n_quantiles) if int(direction) >= 0 else 1


def _long_only_metrics(
    factor: np.ndarray,
    target: np.ndarray,
    *,
    n_quantiles: int,
    direction: int,
    transaction_cost: float,
) -> dict:
    quantile_returns = _quantile_returns_array(factor, target, n_quantiles=n_quantiles)
    benchmark_returns = _benchmark_returns_array(target)
    buy_group = _buy_group(n_quantiles, direction)
    selected_returns = quantile_returns[:, buy_group - 1]
    selected_turnover = _selected_turnover_array(factor, target, n_quantiles=n_quantiles, direction=direction)
    selected_cost = selected_turnover * float(transaction_cost)
    selected_returns_net = selected_returns - selected_cost
    excess_returns = selected_returns - benchmark_returns
    excess_returns_net = selected_returns_net - benchmark_returns
    selected_equity = pd.Series(1.0 + np.nan_to_num(selected_returns_net, nan=0.0).cumsum(), dtype=float)
    excess_equity = pd.Series(1.0 + np.nan_to_num(excess_returns_net, nan=0.0).cumsum(), dtype=float)
    selected_valid = np.isfinite(selected_returns)
    selected_net_valid = np.isfinite(selected_returns_net)
    return {
        "direction": int(1 if direction >= 0 else -1),
        "buy_group": int(buy_group),
        "selected_returns_before_cost": float(np.nansum(selected_returns)),
        "selected_returns": float(np.nansum(selected_returns_net)),
        "universe_returns": float(np.nansum(benchmark_returns)),
        "selected_excess_returns_before_cost": float(np.nansum(excess_returns)),
        "selected_excess_returns": float(np.nansum(excess_returns_net)),
        "selected_drawdown": float(max_drawdown(selected_equity)),
        "selected_excess_drawdown": float(max_drawdown(excess_equity)),
        "selected_turnover": float(np.nanmean(selected_turnover)) if np.isfinite(selected_turnover).any() else 0.0,
        "selected_cost_drag": float(np.nansum(selected_cost)),
        "selected_win_rate": float(np.mean(selected_returns_net[selected_net_valid] > 0))
        if selected_net_valid.any()
        else 0.0,
        "selected_valid_count": int(selected_valid.sum()),
    }


def _field_frame(wide: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in wide.columns.get_level_values(0):
        raise ValueError(f"Field not found in external factor frame: {field}")
    frame = wide[field]
    frame.index.name = "date"
    frame.columns.name = "symbol"
    return frame


def _valid_date_count(factor: pd.DataFrame, target: pd.DataFrame, *, n_quantiles: int) -> int:
    valid = np.isfinite(factor.to_numpy(dtype=float, copy=False)) & np.isfinite(
        target.to_numpy(dtype=float, copy=False)
    )
    return int((valid.sum(axis=1) >= int(n_quantiles)).sum())


def _quantile_returns_array(factor: np.ndarray, target: np.ndarray, *, n_quantiles: int) -> np.ndarray:
    valid = np.isfinite(factor) & np.isfinite(target)
    out = np.full((factor.shape[0], n_quantiles), np.nan, dtype=np.float64)
    for date_idx in range(factor.shape[0]):
        mask = valid[date_idx]
        n_valid = int(mask.sum())
        if n_valid < n_quantiles:
            continue
        valid_factor = factor[date_idx][mask]
        valid_target = target[date_idx][mask]
        order = np.argsort(valid_factor, kind="mergesort")
        base_group_size, remainder = divmod(n_valid, n_quantiles)
        start = 0
        for group_idx in range(n_quantiles):
            size = base_group_size + (1 if group_idx < remainder else 0)
            stop = start + size
            if stop > start:
                out[date_idx, group_idx] = float(np.nanmean(valid_target[order[start:stop]]))
            start = stop
    return out


def _benchmark_returns_array(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(np.isfinite(target), target, np.nan), axis=1)


def _selected_turnover_array(
    factor: np.ndarray,
    target: np.ndarray,
    *,
    n_quantiles: int,
    direction: int,
) -> np.ndarray:
    factor = np.asarray(factor, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(factor) & np.isfinite(target)
    turnover = np.full(factor.shape[0], np.nan, dtype=np.float64)
    prev_selected_idx: np.ndarray | None = None
    buy_group = _buy_group(n_quantiles, direction) - 1
    for date_idx in range(factor.shape[0]):
        mask = valid[date_idx]
        n_valid = int(mask.sum())
        if n_valid < n_quantiles:
            continue
        valid_factor = factor[date_idx][mask]
        valid_idx = np.flatnonzero(mask)
        order = np.argsort(valid_factor, kind="mergesort")
        base_group_size, remainder = divmod(n_valid, n_quantiles)
        start = 0
        selected_idx = np.array([], dtype=int)
        for group_idx in range(n_quantiles):
            size = base_group_size + (1 if group_idx < remainder else 0)
            stop = start + size
            if group_idx == buy_group:
                selected_idx = valid_idx[order[start:stop]]
                break
            start = stop
        if prev_selected_idx is None:
            turnover[date_idx] = 0.0
        else:
            turnover[date_idx] = _turnover_ratio(selected_idx, prev_selected_idx)
        prev_selected_idx = selected_idx
    return turnover


def _turnover_ratio(current_idx: np.ndarray, prev_idx: np.ndarray) -> float:
    if current_idx.size == 0:
        return 0.0
    changed = np.setdiff1d(current_idx, prev_idx, assume_unique=False).size
    return float(changed / current_idx.size)
