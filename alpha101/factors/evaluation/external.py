from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

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
) -> pd.DataFrame:
    """Score precomputed factors against a precomputed target matrix."""
    target = _field_frame(wide, target_col)
    names = list(factor_names) if factor_names is not None else external_factor_names(wide, target_col=target_col)
    rows = []
    for factor_name in names:
        factor = _field_frame(wide, factor_name).reindex(index=target.index, columns=target.columns)
        metrics = score_factor_search_array(
            factor.to_numpy(dtype=float, copy=False),
            target.to_numpy(dtype=float, copy=False),
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
) -> pd.DataFrame:
    """Build a simple long-short curve for a precomputed external factor."""
    target = _field_frame(wide, target_col)
    factor = _field_frame(wide, factor_name).reindex(index=target.index, columns=target.columns)
    factor_arr = factor.to_numpy(dtype=float, copy=False)
    target_arr = target.to_numpy(dtype=float, copy=False)
    pnl_gross = factor_pnl_series_array(
        factor_arr,
        target_arr,
        n_quantiles=n_quantiles,
        transaction_cost=0.0,
    )
    pnl_net = factor_pnl_series_array(
        factor_arr,
        target_arr,
        n_quantiles=n_quantiles,
        transaction_cost=transaction_cost,
    )
    quantile_returns = _quantile_returns_array(factor_arr, target_arr, n_quantiles=n_quantiles)

    curve = pd.DataFrame(
        {
            "date": [str(dt) for dt in target.index],
            "pnl": pnl_gross,
            "pnl_net": pnl_net,
            "cum_pnl": 1.0 + np.nan_to_num(pnl_gross, nan=0.0).cumsum(),
            "cum_pnl_net": 1.0 + np.nan_to_num(pnl_net, nan=0.0).cumsum(),
        }
    )
    for idx in range(n_quantiles):
        values = quantile_returns[:, idx]
        curve[f"q{idx + 1}_cum"] = 1.0 + np.nan_to_num(values, nan=0.0).cumsum()
    return curve


def _field_frame(wide: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in wide.columns.get_level_values(0):
        raise ValueError(f"Field not found in external factor frame: {field}")
    frame = wide[field]
    frame.index.name = "date"
    frame.columns.name = "symbol"
    return frame


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
