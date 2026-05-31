from __future__ import annotations

import traceback
from typing import Any

import pandas as pd

from alpha101.factors.evaluation import forward_returns_array, score_factor_search_array
from alpha101.factors.expression.runtime import build_eval_env, normalize_expression_code
from alpha101.factors.expression.worker import prepare_factor_result, result_is_all_nan


_SEARCH_WORKER_ENV_BASE: dict[str, Any] | None = None
_SEARCH_WORKER_CLOSE_COLUMNS: pd.Index | None = None
_SEARCH_WORKER_TARGET: Any = None
_SEARCH_WORKER_N_QUANTILES = 5
_SEARCH_WORKER_FORWARD_PERIODS = 1
_SEARCH_WORKER_MIN_OBS = 30
_SEARCH_WORKER_SEGMENT_RATIOS = (0.70, 0.15, 0.15)
_SEARCH_WORKER_TRANSACTION_COST = 0.001


def init_search_worker(
    fields: dict,
    close: pd.DataFrame,
    n_quantiles: int,
    forward_periods: int,
    min_obs: int,
    segment_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    transaction_cost: float = 0.001,
) -> None:
    global _SEARCH_WORKER_ENV_BASE
    global _SEARCH_WORKER_CLOSE_COLUMNS
    global _SEARCH_WORKER_TARGET
    global _SEARCH_WORKER_N_QUANTILES
    global _SEARCH_WORKER_FORWARD_PERIODS
    global _SEARCH_WORKER_MIN_OBS
    global _SEARCH_WORKER_SEGMENT_RATIOS
    global _SEARCH_WORKER_TRANSACTION_COST

    _SEARCH_WORKER_ENV_BASE = build_eval_env(fields)
    _SEARCH_WORKER_CLOSE_COLUMNS = close.columns
    _SEARCH_WORKER_TARGET = forward_returns_array(close.to_numpy(dtype=float, copy=False), forward_periods)
    _SEARCH_WORKER_N_QUANTILES = n_quantiles
    _SEARCH_WORKER_FORWARD_PERIODS = forward_periods
    _SEARCH_WORKER_MIN_OBS = min_obs
    _SEARCH_WORKER_SEGMENT_RATIOS = tuple(float(value) for value in segment_ratios)
    _SEARCH_WORKER_TRANSACTION_COST = float(transaction_cost)


def evaluate_search_item(item: tuple[str, str]) -> tuple[str, dict | None, str | None, str | None]:
    factor_name, expr = item
    try:
        if _SEARCH_WORKER_ENV_BASE is None or _SEARCH_WORKER_CLOSE_COLUMNS is None or _SEARCH_WORKER_TARGET is None:
            raise RuntimeError("Search worker was not initialized")

        env = dict(_SEARCH_WORKER_ENV_BASE)
        lines = [line.strip() for line in normalize_expression_code(expr).split(";") if line.strip()]
        if not lines:
            raise ValueError("Expression cannot be empty")
        for line in lines[:-1]:
            var, sub_expr = line.split("=", 1)
            env[var.strip()] = eval(sub_expr.strip(), {}, env)

        result = eval(lines[-1], {}, env)
        if result is None:
            raise ValueError("Evaluation returned no result")
        if result_is_all_nan(result):
            raise ValueError("All NaN result")
        factor = prepare_factor_result(result)
        factor_values = factor.to_numpy(dtype=float, copy=False)
        factor_columns = factor.columns

        target_indices = _SEARCH_WORKER_CLOSE_COLUMNS.get_indexer(factor_columns)
        if (target_indices < 0).any():
            raise ValueError("Factor contains symbols missing from close target")
        metrics = score_factor_search_array(
            factor_values,
            _SEARCH_WORKER_TARGET[:, target_indices],
            n_quantiles=_SEARCH_WORKER_N_QUANTILES,
            min_segment_obs=_SEARCH_WORKER_MIN_OBS,
            segment_ratios=_SEARCH_WORKER_SEGMENT_RATIOS,
            transaction_cost=_SEARCH_WORKER_TRANSACTION_COST,
        )
        if metrics["obs_count"] < _SEARCH_WORKER_MIN_OBS:
            raise ValueError(f"Insufficient observations: {metrics['obs_count']} < {_SEARCH_WORKER_MIN_OBS}")
        return factor_name, metrics, None, None
    except Exception as exc:
        return factor_name, None, str(exc), traceback.format_exc()
