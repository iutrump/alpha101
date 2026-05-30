from __future__ import annotations

import traceback
from typing import Any

import pandas as pd

from alpha101.factors.evaluation import forward_returns_array, score_factor_cross_section_array
from alpha101.factors.expression.runtime import build_eval_env, normalize_expression_code
from alpha101.factors.expression.worker import prepare_factor_result, result_is_all_nan


_SEARCH_WORKER_ENV_BASE: dict[str, Any] | None = None
_SEARCH_WORKER_CLOSE_COLUMNS: pd.Index | None = None
_SEARCH_WORKER_TARGET: Any = None
_SEARCH_WORKER_N_QUANTILES = 5
_SEARCH_WORKER_FORWARD_PERIODS = 1
_SEARCH_WORKER_MIN_OBS = 30
_SEARCH_WORKER_EXPRESSION_BACKEND = "pandas"


def init_search_worker(
    fields: dict,
    close: pd.DataFrame,
    n_quantiles: int,
    forward_periods: int,
    min_obs: int,
    expression_backend: str = "pandas",
) -> None:
    global _SEARCH_WORKER_ENV_BASE
    global _SEARCH_WORKER_CLOSE_COLUMNS
    global _SEARCH_WORKER_TARGET
    global _SEARCH_WORKER_N_QUANTILES
    global _SEARCH_WORKER_FORWARD_PERIODS
    global _SEARCH_WORKER_MIN_OBS
    global _SEARCH_WORKER_EXPRESSION_BACKEND

    _SEARCH_WORKER_EXPRESSION_BACKEND = expression_backend
    if expression_backend == "polars":
        from alpha101.factors.expression.polars_runtime import alpha_polars_fields, build_polars_eval_env

        _SEARCH_WORKER_ENV_BASE = build_polars_eval_env(alpha_polars_fields(_AlphaFieldAdapter(fields)))
    else:
        _SEARCH_WORKER_ENV_BASE = build_eval_env(fields)
    _SEARCH_WORKER_CLOSE_COLUMNS = close.columns
    _SEARCH_WORKER_TARGET = forward_returns_array(close.to_numpy(dtype=float, copy=False), forward_periods)
    _SEARCH_WORKER_N_QUANTILES = n_quantiles
    _SEARCH_WORKER_FORWARD_PERIODS = forward_periods
    _SEARCH_WORKER_MIN_OBS = min_obs


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
        if _SEARCH_WORKER_EXPRESSION_BACKEND == "polars":
            from alpha101.factors.expression.polars_runtime import polars_result_is_all_nan, polars_result_to_numpy
            from alpha101.factors.operator_lib.polars import process_factor_wide_format

            if polars_result_is_all_nan(result):
                raise ValueError("All NaN result")
            factor = process_factor_wide_format(result)
            factor_values = polars_result_to_numpy(factor)
            factor_columns = result.columns
        else:
            if result_is_all_nan(result):
                raise ValueError("All NaN result")
            factor = prepare_factor_result(result)
            factor_values = factor.to_numpy(dtype=float, copy=False)
            factor_columns = factor.columns

        target_indices = _SEARCH_WORKER_CLOSE_COLUMNS.get_indexer(factor_columns)
        if (target_indices < 0).any():
            raise ValueError("Factor contains symbols missing from close target")
        metrics = score_factor_cross_section_array(
            factor_values,
            _SEARCH_WORKER_TARGET[:, target_indices],
            n_quantiles=_SEARCH_WORKER_N_QUANTILES,
        )
        if metrics["obs_count"] < _SEARCH_WORKER_MIN_OBS:
            raise ValueError(f"Insufficient observations: {metrics['obs_count']} < {_SEARCH_WORKER_MIN_OBS}")
        return factor_name, metrics, None, None
    except Exception as exc:
        return factor_name, None, str(exc), traceback.format_exc()


class _AlphaFieldAdapter:
    def __init__(self, fields: dict[str, Any]):
        self.open = fields["open"]
        self.high = fields["high"]
        self.low = fields["low"]
        self.close = fields["close"]
        self.volume = fields["volume"]
        self.returns = fields["returns"]
        self.vwap = fields["vwap"]
        self.market_return = fields["market_return"]
        self.funding = fields["funding"]
        self.cap = fields["cap"]
