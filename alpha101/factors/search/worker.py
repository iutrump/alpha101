from __future__ import annotations

import traceback
from typing import Any

import pandas as pd

from alpha101.factors.evaluation import score_factor_cross_section
from alpha101.factors.expression.runtime import build_eval_env, normalize_expression_code
from alpha101.factors.expression.worker import prepare_factor_result, result_is_all_nan


_SEARCH_WORKER_ENV_BASE: dict[str, Any] | None = None
_SEARCH_WORKER_CLOSE: pd.DataFrame | None = None
_SEARCH_WORKER_N_QUANTILES = 5
_SEARCH_WORKER_FORWARD_PERIODS = 1
_SEARCH_WORKER_MIN_OBS = 30


def init_search_worker(
    fields: dict,
    close: pd.DataFrame,
    n_quantiles: int,
    forward_periods: int,
    min_obs: int,
) -> None:
    global _SEARCH_WORKER_ENV_BASE
    global _SEARCH_WORKER_CLOSE
    global _SEARCH_WORKER_N_QUANTILES
    global _SEARCH_WORKER_FORWARD_PERIODS
    global _SEARCH_WORKER_MIN_OBS

    _SEARCH_WORKER_ENV_BASE = build_eval_env(fields)
    _SEARCH_WORKER_CLOSE = close
    _SEARCH_WORKER_N_QUANTILES = n_quantiles
    _SEARCH_WORKER_FORWARD_PERIODS = forward_periods
    _SEARCH_WORKER_MIN_OBS = min_obs


def evaluate_search_item(item: tuple[str, str]) -> tuple[str, dict | None, str | None, str | None]:
    factor_name, expr = item
    try:
        if _SEARCH_WORKER_ENV_BASE is None or _SEARCH_WORKER_CLOSE is None:
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
        metrics = score_factor_cross_section(
            factor,
            _SEARCH_WORKER_CLOSE.reindex(columns=factor.columns),
            n_quantiles=_SEARCH_WORKER_N_QUANTILES,
            forward_periods=_SEARCH_WORKER_FORWARD_PERIODS,
            preprocess=False,
        )
        if metrics["obs_count"] < _SEARCH_WORKER_MIN_OBS:
            raise ValueError(f"Insufficient observations: {metrics['obs_count']} < {_SEARCH_WORKER_MIN_OBS}")
        return factor_name, metrics, None, None
    except Exception as exc:
        return factor_name, None, str(exc), traceback.format_exc()
