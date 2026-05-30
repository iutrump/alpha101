from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.expression.runtime import build_eval_env, normalize_expression_code


_FAST_WORKER_ENV_BASE = None


def init_worker(fields: dict) -> None:
    global _FAST_WORKER_ENV_BASE
    _FAST_WORKER_ENV_BASE = build_eval_env(fields)


def result_is_all_nan(result) -> bool:
    try:
        return bool(np.asarray(result.isnull()).all())
    except Exception:
        return False


def prepare_factor_result(result: pd.DataFrame) -> pd.DataFrame:
    from alpha101.factors.operators import process_factor_wide_format

    result.index.name = "date"
    result.columns.name = "symbol"
    return process_factor_wide_format(result)


def eval_one_worker(item):
    factor_name, expr = item
    try:
        env = dict(_FAST_WORKER_ENV_BASE)
        lines = [line.strip() for line in normalize_expression_code(expr).split(";") if line.strip()]
        if not lines:
            raise ValueError("Expression cannot be empty")
        for line in lines[:-1]:
            var, sub_expr = line.split("=", 1)
            env[var.strip()] = eval(sub_expr.strip(), {}, env)

        result = eval(lines[-1], {}, env)
        if result is None:
            return factor_name, None, expr, f"Warning: {factor_name} returned None"
        if result_is_all_nan(result):
            return factor_name, None, expr, f"Warning: {factor_name} returned all NaN"
        return factor_name, prepare_factor_result(result), expr, None
    except Exception as exc:
        return factor_name, None, expr, f"Error evaluating {factor_name}: {exc}"
