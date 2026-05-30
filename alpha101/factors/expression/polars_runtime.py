from __future__ import annotations

import numpy as np

from alpha101.factors.expression.runtime import normalize_expression_code
from alpha101.factors.operator_lib.polars import OPERATOR_REGISTRY
from alpha101.factors.operator_lib.polars.utils import from_numpy_like, require_polars, to_numpy


def alpha_polars_fields(alpha_instance) -> dict:
    return {
        "open": _to_polars(alpha_instance.open),
        "high": _to_polars(alpha_instance.high),
        "low": _to_polars(alpha_instance.low),
        "close": _to_polars(alpha_instance.close),
        "volume": _to_polars(alpha_instance.volume),
        "returns": _to_polars(alpha_instance.returns),
        "vwap": _to_polars(alpha_instance.vwap),
        "market_return": _to_polars(alpha_instance.market_return),
        "funding": _to_polars(alpha_instance.funding),
        "cap": _to_polars(alpha_instance.cap),
    }


def build_polars_eval_env(fields: dict) -> dict:
    pl = require_polars()
    env = {
        **fields,
        "np": np,
        "pl": pl,
        "abs": abs_df,
        "log": signed_log_df,
        "sign": sign_df,
        "sqrt": sqrt_df,
        "max": maximum_df,
        "min": minimum_df,
        "exp": exp_df,
    }
    env.update(OPERATOR_REGISTRY)
    return env


def evaluate_polars_expression(code: str, fields: dict):
    env = build_polars_eval_env(fields)
    lines = [line.strip() for line in normalize_expression_code(code).split(";") if line.strip()]
    if not lines:
        raise ValueError("Expression cannot be empty")

    for line in lines[:-1]:
        var, expr = line.split("=", 1)
        env[var.strip()] = eval(expr.strip(), {}, env)
    return eval(lines[-1], {}, env)


def polars_result_is_all_nan(result) -> bool:
    return bool(np.isnan(to_numpy(result)).all())


def polars_result_to_numpy(result) -> np.ndarray:
    return to_numpy(result)


def _to_polars(df):
    pl = require_polars()
    return pl.DataFrame(df.to_numpy(dtype=float, copy=False), schema=list(df.columns), orient="row")


def abs_df(df):
    pl = require_polars()
    return df.select([pl.col(col).abs().alias(col) for col in df.columns])


def signed_log_df(df):
    arr = to_numpy(df)
    return from_numpy_like(np.sign(arr) * np.log(np.abs(arr) + 1.0), df)


def sign_df(df):
    return from_numpy_like(np.sign(to_numpy(df)), df)


def sqrt_df(df):
    pl = require_polars()
    return df.select([pl.col(col).sqrt().alias(col) for col in df.columns])


def exp_df(df):
    pl = require_polars()
    return df.select([pl.col(col).exp().alias(col) for col in df.columns])


def maximum_df(left, right):
    if hasattr(left, "to_numpy") and hasattr(right, "to_numpy"):
        return from_numpy_like(np.maximum(to_numpy(left), to_numpy(right)), left)
    if hasattr(left, "to_numpy"):
        return from_numpy_like(np.maximum(to_numpy(left), right), left)
    return from_numpy_like(np.maximum(left, to_numpy(right)), right)


def minimum_df(left, right):
    if hasattr(left, "to_numpy") and hasattr(right, "to_numpy"):
        return from_numpy_like(np.minimum(to_numpy(left), to_numpy(right)), left)
    if hasattr(left, "to_numpy"):
        return from_numpy_like(np.minimum(to_numpy(left), right), left)
    return from_numpy_like(np.minimum(left, to_numpy(right)), right)
