from __future__ import annotations

import numpy as np

from alpha101.data.views import LazyPolarsFactor, PolarsFactor
from alpha101.factors.expression.runtime import normalize_expression_code
from alpha101.factors.operator_lib.polars import OPERATOR_REGISTRY
from alpha101.factors.operator_lib.polars.utils import require_polars, to_numpy, to_pandas


def alpha_polars_fields(alpha_instance) -> dict:
    close = _to_polars(alpha_instance.close)
    return {
        "open": _to_polars(alpha_instance.open, template=close),
        "high": _to_polars(alpha_instance.high, template=close),
        "low": _to_polars(alpha_instance.low, template=close),
        "close": close,
        "volume": _to_polars(alpha_instance.volume, template=close),
        "returns": _to_polars(alpha_instance.returns, template=close),
        "vwap": _to_polars(alpha_instance.vwap, template=close),
        "market_return": _to_polars(alpha_instance.market_return, template=close),
        "funding": _to_polars(alpha_instance.funding, template=close),
        "cap": _to_polars(alpha_instance.cap, template=close),
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
    return result.is_all_nan()


def polars_result_to_numpy(result) -> np.ndarray:
    return to_numpy(result)


def polars_result_to_pandas(result):
    return to_pandas(result)


def _to_polars(value, *, template: LazyPolarsFactor | PolarsFactor | None = None):
    if isinstance(value, LazyPolarsFactor):
        return value
    if isinstance(value, PolarsFactor):
        return LazyPolarsFactor.from_factor(value)
    return LazyPolarsFactor.from_wide(value, template=template)


def abs_df(factor):
    return abs(factor)


def signed_log_df(factor):
    pl = require_polars()
    value = pl.col("value")
    return factor.map_value(value.sign() * (value.abs() + 1.0).log())


def sign_df(factor):
    return factor.map_value(require_polars().col("value").sign())


def sqrt_df(factor):
    pl = require_polars()
    return factor.map_value(pl.max_horizontal(pl.col("value"), pl.lit(0.0)).sqrt())


def exp_df(factor):
    return factor.map_value(require_polars().col("value").exp())


def maximum_df(left, right):
    pl = require_polars()
    if isinstance(left, (PolarsFactor, LazyPolarsFactor)) and isinstance(right, (PolarsFactor, LazyPolarsFactor)):
        joined = left.frame.rename({"value": "left"}).join(
            right.frame.rename({"value": "right"}),
            on=["date", "symbol"],
            how="left",
        )
        return left.with_frame(
            joined.with_columns(pl.max_horizontal("left", "right").alias("value")).select(["date", "symbol", "value"])
        )
    if isinstance(left, (PolarsFactor, LazyPolarsFactor)):
        return left.map_value(pl.max_horizontal(pl.col("value"), pl.lit(right)))
    return right.map_value(pl.max_horizontal(pl.lit(left), pl.col("value")))


def minimum_df(left, right):
    pl = require_polars()
    if isinstance(left, (PolarsFactor, LazyPolarsFactor)) and isinstance(right, (PolarsFactor, LazyPolarsFactor)):
        joined = left.frame.rename({"value": "left"}).join(
            right.frame.rename({"value": "right"}),
            on=["date", "symbol"],
            how="left",
        )
        return left.with_frame(
            joined.with_columns(pl.min_horizontal("left", "right").alias("value")).select(["date", "symbol", "value"])
        )
    if isinstance(left, (PolarsFactor, LazyPolarsFactor)):
        return left.map_value(pl.min_horizontal(pl.col("value"), pl.lit(right)))
    return right.map_value(pl.min_horizontal(pl.lit(left), pl.col("value")))
