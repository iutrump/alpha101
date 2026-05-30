from __future__ import annotations

from alpha101.data.views import PolarsFactor
from alpha101.factors.operator_lib.polars.utils import ensure_factor, require_polars

__all__ = [
    "rank",
    "scale",
    "winsorize",
    "winsorize_group",
    "winsorize_mad",
    "zscore",
]


def rank(factor: PolarsFactor):
    factor = ensure_factor(factor)
    pl = require_polars()
    value = pl.col("value")
    valid_count = value.count().over("date")
    return factor.map_value(value.rank(method="average").over("date") / valid_count)


def scale(factor: PolarsFactor, scale=1, longscale=None, shortscale=None):
    factor = ensure_factor(factor)
    pl = require_polars()
    value = pl.col("value")
    if longscale is not None or shortscale is not None:
        longscale = 1 if longscale is None else longscale
        shortscale = 1 if shortscale is None else shortscale
        long_sum = pl.when(value > 0).then(value).otherwise(0.0).sum().over("date")
        short_sum = pl.when(value < 0).then(value.abs()).otherwise(0.0).sum().over("date")
        out = (
            pl.when((value > 0) & (long_sum != 0))
            .then(value * (longscale / long_sum))
            .when((value < 0) & (short_sum != 0))
            .then(value * (shortscale / short_sum))
            .otherwise(0.0)
        )
        return factor.map_value(out)

    denom = value.abs().sum().over("date")
    return factor.map_value(pl.when(denom == 0).then(None).otherwise(value / denom * scale))


def zscore(factor: PolarsFactor):
    factor = ensure_factor(factor)
    pl = require_polars()
    value = pl.col("value")
    mean = value.mean().over("date")
    std = value.std(ddof=1).over("date")
    return factor.map_value(pl.when((std == 0) | std.is_null()).then(0.0).otherwise((value - mean) / std))


def winsorize(factor: PolarsFactor, std: float = 4):
    factor = ensure_factor(factor)
    pl = require_polars()
    value = pl.col("value")
    mean = value.mean().over("date")
    row_std = value.std(ddof=0).over("date")
    lower = mean - std * row_std
    upper = mean + std * row_std
    return factor.map_value(
        pl.when(row_std.is_null() | (row_std <= 0))
        .then(value)
        .when(value < lower)
        .then(lower)
        .when(value > upper)
        .then(upper)
        .otherwise(value)
    )


def winsorize_mad(factor: PolarsFactor, n=3.0):
    factor = ensure_factor(factor)
    pl = require_polars()
    value = pl.col("value")
    median = value.median().over("date")
    mad = (value - median).abs().median().over("date")
    limit = mad * n * 1.4826
    lower = median - limit
    upper = median + limit
    return factor.map_value(
        pl.when(mad.is_null())
        .then(value)
        .when(value < lower)
        .then(lower)
        .when(value > upper)
        .then(upper)
        .otherwise(value)
    )


def winsorize_group(factor: PolarsFactor):
    factor = ensure_factor(factor)
    pl = require_polars()
    value = pl.col("value")
    lower = value.quantile(0.01).over("date")
    upper = value.quantile(0.99).over("date")
    return factor.map_value(
        pl.when(value < lower).then(lower).when(value > upper).then(upper).otherwise(value)
    )
