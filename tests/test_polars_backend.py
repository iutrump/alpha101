from __future__ import annotations

import math

from alpha101.data import LazyPolarsFactor
from alpha101.factors.expression.polars_runtime import alpha_polars_fields, evaluate_polars_expression
from alpha101.factors.search import FactorSearchEngine
from tests.test_expression_runtime import make_wide_data


try:
    import polars  # noqa: F401
except ImportError:
    polars = None


def test_polars_search_backend_evaluates_expression():
    if polars is None:
        return
    search_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, expression_backend="polars")
    search_engine.min_obs = 1

    results = search_engine.evaluate_factors_batch(
        {"alpha_a": "rank(ts_mean(close, 2))"},
        backend="serial",
        progress_bar=False,
    )

    assert results["alpha_a"]["status"] == "success"
    assert "fitness" in results["alpha_a"]


def test_polars_runtime_returns_long_factor():
    if polars is None:
        return
    search_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, expression_backend="polars")
    result = evaluate_polars_expression(
        "rank(ts_mean(close, 2))",
        alpha_polars_fields(search_engine.alpha_obj),
    )

    assert isinstance(result, LazyPolarsFactor)
    assert result.frame.collect_schema().names() == ["date", "symbol", "value"]


def test_polars_search_backend_matches_pandas_metrics():
    if polars is None:
        return
    expressions = {
        "mean_rank": "rank(ts_mean(close, 2))",
        "delta_z": "zscore(ts_delta(vwap, 1))",
        "delay_rank": "rank(ts_delay(close, 1))",
        "minmax": "zscore(ts_max(high, 3) - ts_min(low, 3))",
        "ts_rank": "ts_rank(close, 3)",
        "ts_skew": "ts_skewness(close, 3)",
        "ts_kurt": "ts_kurtosis(close, 4)",
        "ts_product": "ts_product(close / 100, 3)",
        "ts_arg_max": "ts_arg_max(close, 3)",
        "ts_arg_min": "ts_arg_min(close, 3)",
        "ts_decay": "ts_decay_linear(close, 3)",
        "ts_slope": "ts_slope(close, 3)",
    }

    pandas_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, expression_backend="pandas")
    pandas_engine.min_obs = 1
    pandas_results = pandas_engine.evaluate_factors_batch(expressions, backend="serial", progress_bar=False)

    polars_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, expression_backend="polars")
    polars_engine.min_obs = 1
    polars_results = polars_engine.evaluate_factors_batch(expressions, backend="serial", progress_bar=False)

    for name in expressions:
        assert pandas_results[name]["status"] == "success"
        assert polars_results[name]["status"] == "success"
        assert polars_results[name]["obs_count"] == pandas_results[name]["obs_count"]
        assert _close(polars_results[name]["nan_ratio"], pandas_results[name]["nan_ratio"])
        assert _close(polars_results[name]["returns"], pandas_results[name]["returns"])
        assert _close(polars_results[name]["sharpe_ratio"], pandas_results[name]["sharpe_ratio"])
        assert _close(polars_results[name]["ic_ir"], pandas_results[name]["ic_ir"])
        assert _close(polars_results[name]["fitness"], pandas_results[name]["fitness"])


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
