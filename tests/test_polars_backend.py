from __future__ import annotations

import pytest

from alpha101.factors.search import FactorSearchEngine
from tests.test_expression_runtime import make_wide_data


pytest.importorskip("polars")


def test_polars_search_backend_evaluates_expression():
    search_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, expression_backend="polars")
    search_engine.min_obs = 1

    results = search_engine.evaluate_factors_batch(
        {"alpha_a": "rank(ts_mean(close, 2))"},
        backend="serial",
        progress_bar=False,
    )

    assert results["alpha_a"]["status"] == "success"
    assert "fitness" in results["alpha_a"]
