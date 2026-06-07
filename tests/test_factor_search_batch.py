from __future__ import annotations

from alpha101.factors.search import FactorSearchEngine
from tests.test_expression_runtime import make_wide_data


def test_factor_search_batch_evaluates_population():
    search_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2)
    search_engine.min_obs = 1

    results = search_engine.evaluate_factors_batch(
        {
            "alpha_a": "rank(ts_mean(close, 2))",
            "alpha_b": "zscore(ts_delta(vwap, 1))",
        },
        backend="serial",
        progress_bar=False,
    )

    assert set(results) == {"alpha_a", "alpha_b"}
    assert all(metrics["status"] == "success" for metrics in results.values())
    assert all("fitness" in metrics for metrics in results.values())


def test_factor_search_seed_reproduces_first_expression():
    first_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, seed=7)
    first_expression = first_engine.new_random_expression()

    second_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, seed=7)

    assert first_expression == second_engine.new_random_expression()
