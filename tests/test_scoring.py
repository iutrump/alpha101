from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.evaluation import (
    forward_returns_array,
    score_factor_cross_section,
    score_factor_cross_section_array,
    score_factor_search_array,
)


def test_score_factor_cross_section_returns_expected_keys():
    dates = pd.date_range("2025-01-01", periods=8, freq="1D", tz="UTC")
    symbols = ["BTC", "ETH", "SOL", "XRP"]
    base = np.tile(np.arange(len(symbols), dtype=float), (len(dates), 1))
    factor = pd.DataFrame(base + np.arange(len(dates))[:, None], index=dates, columns=symbols)
    close = pd.DataFrame(100.0 + base + np.arange(len(dates))[:, None], index=dates, columns=symbols)

    metrics = score_factor_cross_section(factor, close, n_quantiles=2, preprocess=False)

    assert metrics["obs_count"] > 0
    assert {"fitness", "returns", "sharpe_ratio", "ic_ir", "drawdown", "nan_ratio"}.issubset(metrics)


def test_array_score_matches_dataframe_score():
    dates = pd.date_range("2025-01-01", periods=8, freq="1D", tz="UTC")
    symbols = ["BTC", "ETH", "SOL", "XRP"]
    base = np.tile(np.arange(len(symbols), dtype=float), (len(dates), 1))
    factor = pd.DataFrame(base + np.arange(len(dates))[:, None], index=dates, columns=symbols)
    close = pd.DataFrame(100.0 + base + np.arange(len(dates))[:, None], index=dates, columns=symbols)

    frame_metrics = score_factor_cross_section(factor, close, n_quantiles=2, preprocess=False)
    array_metrics = score_factor_cross_section_array(
        factor.to_numpy(dtype=float),
        forward_returns_array(close.to_numpy(dtype=float), periods=1),
        n_quantiles=2,
    )

    assert array_metrics["obs_count"] == frame_metrics["obs_count"]
    assert np.isclose(array_metrics["returns"], frame_metrics["returns"])
    assert np.isclose(array_metrics["sharpe_ratio"], frame_metrics["sharpe_ratio"])


def test_array_score_uses_simple_returns_and_cost_adjusted_sharpe():
    factor = np.tile(np.arange(4, dtype=float), (6, 1))
    multipliers = np.array([1.0, 0.6, 1.2, 0.8, 1.4, 0.9])[:, None]
    target = np.array([-0.01, -0.005, 0.005, 0.01], dtype=float) * multipliers

    no_cost = score_factor_cross_section_array(factor, target, n_quantiles=2, transaction_cost=0.0)

    assert np.isclose(no_cost["returns"], float((0.015 * multipliers.ravel()).sum()))

    rotating_factor = factor.copy()
    rotating_factor[1::2] = rotating_factor[1::2, ::-1]
    with_cost = score_factor_cross_section_array(rotating_factor, target, n_quantiles=2, transaction_cost=0.001)
    assert with_cost["returns"] < no_cost["returns"]
    assert with_cost["sharpe_ratio"] <= no_cost["sharpe_ratio"]
    assert np.isclose(with_cost["returns"], with_cost["returns_after_cost"])
    assert np.isclose(no_cost["returns"], no_cost["returns_before_cost"])


def test_search_score_uses_train_valid_for_fitness_and_hides_test():
    n_dates = 80
    factor = np.tile(np.arange(4, dtype=float), (n_dates, 1))
    target = np.tile(np.array([-0.01, -0.005, 0.005, 0.01], dtype=float), (n_dates, 1))
    metrics = score_factor_search_array(
        factor,
        target,
        n_quantiles=2,
        min_segment_obs=5,
        transaction_cost=0.0,
    )

    expected = min(metrics["train_fitness"], metrics["valid_fitness"])
    max_abs = max(abs(metrics["train_fitness"]), abs(metrics["valid_fitness"]))
    min_abs = min(abs(metrics["train_fitness"]), abs(metrics["valid_fitness"]))
    expected *= min_abs / max_abs if max_abs > 0 else 0.0

    assert np.isclose(metrics["fitness"], expected)
    assert metrics["scoring_mode"] == "search"
    assert "test_fitness" not in metrics
    assert "test" not in metrics["segment_metrics"]


def test_final_score_reports_test_segment():
    n_dates = 80
    factor = np.tile(np.arange(4, dtype=float), (n_dates, 1))
    target = np.tile(np.array([-0.01, -0.005, 0.005, 0.01], dtype=float), (n_dates, 1))
    metrics = score_factor_search_array(
        factor,
        target,
        n_quantiles=2,
        min_segment_obs=5,
        transaction_cost=0.0,
        mode="final",
    )

    assert metrics["scoring_mode"] == "final"
    assert "test_fitness" in metrics
    assert metrics["test_obs_count"] > 0
    assert "test" in metrics["segment_metrics"]
