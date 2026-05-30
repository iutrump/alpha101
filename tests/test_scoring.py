from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.evaluation import (
    forward_returns_array,
    score_factor_cross_section,
    score_factor_cross_section_array,
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
