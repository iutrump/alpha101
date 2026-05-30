from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.backtesting import LongShortBacktestConfig, backtest_long_short


def test_backtest_long_short_returns_curve_and_metrics():
    dates = pd.date_range("2025-01-01", periods=10, freq="8h", tz="UTC")
    symbols = ["BTC", "ETH", "SOL", "XRP"]
    base = np.tile(np.arange(len(symbols), dtype=float), (len(dates), 1))

    panel = pd.concat(
        {
            "alpha_test": pd.DataFrame(base + np.arange(len(dates))[:, None], index=dates, columns=symbols),
            "target": pd.DataFrame(base * 0.001 + 0.01, index=dates, columns=symbols),
            "funding": pd.DataFrame(np.zeros_like(base), index=dates, columns=symbols),
        },
        axis=1,
    )

    curve, metrics = backtest_long_short(
        panel,
        factor_name="alpha_test",
        target_col="target",
        config=LongShortBacktestConfig(n_quantiles=2, long_group=2, short_group=1, freq="8h"),
    )

    assert len(curve) == len(dates)
    assert {"pnl", "cum_pnl", "pnl_net", "q1_pnl", "q2_pnl"}.issubset(curve.columns)
    assert metrics["symbols"] == len(symbols)
    assert metrics["long_group"] == 2
    assert metrics["short_group"] == 1
