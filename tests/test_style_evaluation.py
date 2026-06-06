from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.evaluation import (
    StyleConfig,
    adf_1d,
    build_style_factors,
    factor_style_exposures,
    residualize_style_factor,
    rolling_market_beta,
    strategy_market_exposure,
    summarize_exposures,
)


def test_factor_style_exposures_recovers_linear_exposure():
    dates = pd.date_range("2025-01-01", periods=6, freq="1D", tz="UTC")
    symbols = [f"s{i}" for i in range(8)]
    base = np.tile(np.linspace(-1.0, 1.0, len(symbols)), (len(dates), 1))

    size = pd.DataFrame(base, index=dates, columns=symbols)
    momentum = pd.DataFrame(np.roll(base, 2, axis=1), index=dates, columns=symbols)
    volatility = pd.DataFrame(np.roll(base, 4, axis=1), index=dates, columns=symbols)
    factor = 0.8 * size - 0.4 * momentum + 0.2 * volatility

    exposures, residual = factor_style_exposures(
        factor,
        {"size": size, "momentum": momentum, "volatility": volatility},
        config=StyleConfig(min_count=5),
    )

    summary = summarize_exposures(exposures)
    assert summary["size"]["mean"] > 0.5
    assert summary["momentum"]["mean"] < -0.2
    assert summary["volatility"]["mean"] > 0.1
    assert float(residual.abs().mean().mean()) < 1e-12


def test_residualize_style_factor_removes_cross_section_exposure():
    dates = pd.date_range("2025-01-01", periods=6, freq="1D", tz="UTC")
    symbols = [f"s{i}" for i in range(10)]
    x = np.tile(np.arange(len(symbols), dtype=float), (len(dates), 1))
    size = pd.DataFrame(x, index=dates, columns=symbols)
    momentum = pd.DataFrame(x * x, index=dates, columns=symbols)
    volatility = pd.DataFrame(np.sin(x), index=dates, columns=symbols)
    factor = 2.0 * size + 0.5 * volatility

    residual = residualize_style_factor(
        factor,
        {"size": size, "momentum": momentum, "volatility": volatility},
        config=StyleConfig(min_count=6),
    )

    assert float(residual.abs().mean().mean()) < 1e-12


def test_build_style_factors_outputs_expected_keys_and_shape():
    dates = pd.date_range("2025-01-01", periods=8, freq="1D", tz="UTC")
    symbols = ["BTC", "ETH", "SOL"]
    close = pd.DataFrame(100.0 + np.arange(24).reshape(8, 3), index=dates, columns=symbols)
    cap = close * 1_000_000
    volume = pd.DataFrame(10_000.0 + np.arange(24).reshape(8, 3), index=dates, columns=symbols)
    funding = pd.DataFrame(np.full((8, 3), 0.0001), index=dates, columns=symbols)
    market_return = close.pct_change(fill_method=None).mean(axis=1)
    market_return = pd.DataFrame(
        np.repeat(market_return.to_numpy()[:, None], len(symbols), axis=1),
        index=dates,
        columns=symbols,
    )

    styles = build_style_factors(
        close,
        cap,
        market_return,
        volume,
        funding,
        momentum_window=2,
        volatility_window=3,
        beta_window=3,
        liquidity_window=3,
        reversal_window=2,
        funding_window=2,
    )

    assert set(styles) == {
        "size",
        "momentum",
        "volatility",
        "beta",
        "liquidity",
        "reversal",
        "funding",
    }
    for frame in styles.values():
        assert frame.shape == close.shape
        assert frame.index.equals(close.index)
        assert frame.columns.equals(close.columns)


def test_adf_statistic_is_more_negative_for_stationary_series():
    rng = np.random.default_rng(7)
    stationary = rng.normal(size=120)
    random_walk = np.cumsum(rng.normal(size=120))

    stationary_result = adf_1d(stationary, max_lag=1)
    random_walk_result = adf_1d(random_walk, max_lag=1)

    assert stationary_result["statistic"] < random_walk_result["statistic"]


def test_rolling_market_beta_recovers_known_beta():
    dates = pd.date_range("2025-01-01", periods=20, freq="1D", tz="UTC")
    symbols = ["a", "b"]
    market = pd.Series(np.linspace(-0.03, 0.04, len(dates)), index=dates)
    market_return = pd.DataFrame(
        np.repeat(market.to_numpy()[:, None], len(symbols), axis=1),
        index=dates,
        columns=symbols,
    )
    returns = pd.DataFrame(
        {
            "a": 2.0 * market.to_numpy(),
            "b": -0.5 * market.to_numpy(),
        },
        index=dates,
    )

    beta = rolling_market_beta(returns, market_return, window=8)

    assert np.isclose(beta["a"].dropna().iloc[-1], 2.0)
    assert np.isclose(beta["b"].dropna().iloc[-1], -0.5)


def test_strategy_market_exposure_recovers_known_beta():
    market = np.linspace(-0.02, 0.03, 30)
    strategy = 0.001 + 1.5 * market

    exposure = strategy_market_exposure(strategy, market)

    assert np.isclose(exposure["alpha"], 0.001)
    assert np.isclose(exposure["beta"], 1.5)
    assert exposure["r2"] > 0.99
