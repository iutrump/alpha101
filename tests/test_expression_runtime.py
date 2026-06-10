from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.data import FactorDataView
from alpha101.factors.expression import FastExpressionEngine
from alpha101.factors.operator_lib import OPERATOR_REGISTRY, OPERATOR_SPECS, operator_params_by_category
from alpha101.factors.operator_lib.pandas.regression import ts_alpha, ts_beta, ts_r2, ts_resid
from alpha101.factors.operator_lib.pandas.time_series import correlation, ts_slope


def make_wide_data() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=8, freq="1D", tz="UTC")
    symbols = ["BTC", "ETH", "SOL"]
    fields = {}
    base = np.arange(len(dates) * len(symbols), dtype=float).reshape(len(dates), len(symbols)) + 10.0
    for idx, field in enumerate(["open", "high", "low", "close", "volume", "vwap", "cap", "funding"]):
        fields[field] = pd.DataFrame(base + idx, index=dates, columns=symbols)
    fields["target"] = fields["close"].pct_change(fill_method=None).shift(-1)
    return pd.concat(fields, axis=1)


def test_operator_registry_has_explicit_public_names():
    assert "ts_mean" in OPERATOR_REGISTRY
    assert "decay_linear" in OPERATOR_REGISTRY
    assert "delay" in OPERATOR_REGISTRY
    assert "delta" in OPERATOR_REGISTRY
    assert "product" in OPERATOR_REGISTRY
    assert "signed_power" in OPERATOR_REGISTRY
    assert "ts_cov" in OPERATOR_REGISTRY
    assert "ts_argmax" in OPERATOR_REGISTRY
    assert "rank" in OPERATOR_REGISTRY
    assert "process_factor_wide_format" in OPERATOR_REGISTRY
    assert "__builtins__" not in OPERATOR_REGISTRY


def test_operator_specs_drive_generation_categories():
    assert OPERATOR_SPECS["ts_mean"].category == "time_series"
    assert OPERATOR_SPECS["ts_corr"].category == "time_series_dual"
    assert OPERATOR_SPECS["rank"].category == "cross_section"
    assert operator_params_by_category("time_series")["ts_mean"] == [3, 12, 20, 30, 60]


def test_expression_batch_serial_and_process_match():
    engine = FastExpressionEngine(FactorDataView(make_wide_data()))
    expressions = {"alpha_a": "rank(ts_mean(close, 2))", "alpha_b": "zscore(ts_delta(vwap, 1))"}

    serial = engine.evaluate_batch(expressions, progress_bar=False, backend="serial")
    process = engine.evaluate_batch(expressions, progress_bar=False, backend="process", max_workers=2)

    pd.testing.assert_frame_equal(serial.sort_index(axis=1), process.sort_index(axis=1))


def test_expression_sqrt_clips_negative_values():
    engine = FastExpressionEngine(FactorDataView(make_wide_data()))

    result = engine.evaluate("sqrt(close - close - 1)")

    assert float(result.to_numpy().max()) == 0.0
    assert float(result.to_numpy().min()) == 0.0


def test_expression_runtime_supports_signed_power_and_decay_alias():
    wide = make_wide_data()
    engine = FastExpressionEngine(FactorDataView(wide))

    signed = engine.evaluate("signed_power(open - close, 2)")
    expected_signed = -np.power(np.abs(wide["open"] - wide["close"]), 2)
    pd.testing.assert_frame_equal(signed, expected_signed)

    decay_alias = engine.evaluate("decay_linear(close, 3)")
    decay_original = engine.evaluate("ts_decay_linear(close, 3)")
    pd.testing.assert_frame_equal(decay_alias, decay_original)


def test_expression_runtime_supports_ga_alpha_operator_names():
    wide = make_wide_data()
    engine = FastExpressionEngine(FactorDataView(wide))

    alias_expr = engine.evaluate("add(delay(close, 1), delta(open, 1))")
    expected = wide["close"].shift(1) + wide["open"].diff(1)
    pd.testing.assert_frame_equal(alias_expr, expected)

    conditional = engine.evaluate("if_else(or(lt(open, close), gt(volume, close)), close, open)")
    pd.testing.assert_frame_equal(conditional, wide["close"])

    product_alias = engine.evaluate("product(close / 100, 3)")
    product_original = engine.evaluate("ts_product(close / 100, 3)")
    pd.testing.assert_frame_equal(product_alias, product_original)

    derived = engine.evaluate("ts_vwap(3) + volume_weighted_price + adv3")
    assert derived.shape == wide["close"].shape
    assert np.isfinite(derived.iloc[2:].to_numpy()).all()


def test_ts_slope_matches_reference_rolling_apply():
    data = make_wide_data()["close"]
    window = 4
    t = np.arange(window, dtype=float)
    t_demean = t - t.mean()
    denominator = np.sum(t_demean ** 2)

    def reference(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return np.sum(t_demean * (values - values.mean())) / denominator

    expected = data.rolling(window=window).apply(reference, raw=True)
    actual = ts_slope(data, window=window)
    pd.testing.assert_frame_equal(actual, expected)


def test_ts_corr_matches_pandas_rolling_corr():
    wide = make_wide_data()
    x = wide["close"]
    y = wide["volume"]
    expected = x.rolling(4).corr(y)
    actual = correlation(x, y, 4)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-10)


def test_ts_regression_outputs_are_consistent():
    wide = make_wide_data()
    y = wide["close"]
    x = wide["volume"]
    window = 4
    beta = ts_beta(y, x, window)
    alpha = ts_alpha(y, x, window)
    resid = ts_resid(y, x, window)
    r2 = ts_r2(y, x, window)

    pd.testing.assert_frame_equal(resid, y - (alpha + beta * x))
    assert np.nanmin(r2.to_numpy()) >= -1e-9
    assert np.nanmax(r2.to_numpy()) <= 1.0 + 1e-9
