from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.alpha_data import Alphas
from alpha101.factors.expression import FastExpressionEngine
from alpha101.factors.operator_lib import OPERATOR_REGISTRY, OPERATOR_SPECS, operator_params_by_category


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
    assert "rank" in OPERATOR_REGISTRY
    assert "process_factor_wide_format" in OPERATOR_REGISTRY
    assert "__builtins__" not in OPERATOR_REGISTRY


def test_operator_specs_drive_generation_categories():
    assert OPERATOR_SPECS["ts_mean"].category == "time_series"
    assert OPERATOR_SPECS["ts_corr"].category == "time_series_dual"
    assert OPERATOR_SPECS["rank"].category == "cross_section"
    assert operator_params_by_category("time_series")["ts_mean"] == [3, 12, 20, 30, 60]


def test_expression_batch_serial_and_process_match():
    engine = FastExpressionEngine(Alphas(make_wide_data()))
    expressions = {"alpha_a": "rank(ts_mean(close, 2))", "alpha_b": "zscore(ts_delta(vwap, 1))"}

    serial = engine.evaluate_batch(expressions, progress_bar=False, backend="serial")
    process = engine.evaluate_batch(expressions, progress_bar=False, backend="process", max_workers=2)

    pd.testing.assert_frame_equal(serial.sort_index(axis=1), process.sort_index(axis=1))
