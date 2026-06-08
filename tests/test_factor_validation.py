from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.factors.validation import annotate_pnl_redundancy
from alpha101.factors.validation import wide_for_report_mode
from tests.test_expression_runtime import make_wide_data


def test_validation_report_mode_uses_only_valid_segment():
    wide = make_wide_data()
    manifest = {"segment_ratios": [0.5, 0.25, 0.25]}

    validation_wide = wide_for_report_mode(wide, manifest, "validation")

    assert list(validation_wide.index) == list(wide.index[4:6])


def test_final_report_mode_keeps_full_dataset():
    wide = make_wide_data()
    manifest = {"segment_ratios": [0.5, 0.25, 0.25]}

    final_wide = wide_for_report_mode(wide, manifest, "final")

    assert final_wide is wide


def test_annotate_pnl_redundancy_marks_similar_pnl():
    index = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    records = [
        {"decision": "accepted_candidate", "expression": "alpha_a", "time_median_sharpe": 2.0},
        {"decision": "accepted_candidate", "expression": "alpha_b", "time_median_sharpe": 1.5},
        {"decision": "accepted_candidate", "expression": "alpha_c", "time_median_sharpe": 1.0},
    ]
    pnl_values = {
        "alpha_a": pd.Series([1, 2, 3, 4, 5], index=index, dtype=float),
        "alpha_b": pd.Series([2, 4, 6, 8, 10], index=index, dtype=float),
        "alpha_c": pd.Series([1, -1, 1, -1, 1], index=index, dtype=float),
    }

    annotate_pnl_redundancy(records, pnl_values, threshold=0.85)

    by_expression = {record["expression"]: record for record in records}
    assert by_expression["alpha_a"]["redundant_by_pnl"] is False
    assert by_expression["alpha_b"]["redundant_by_pnl"] is True
    assert by_expression["alpha_b"]["nearest_pnl_corr_expression"] == "alpha_a"
    assert np.isclose(by_expression["alpha_b"]["max_pnl_corr"], 1.0)
    assert by_expression["alpha_c"]["redundant_by_pnl"] is False
