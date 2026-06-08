from __future__ import annotations

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
