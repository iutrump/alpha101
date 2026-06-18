from __future__ import annotations

from alpha101.experiments.protocol import (
    DateSplit,
    ashare_microcap_weekly_protocol,
    protocol_to_dict,
    validate_no_overlap,
)


def test_default_ashare_microcap_protocol_has_no_leakage_splits():
    protocol = ashare_microcap_weekly_protocol()

    assert validate_no_overlap(protocol.splits) == []
    assert protocol.name == "ashare_microcap_weekly_alphaprobe"
    assert protocol.frequency == "weekly"
    assert protocol.signal_weekday == "Monday"
    assert protocol.rebalance_weekday == "Tuesday"
    assert protocol.trade_lag_days == 1
    assert protocol.label_embargo_periods == 1

    payload = protocol_to_dict(protocol)
    assert [split["name"] for split in payload["splits"]] == ["train", "validation", "test"]
    assert payload["preprocessing"] == [
        "daily cross-sectional winsorize",
        "rank percentile normalization",
        "optional cap and industry neutralization A/B tests",
    ]
    assert "exclude the most recent unfinished forward label" in payload["no_leakage_rules"]
    assert "fit selection and ridge weights using history only" in payload["no_leakage_rules"]


def test_validate_no_overlap_reports_adjacent_or_overlapping_splits():
    splits = [
        DateSplit(name="train", start="2025-01-01", end="2025-01-10"),
        DateSplit(name="validation", start="2025-01-10", end="2025-01-20"),
    ]

    errors = validate_no_overlap(splits)

    assert errors == ["train ends on 2025-01-10 but validation starts on 2025-01-10"]
