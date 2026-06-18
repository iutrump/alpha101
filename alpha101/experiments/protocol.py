from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass(frozen=True)
class DateSplit:
    name: str
    start: str
    end: str


@dataclass(frozen=True)
class ExperimentProtocol:
    name: str
    market: str
    universe: str
    frequency: str
    signal_weekday: str
    rebalance_weekday: str
    trade_lag_days: int
    target_horizon_days: int
    label_embargo_periods: int
    splits: list[DateSplit]
    preprocessing: list[str] = field(default_factory=list)
    selection: list[str] = field(default_factory=list)
    model: list[str] = field(default_factory=list)
    deployment: list[str] = field(default_factory=list)
    no_leakage_rules: list[str] = field(default_factory=list)


def ashare_microcap_weekly_protocol() -> ExperimentProtocol:
    """Default protocol for AlphaPROBE accepted factors in A-share microcaps."""
    return ExperimentProtocol(
        name="ashare_microcap_weekly_alphaprobe",
        market="A-share",
        universe="microcap400",
        frequency="weekly",
        signal_weekday="Monday",
        rebalance_weekday="Tuesday",
        trade_lag_days=1,
        target_horizon_days=5,
        label_embargo_periods=1,
        splits=[
            DateSplit(name="train", start="2022-05-31", end="2023-12-31"),
            DateSplit(name="validation", start="2024-01-01", end="2024-12-31"),
            DateSplit(name="test", start="2025-01-01", end="2025-12-27"),
        ],
        preprocessing=[
            "daily cross-sectional winsorize",
            "rank percentile normalization",
            "optional cap and industry neutralization A/B tests",
        ],
        selection=[
            "rolling ICIR ranking",
            "low factor-correlation pruning",
            "signed long-only PnL similarity pruning",
        ],
        model=[
            "weekly walk-forward ridge",
            "candidate TopN then selected TopK",
            "long-only microcap400 holdings",
        ],
        deployment=[
            "export weekly holdings CSV",
            "serve holdings to JoinQuant external reader",
            "keep stop-loss and rebuy cooldown in execution layer",
        ],
        no_leakage_rules=[
            "exclude the most recent unfinished forward label",
            "fit selection and ridge weights using history only",
            "do not tune test-period parameters after seeing test results",
            "compute PnL similarity from historical signed long-only curves only",
        ],
    )


def validate_no_overlap(splits: list[DateSplit]) -> list[str]:
    ordered = sorted(splits, key=lambda item: pd.Timestamp(item.start))
    errors: list[str] = []
    for left, right in zip(ordered, ordered[1:]):
        left_end = pd.Timestamp(left.end)
        right_start = pd.Timestamp(right.start)
        if left_end >= right_start:
            errors.append(
                f"{left.name} ends on {left.end} but {right.name} starts on {right.start}"
            )
    return errors


def protocol_to_dict(protocol: ExperimentProtocol) -> dict:
    payload = asdict(protocol)
    payload["split_errors"] = validate_no_overlap(protocol.splits)
    return payload
