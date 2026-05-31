from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongShortBacktestConfig:
    n_quantiles: int = 5
    long_group: int = 5
    short_group: int = 1
    leverage: float = 1.0
    k_bars: int = 1
    freq: str = "1d"
    factor_agg: str = "ewma"
    single_side_fee: float = 0.0005
    round_trip_fee: float = 0.001
