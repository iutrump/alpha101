from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def normalize_pair(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


@dataclass
class ProjectConfig:
    data_root: Path = Path("user_data/data/binance/futures")
    pairs: list[str] = field(default_factory=list)
    timeframe: str = "4h"
    pre_buffer_candles: int = 200
    trade_per_k_bars: int = 1
    n_quantiles: int = 5
    long_group: int | None = None
    short_group: int = 1
    lookback_days: int = 0
    test_start_date: Optional[str] = "2025-01-01"
    test_end_date: Optional[str] = "2026-03-30"
    strategy_config_path: Optional[Path] = None
    output_dir: Path = Path("factor_search_results")
    single_side_fee: float = 0.0005
    round_trip_fee: float = 0.001
    search_segment_ratios: list[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pairs_from_strategy(path: Path) -> list[str]:
    payload = _load_json(path)
    pairs = payload.get("exchange", {}).get("pair_whitelist", [])
    return [normalize_pair(pair) for pair in pairs]


def get_config(config_path: str | Path | None = None) -> ProjectConfig:
    config_path = Path(config_path or os.getenv("ALPHA101_CONFIG", "configs/alpha101.json"))
    cfg = ProjectConfig()
    payload = _load_json(config_path)
    for key, value in payload.items():
        if not hasattr(cfg, key):
            continue
        if value is not None and (key.endswith("_path") or key in {"data_root", "output_dir"}):
            value = Path(value)
        setattr(cfg, key, value)

    if "round_trip_fee" not in payload and "single_side_fee" in payload:
        cfg.round_trip_fee = float(cfg.single_side_fee) * 2.0
    cfg.single_side_fee = float(cfg.single_side_fee)
    cfg.round_trip_fee = float(cfg.round_trip_fee)
    cfg.search_segment_ratios = [float(value) for value in cfg.search_segment_ratios]
    if len(cfg.search_segment_ratios) != 3 or any(value <= 0 for value in cfg.search_segment_ratios):
        raise ValueError("search_segment_ratios must contain three positive values")

    if not cfg.pairs:
        pair_source = cfg.strategy_config_path or config_path
        cfg.pairs = load_pairs_from_strategy(pair_source)
    return cfg
