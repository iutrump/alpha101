from __future__ import annotations

import pandas as pd

from alpha101.config import get_config
from alpha101.data.loading import compute_data_window, lookback_days_to_bars


def test_lookback_days_are_converted_to_timeframe_bars():
    assert lookback_days_to_bars(30, "4h") == 180
    assert lookback_days_to_bars(1.5, "1d") == 2


def test_compute_data_window_uses_days_plus_buffer_candles():
    start, end = compute_data_window(
        "4h",
        lookback_days=30,
        train_bars=None,
        test_start_date="2025-01-01",
        test_end_date="2025-01-31",
        buffer=200,
    )

    expected_history = pd.Timedelta(days=30) + 200 * pd.Timedelta(hours=4)
    assert start == pd.Timestamp("2025-01-01", tz="UTC") - expected_history
    assert end == pd.Timestamp("2025-01-31", tz="UTC") + pd.Timedelta(hours=4)


def test_compute_data_window_keeps_open_start_when_no_test_start():
    start, end = compute_data_window(
        "1d",
        lookback_days=30,
        train_bars=10,
        test_start_date=None,
        test_end_date="2025-01-31",
        buffer=200,
    )

    assert start is None
    assert end == pd.Timestamp("2025-01-31", tz="UTC") + pd.Timedelta(days=1)


def test_config_loads_quantile_groups(tmp_path):
    config_path = tmp_path / "alpha101.json"
    config_path.write_text(
        """
        {
          "pairs": ["BTC_USDT_USDT"],
          "n_quantiles": 10,
          "long_group": 10,
          "short_group": 1
        }
        """,
        encoding="utf-8",
    )

    cfg = get_config(config_path)

    assert cfg.n_quantiles == 10
    assert cfg.long_group == 10
    assert cfg.short_group == 1


def test_config_loads_pairs_from_same_file_exchange_whitelist(tmp_path):
    config_path = tmp_path / "alpha101.json"
    config_path.write_text(
        """
        {
          "pairs": [],
          "exchange": {
            "pair_whitelist": [
              "BTC/USDT:USDT",
              "ETH/USDT:USDT"
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    cfg = get_config(config_path)

    assert cfg.pairs == ["BTC_USDT_USDT", "ETH_USDT_USDT"]
