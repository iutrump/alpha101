from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from alpha101.cli.factor_search import _load_wide_data_from_args


def test_factor_search_cli_can_load_ashare_csv(tmp_path):
    csv_path = tmp_path / "ashare.csv"
    pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "code": "000001.XSHE",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "volume": 1000.0,
                "market_cap": 100_000_000.0,
                "label_5d": 0.03,
                "is_tradeable": True,
            },
            {
                "date": "2025-01-02",
                "code": "000001.XSHE",
                "open": 10.2,
                "high": 10.8,
                "low": 10.1,
                "close": 10.6,
                "volume": 1200.0,
                "market_cap": 101_000_000.0,
                "label_5d": -0.01,
                "is_tradeable": True,
            },
        ]
    ).to_csv(csv_path, index=False)
    args = SimpleNamespace(
        ashare_csv=[str(csv_path)],
        ashare_date_col="date",
        ashare_symbol_col="code",
        ashare_cap_col="market_cap",
        ashare_target_col="label_5d",
        ashare_tradeable_col="is_tradeable",
        ashare_include_vwap=False,
    )
    cfg = SimpleNamespace(
        pairs=[],
        lookback_days=0,
        data_root="",
        timeframe="1d",
        test_start_date=None,
        test_end_date=None,
        pre_buffer_candles=0,
    )

    wide = _load_wide_data_from_args(args, cfg)

    assert "target" in wide.columns.get_level_values(0)
    assert "vwap" not in wide.columns.get_level_values(0)
    assert float(wide.loc[pd.Timestamp("2025-01-01"), ("target", "000001.XSHE")]) == 0.03
