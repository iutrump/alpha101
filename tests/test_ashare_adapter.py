from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101.data import FactorDataView, build_ashare_wide_frame
from alpha101.factors.expression.runtime import alpha_fields


def _ashare_long_frame() -> pd.DataFrame:
    rows = []
    for date_idx, dt in enumerate(pd.date_range("2025-01-01", periods=3, freq="1D")):
        for code_idx, code in enumerate(["000001.XSHE", "600000.XSHG"]):
            rows.append(
                {
                    "date": dt,
                    "code": code,
                    "open": 10.0 + date_idx + code_idx,
                    "high": 10.5 + date_idx + code_idx,
                    "low": 9.5 + date_idx + code_idx,
                    "close": 10.2 + date_idx + code_idx,
                    "volume": 1000.0 + 10 * date_idx + code_idx,
                    "market_cap": 1_000_000.0 + 100 * code_idx,
                    "label_5d": 0.01 * (date_idx + 1) * (1 if code_idx == 0 else -1),
                    "is_tradeable": not (dt == pd.Timestamp("2025-01-02") and code == "600000.XSHG"),
                }
            )
    return pd.DataFrame(rows)


def test_build_ashare_wide_frame_omits_vwap_and_masks_untradeable_rows_by_default():
    wide = build_ashare_wide_frame(_ashare_long_frame(), tradeable_col="is_tradeable")

    assert isinstance(wide.columns, pd.MultiIndex)
    assert "vwap" not in wide.columns.get_level_values(0)
    assert {"open", "high", "low", "close", "volume", "cap", "target"}.issubset(
        set(wide.columns.get_level_values(0))
    )
    assert np.isnan(wide.loc[pd.Timestamp("2025-01-02"), ("close", "600000.XSHG")])
    assert np.isnan(wide.loc[pd.Timestamp("2025-01-02"), ("target", "600000.XSHG")])

    view = FactorDataView(wide)
    fields = alpha_fields(view)
    assert "vwap" not in fields
    assert "funding" not in fields
    assert fields["target"].equals(wide["target"])


def test_build_ashare_wide_frame_can_include_typical_price_vwap_when_explicitly_requested():
    wide = build_ashare_wide_frame(_ashare_long_frame(), include_vwap=True)

    expected = (wide["high"] + wide["low"] + wide["close"]) / 3.0
    pd.testing.assert_frame_equal(wide["vwap"], expected)
