from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ASHARE_OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


def build_ashare_wide_frame(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "code",
    field_cols: Iterable[str] = ASHARE_OHLCV_FIELDS,
    cap_col: str | None = "market_cap",
    target_col: str | None = "label_5d",
    tradeable_col: str | None = None,
    include_vwap: bool = False,
) -> pd.DataFrame:
    """Convert a point-in-time A-share long panel to alpha101 wide format.

    Output columns use alpha101's expected MultiIndex shape: (field, symbol).
    By default, no synthetic VWAP is exposed; pass include_vwap=True to add
    typical-price VWAP for explicit exploratory use.
    """
    required = {date_col, symbol_col, *field_cols}
    if cap_col:
        required.add(cap_col)
    if target_col:
        required.add(target_col)
    if tradeable_col:
        required.add(tradeable_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"A-share frame is missing columns: {missing}")

    data = frame.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data[symbol_col] = data[symbol_col].astype(str)

    output_fields: list[tuple[str, str]] = [(str(col), str(col)) for col in field_cols]
    if cap_col:
        output_fields.append((str(cap_col), "cap"))
    if target_col:
        output_fields.append((str(target_col), "target"))

    numeric_cols = [source for source, _ in output_fields]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if tradeable_col:
        tradeable = data[tradeable_col].astype(bool)
        data.loc[~tradeable, numeric_cols] = np.nan

    if include_vwap:
        if {"high", "low", "close"} - set(data.columns):
            raise ValueError("include_vwap=True requires high, low, and close columns")
        data["vwap"] = (data["high"] + data["low"] + data["close"]) / 3.0
        output_fields.append(("vwap", "vwap"))

    wide_parts: dict[str, pd.DataFrame] = {}
    for source, field_name in output_fields:
        wide_parts[field_name] = (
            data.pivot_table(index=date_col, columns=symbol_col, values=source, aggfunc="last")
            .sort_index()
            .sort_index(axis=1)
        )

    wide = pd.concat(wide_parts, axis=1)
    wide.index.name = "date"
    wide.columns.names = ["field", "symbol"]
    return wide.sort_index(axis=1, level=[0, 1], sort_remaining=True)


def build_ashare_wide_frame_from_csv(
    paths: str | Path | Iterable[str | Path],
    **kwargs,
) -> pd.DataFrame:
    """Read one or more long CSV files and convert them to A-share wide format."""
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = list(paths)
    if not path_list:
        raise ValueError("At least one CSV path is required")
    frames = [pd.read_csv(path) for path in path_list]
    return build_ashare_wide_frame(pd.concat(frames, ignore_index=True), **kwargs)
