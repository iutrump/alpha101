from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_EXTERNAL_METADATA_COLUMNS = {
    "date",
    "signal_date",
    "code",
    "symbol",
    "market_cap",
    "log_market_cap",
    "industry",
    "label_5d",
    "target",
}


def infer_external_factor_columns(
    frame: pd.DataFrame,
    *,
    metadata_cols: Iterable[str] = DEFAULT_EXTERNAL_METADATA_COLUMNS,
) -> list[str]:
    """Infer precomputed factor columns from an accepted-factor export."""
    metadata = set(metadata_cols)
    return [str(col) for col in frame.columns if str(col) not in metadata]


def build_external_factor_wide_frame(
    paths: str | Path | Iterable[str | Path],
    *,
    date_col: str = "date",
    symbol_col: str = "code",
    target_col: str = "label_5d",
    factor_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read precomputed factor CSVs into alpha101 wide format.

    Output fields include ``target`` and one field per factor id, with columns
    shaped as ``(field, symbol)``.
    """
    frame = _read_external_factor_paths(paths)
    required = {date_col, symbol_col, target_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"External factor frame is missing columns: {missing}")

    data = frame.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data[symbol_col] = data[symbol_col].astype(str)
    factors = list(factor_cols) if factor_cols is not None else infer_external_factor_columns(data)
    if not factors:
        raise ValueError("No external factor columns found")

    missing_factors = sorted(set(factors) - set(data.columns))
    if missing_factors:
        raise ValueError(f"External factor columns are missing: {missing_factors}")

    numeric_cols = [target_col, *factors]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

    wide_parts: dict[str, pd.DataFrame] = {
        "target": _pivot_external_field(data, date_col, symbol_col, target_col),
    }
    for factor in factors:
        wide_parts[str(factor)] = _pivot_external_field(data, date_col, symbol_col, factor)

    wide = pd.concat(wide_parts, axis=1)
    wide.index.name = "date"
    wide.columns.names = ["field", "symbol"]
    return wide.sort_index(axis=1, level=[0, 1], sort_remaining=True)


def _read_external_factor_paths(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        path_items = _expand_path(paths)
    else:
        path_items = []
        for path in paths:
            path_items.extend(_expand_path(path))
    if not path_items:
        raise ValueError("At least one external factor CSV path is required")
    frames = [pd.read_csv(path) for path in path_items]
    return pd.concat(frames, ignore_index=True)


def _expand_path(path: str | Path) -> list[Path]:
    path_str = str(path)
    if any(ch in path_str for ch in "*?[]"):
        return [Path(item) for item in sorted(glob(path_str))]
    return [Path(path_str)]


def _pivot_external_field(
    data: pd.DataFrame,
    date_col: str,
    symbol_col: str,
    value_col: str,
) -> pd.DataFrame:
    return (
        data.pivot_table(index=date_col, columns=symbol_col, values=value_col, aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )
