from __future__ import annotations

import numpy as np


def require_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "Polars expression backend requires `polars`. Install with `python -m pip install -e \".[polars]\"`."
        ) from exc
    return pl


def from_numpy_like(values: np.ndarray, template):
    pl = require_polars()
    return pl.DataFrame(values, schema=list(template.columns), orient="row")


def to_numpy(df) -> np.ndarray:
    return df.to_numpy().astype(float, copy=False)


def rolling_apply_numpy(df, window: int, func) -> object:
    arr = to_numpy(df)
    out = np.full(arr.shape, np.nan, dtype=float)
    if window <= 0:
        raise ValueError("window must be a positive integer")
    for row_idx in range(window - 1, arr.shape[0]):
        out[row_idx] = func(arr[row_idx - window + 1: row_idx + 1])
    return from_numpy_like(out, df)


def rowwise_rank_array(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for row_idx, row in enumerate(values):
        mask = np.isfinite(row)
        if not mask.any():
            continue
        valid = row[mask]
        order = np.argsort(valid, kind="mergesort")
        ranks = np.empty(valid.shape, dtype=float)
        ranks[order] = (np.arange(len(valid), dtype=float) + 1.0) / len(valid)
        out[row_idx, mask] = ranks
    return out
