from __future__ import annotations

import numpy as np

from alpha101.factors.operator_lib.polars.utils import from_numpy_like, rowwise_rank_array, to_numpy

__all__ = [
    "rank",
    "scale",
    "winsorize",
    "winsorize_group",
    "winsorize_mad",
    "zscore",
]


def rank(df):
    return from_numpy_like(rowwise_rank_array(to_numpy(df)), df)


def scale(df, scale=1, longscale=None, shortscale=None):
    arr = to_numpy(df)
    if longscale is not None or shortscale is not None:
        out = np.zeros_like(arr, dtype=float)
        longscale = 1 if longscale is None else longscale
        shortscale = 1 if shortscale is None else shortscale
        for row_idx, row in enumerate(arr):
            longs = row > 0
            shorts = row < 0
            long_sum = np.nansum(row[longs])
            short_sum = np.nansum(np.abs(row[shorts]))
            if long_sum:
                out[row_idx, longs] = row[longs] * (longscale / long_sum)
            if short_sum:
                out[row_idx, shorts] = row[shorts] * (shortscale / short_sum)
        return from_numpy_like(out, df)

    denom = np.nansum(np.abs(arr), axis=1)
    denom[denom == 0] = np.nan
    return from_numpy_like(arr / denom[:, None] * scale, df)


def zscore(df):
    arr = to_numpy(df)
    mean = np.nanmean(arr, axis=1)
    std = np.nanstd(arr, axis=1, ddof=1)
    std[std == 0] = np.nan
    out = (arr - mean[:, None]) / std[:, None]
    return from_numpy_like(np.nan_to_num(out, nan=0.0), df)


def winsorize(df, std: float = 4):
    arr = to_numpy(df)
    mean = np.nanmean(arr, axis=1)
    row_std = np.nanstd(arr, axis=1)
    lower = mean - std * row_std
    upper = mean + std * row_std
    return from_numpy_like(np.clip(arr, lower[:, None], upper[:, None]), df)


def winsorize_mad(df, n=3.0):
    arr = to_numpy(df)
    median = np.nanmedian(arr, axis=1)
    mad = np.nanmedian(np.abs(arr - median[:, None]), axis=1)
    limit = mad * n * 1.4826
    return from_numpy_like(np.clip(arr, (median - limit)[:, None], (median + limit)[:, None]), df)


def winsorize_group(df):
    arr = to_numpy(df)
    lower = np.nanquantile(arr, 0.01, axis=1)
    upper = np.nanquantile(arr, 0.99, axis=1)
    return from_numpy_like(np.clip(arr, lower[:, None], upper[:, None]), df)
