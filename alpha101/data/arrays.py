from __future__ import annotations

import numpy as np


def sample_indices_after_agg(n_rows: int, k_bars: int) -> np.ndarray:
    if n_rows <= 0:
        return np.array([], dtype=np.int64)
    k_bars = max(1, int(k_bars))
    start = k_bars - 1
    return np.arange(start, n_rows, k_bars, dtype=np.int64)


def nan_rowwise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape[0], np.nan, dtype=float)
    for i in range(x.shape[0]):
        mask = np.isfinite(x[i]) & np.isfinite(y[i])
        if int(mask.sum()) < 3:
            continue
        xv = x[i, mask]
        yv = y[i, mask]
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue
        out[i] = np.corrcoef(xv, yv)[0, 1]
    return out
