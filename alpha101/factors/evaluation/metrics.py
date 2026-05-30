from __future__ import annotations

import numpy as np
import pandas as pd


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax().replace(0, np.nan)
    drawdown = equity.div(running_max).sub(1.0)
    return safe_float(drawdown.min())


def max_drawdown_array(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / np.maximum(running_max, 1e-8)
    return safe_float(np.min(drawdown))


def sharpe_ratio(returns, *, scale: float = 1.0) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    mean = safe_float(np.nanmean(arr))
    std = safe_float(np.nanstd(arr, ddof=1))
    return mean / std * scale if std > 0 else 0.0


def information_ratio(values, eps: float = 1e-8) -> tuple[float, float, float]:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    mean = safe_float(series.mean())
    std = safe_float(series.std(ddof=1))
    return mean, std, mean / (std + eps)


def win_rate(returns) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    return safe_float(np.sum(arr > 0) / arr.size)
