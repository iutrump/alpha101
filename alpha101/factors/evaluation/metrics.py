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


def sharpe_ratio(returns, *, scale: float = 1.0, eps: float = 1e-12) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    mean = safe_float(np.nanmean(arr))
    finite = arr[np.isfinite(arr)]
    std = safe_float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return mean / std * scale if std > eps else 0.0


def information_ratio(values, eps: float = 1e-8) -> tuple[float, float, float]:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    mean = safe_float(series.mean())
    valid = series.dropna()
    std = safe_float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    return mean, std, mean / (std + eps)


def win_rate(returns) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    return safe_float(np.sum(arr > 0) / arr.size)


def profit_loss_ratio(returns, eps: float = 1e-12) -> float:
    arr = np.asarray(returns, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    wins = finite[finite > 0]
    losses = finite[finite < 0]
    if wins.size == 0 or losses.size == 0:
        return 0.0
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    return safe_float(avg_win / avg_loss) if avg_loss > eps else 0.0
