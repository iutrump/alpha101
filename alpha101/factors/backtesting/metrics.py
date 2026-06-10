from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alpha101.data import nan_rowwise_corr
from alpha101.factors.backtesting.config import LongShortBacktestConfig
from alpha101.factors.evaluation.metrics import (
    information_ratio,
    max_drawdown_array,
    periods_per_year,
    profit_loss_ratio,
    safe_float,
    sharpe_ratio,
    win_rate,
)


def compute_metrics(
    *,
    daily: dict[str, Any],
    factor_eval_raw: np.ndarray,
    target_eval_raw: np.ndarray,
    n_symbols: int,
    config: LongShortBacktestConfig,
) -> dict[str, Any]:
    daily_pnl = daily["daily_pnl"]
    daily_pnl_net = daily["daily_pnl_net"]
    cum_pnl = np.cumprod(1.0 + daily_pnl)
    cum_pnl_net = np.cumprod(1.0 + daily_pnl_net)
    max_drawdown = max_drawdown_array(cum_pnl)
    max_drawdown_net = max_drawdown_array(cum_pnl_net)
    annualization = periods_per_year(config.freq, config.k_bars)

    returns_mean = float(np.mean(daily_pnl))
    sharpe = sharpe_ratio(daily_pnl, scale=np.sqrt(annualization))
    returns_mean_net = float(np.mean(daily_pnl_net))
    sharpe_net = sharpe_ratio(daily_pnl_net, scale=np.sqrt(annualization))
    total_ret = float(np.prod(1.0 + daily_pnl) - 1.0)
    total_ret_net = float(np.prod(1.0 + daily_pnl_net) - 1.0)
    n_years = len(daily_pnl) * pd.to_timedelta(config.freq).total_seconds() * config.k_bars / (365.25 * 24 * 3600)
    returns_annual = float(np.sum(daily_pnl) / max(n_years, 0.01))
    returns_annual_net = float(np.sum(daily_pnl_net) / max(n_years, 0.01))
    funding_annual = float(np.sum(daily["funding_cost_daily"]) / max(n_years, 0.01))
    cagr = float((1 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1)
    cagr_net = float((1 + total_ret_net) ** (1.0 / max(n_years, 0.01)) - 1)

    ic_series = nan_rowwise_corr(factor_eval_raw, target_eval_raw)
    ic_mean, ic_std, ic_ir = information_ratio(ic_series)

    return {
        "sharpe": float(sharpe),
        "sharpe_after_cost": float(sharpe_net),
        "cagr": float(cagr),
        "cagr_after_cost": float(cagr_net),
        "returns": float(returns_annual),
        "returns_after_cost": float(returns_annual_net),
        "turnover": float(daily["turnover"]),
        "win_rate": float(win_rate(daily_pnl)),
        "win_rate_after_cost": float(win_rate(daily_pnl_net)),
        "profit_loss_ratio": float(profit_loss_ratio(daily_pnl)),
        "profit_loss_ratio_after_cost": float(profit_loss_ratio(daily_pnl_net)),
        "drawdown": float(max_drawdown_net),
        "drawdown_before_cost": float(max_drawdown),
        "drawdown_after_cost": float(max_drawdown_net),
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "ic_ir": float(ic_ir),
        "obs_count": int(len(daily_pnl)),
        "symbols": int(n_symbols),
        "margin": float(safe_float(returns_mean) * 1000.0),
        "margin_after_cost": float(safe_float(returns_mean_net) * 1000.0),
        "single_side_fee": float(config.single_side_fee),
        "round_trip_fee": float(config.round_trip_fee),
        "avg_long_funding": float(daily["avg_long_funding"]),
        "avg_short_funding": float(daily["avg_short_funding"]),
        "funding_annual": float(funding_annual),
        "annual_cost_drag": float(returns_annual - returns_annual_net),
        "leverage": float(config.leverage),
        "long_group": int(config.long_group),
        "short_group": int(config.short_group),
    }
