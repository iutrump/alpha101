from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from alpha101.config import get_config
from alpha101.data.panel import build_wide_df, nan_rowwise_corr, sample_indices_after_agg
from alpha101.factors.alpha_data import Alphas
from alpha101.factors.operators import process_factor_wide_format
from alpha101.factors.expression_engine import FastExpressionEngine


app = FastAPI(title="Alpha101 Factor Research Server", version="1.0.0")

_context_lock = threading.Lock()
_context: dict[str, Any] | None = None
SINGLE_SIDE_FEE = 0.0005
ROUND_TRIP_FEE = SINGLE_SIDE_FEE*2


class BacktestRequest(BaseModel):
    expression: str = Field(..., min_length=1, max_length=5000)
    n_quintiles: int = Field(5, ge=2, le=50)
    leverage: float = Field(1.0, ge=1.0, le=20.0)
    long_group: int | None = Field(None, ge=1, le=50)
    short_group: int | None = Field(None, ge=1, le=50)


def _load_context() -> dict[str, Any]:
    cfg = get_config()
    wide_data = build_wide_df(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        buffer=cfg.pre_buffer_candles,
    )
    engine = FastExpressionEngine(Alphas(wide_data))
    n_bars = int(
        cfg.pre_buffer_candles
        * pd.Timedelta("1d").total_seconds()
        // pd.Timedelta(cfg.timeframe).total_seconds()
    )
    return {
        "cfg": cfg,
        "wide_data": wide_data,
        "engine": engine,
        "n_bars": n_bars,
    }


def get_context() -> dict[str, Any]:
    global _context
    with _context_lock:
        if _context is None:
            _context = _load_context()
    return _context


def _compute_curve_and_metrics(
    panel_df: pd.DataFrame,
    alpha_name: str,
    target_col: str,
    n_quintiles: int,
    long_group: int,
    short_group: int,
    leverage: float,
    k_bars: int,
    freq: str,
    factor_agg: str = "ewma",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if alpha_name not in panel_df.columns.get_level_values(0):
        raise ValueError(f"Factor {alpha_name} not found in panel")
    if target_col not in panel_df.columns.get_level_values(0):
        raise ValueError(f"Target {target_col} not found in panel")

    target_symbols = list(panel_df[target_col].columns)
    target_symbol_set = set(target_symbols)
    target_symbol_to_idx = {sym: idx for idx, sym in enumerate(target_symbols)}

    factor_symbols = list(panel_df[alpha_name].columns)
    common_symbols = [sym for sym in factor_symbols if sym in target_symbol_set]
    if len(common_symbols) < n_quintiles:
        raise ValueError(
            f"Not enough valid symbols ({len(common_symbols)}) for n_quintiles={n_quintiles}"
        )

    factor_cols = [(alpha_name, sym) for sym in common_symbols]
    funding_cols = [("funding", sym) for sym in common_symbols]
    target_indices = np.fromiter(
        (target_symbol_to_idx[sym] for sym in common_symbols), dtype=np.int32
    )

    factor_raw = panel_df[factor_cols].to_numpy(dtype=np.float32, copy=False)
    if k_bars > 1:
        factor_df = pd.DataFrame(factor_raw, copy=False)
        if factor_agg == "mean":
            factor_data_full = factor_df.rolling(window=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)
        elif factor_agg == "std":
            factor_data_full = factor_df.rolling(window=k_bars, min_periods=k_bars).std().to_numpy(dtype=np.float32)
        else:
            factor_data_full = factor_df.ewm(span=k_bars, min_periods=k_bars).mean().to_numpy(dtype=np.float32)
    else:
        factor_data_full = factor_raw

    target_raw_full = panel_df[target_col].to_numpy(dtype=np.float32, copy=False)
    funding_raw_full = panel_df[funding_cols].to_numpy(dtype=np.float32, copy=False)
    if k_bars > 1:
        target_forward_full = np.ones_like(target_raw_full, dtype=np.float32)
        for step in range(1, k_bars + 1):
            valid_len = target_raw_full.shape[0] - step
            if valid_len > 0:
                target_forward_full[:valid_len] *= (1.0 + target_raw_full[step:, :])
            target_forward_full[valid_len:, :] = np.nan
        target_forward_full -= 1.0
    else:
        target_forward_full = target_raw_full

    sampled_idx = sample_indices_after_agg(len(panel_df), k_bars)
    if sampled_idx.size == 0:
        raise ValueError("No sampled bars available after aggregation")

    eval_dates = panel_df.index[sampled_idx]
    factor_data = factor_data_full[sampled_idx]
    target_forward_eval = target_forward_full[sampled_idx][:, target_indices]
    target_eval_raw = target_raw_full[sampled_idx][:, target_indices]
    factor_eval_raw = factor_raw[sampled_idx]
    funding_eval_raw = funding_raw_full[sampled_idx]
    eval_ts = pd.DatetimeIndex(eval_dates)
    funding_period_ns = pd.Timedelta(hours=8).value
    is_funding_time = ((eval_ts.view("int64") % funding_period_ns) == 0)

    valid_mask = np.isfinite(factor_data) & np.isfinite(target_forward_eval)
    n_dates = len(sampled_idx)
    daily_pnl = np.zeros(n_dates, dtype=np.float32)
    quantile_daily = np.zeros((n_dates, n_quintiles), dtype=np.float32)
    turnover_sum = 0.0
    turnover_count = 0
    turnover_daily = np.zeros(n_dates, dtype=np.float32)
    funding_cost_daily = np.zeros(n_dates, dtype=np.float32)
    avg_long_funding_sum = 0.0
    avg_short_funding_sum = 0.0
    funding_count = 0
    prev_long_idx = None
    prev_short_idx = None

    for date_idx in np.flatnonzero(valid_mask.sum(axis=1) >= n_quintiles):
        valid_idx = np.where(valid_mask[date_idx])[0]
        n_valid = len(valid_idx)
        if n_valid < n_quintiles:
            continue
        factor_vals = factor_data[date_idx, valid_idx]
        target_vals = target_forward_eval[date_idx, valid_idx]
        sorted_local_idx = np.argsort(factor_vals, kind="mergesort")
        groups = np.array_split(sorted_local_idx, n_quintiles)

        for q in range(n_quintiles):
            q_local_idx = groups[q]
            if q_local_idx.size > 0:
                quantile_daily[date_idx, q] = float(target_vals[q_local_idx].mean())

        short_local_idx = groups[short_group - 1]
        long_local_idx = groups[long_group - 1]
        short_idx = valid_idx[short_local_idx]
        long_idx = valid_idx[long_local_idx]
        short_ret = float(target_vals[short_local_idx].mean())
        long_ret = float(target_vals[long_local_idx].mean())
        daily_pnl[date_idx] = (long_ret - short_ret) * leverage
        funding_vals = funding_eval_raw[date_idx, valid_idx]
        long_funding = float(funding_vals[long_local_idx].mean()) if long_local_idx.size > 0 else 0.0
        short_funding = float(funding_vals[short_local_idx].mean()) if short_local_idx.size > 0 else 0.0
        avg_long_funding_sum += long_funding
        avg_short_funding_sum += short_funding
        funding_count += 1
        if bool(is_funding_time[date_idx]):
            funding_cost_daily[date_idx] = (long_funding - short_funding) * leverage

        if prev_long_idx is not None:
            long_turnover = len(np.setdiff1d(long_idx, prev_long_idx)) / max(len(long_idx), 1)
            short_turnover = len(np.setdiff1d(short_idx, prev_short_idx)) / max(len(short_idx), 1)
            turnover_now = (long_turnover + short_turnover) * 0.5
            turnover_daily[date_idx] = turnover_now
            turnover_sum += turnover_now
            turnover_count += 1
        prev_long_idx = long_idx
        prev_short_idx = short_idx

    cum_pnl = np.cumprod(1.0 + daily_pnl)
    trading_cost_daily = turnover_daily * ROUND_TRIP_FEE * leverage
    daily_pnl_net = daily_pnl - trading_cost_daily - funding_cost_daily
    cum_pnl_net = np.cumprod(1.0 + daily_pnl_net)
    quantile_cum = np.cumprod(1.0 + quantile_daily, axis=0)
    running_max = np.maximum.accumulate(cum_pnl)
    max_drawdown = float(np.min((cum_pnl - running_max) / np.maximum(running_max, 1e-8)))

    returns_mean = float(np.mean(daily_pnl))
    returns_std = float(np.std(daily_pnl, ddof=1))
    sharpe = (returns_mean / returns_std) * np.sqrt(len(daily_pnl)) if returns_std > 0 else 0.0
    returns_mean_net = float(np.mean(daily_pnl_net))
    returns_std_net = float(np.std(daily_pnl_net, ddof=1))
    sharpe_net = (returns_mean_net / returns_std_net) * np.sqrt(len(daily_pnl_net)) if returns_std_net > 0 else 0.0
    total_ret = float(np.prod(1.0 + daily_pnl) - 1.0)
    total_ret_net = float(np.prod(1.0 + daily_pnl_net) - 1.0)
    n_years = len(daily_pnl) * pd.to_timedelta(freq).total_seconds() * k_bars / (365.25 * 24 * 3600)
    returns_annual = float(np.sum(daily_pnl) / max(n_years, 0.01))
    returns_annual_net = float(np.sum(daily_pnl_net) / max(n_years, 0.01))
    funding_annual = float(np.sum(funding_cost_daily) / max(n_years, 0.01))
    cagr = float((1 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1)
    cagr_net = float((1 + total_ret_net) ** (1.0 / max(n_years, 0.01)) - 1)
    win_rate = float(np.sum(daily_pnl > 0) / len(daily_pnl))
    turnover = float((turnover_sum / turnover_count) if turnover_count > 0 else 0.0)
    avg_long_funding = float(avg_long_funding_sum / funding_count) if funding_count > 0 else 0.0
    avg_short_funding = float(avg_short_funding_sum / funding_count) if funding_count > 0 else 0.0

    ic_series = nan_rowwise_corr(factor_eval_raw, target_eval_raw)
    ic_mean = float(np.nanmean(ic_series)) if ic_series.size > 0 else 0.0
    ic_std = float(np.nanstd(ic_series, ddof=1)) if np.isfinite(ic_series).sum() > 1 else 0.0
    ic_ir = ic_mean / (ic_std + 1e-8)

    base_curve_df = pd.DataFrame(
        {
            "date": pd.to_datetime(eval_dates).astype(str),
            "pnl": daily_pnl.astype(float),
            "cum_pnl": cum_pnl.astype(float),
            "pnl_net": daily_pnl_net.astype(float),
            "cum_pnl_net": cum_pnl_net.astype(float),
            "trading_cost": trading_cost_daily.astype(float),
            "funding_cost": funding_cost_daily.astype(float),
        }
    )
    quantile_cols: dict[str, np.ndarray] = {}
    for q in range(n_quintiles):
        quantile_cols[f"q{q + 1}_pnl"] = quantile_daily[:, q].astype(float)
        quantile_cols[f"q{q + 1}_cum"] = quantile_cum[:, q].astype(float)
    quantile_df = pd.DataFrame(quantile_cols, index=base_curve_df.index)
    curve_df = pd.concat([base_curve_df, quantile_df], axis=1)
    metrics = {
        "sharpe": float(sharpe),
        "sharpe_after_cost": float(sharpe_net),
        "cagr": float(cagr),
        "cagr_after_cost": float(cagr_net),
        "returns": float(returns_annual),
        "returns_after_cost": float(returns_annual_net),
        "turnover": float(turnover),
        "win_rate": float(win_rate),
        "drawdown": float(max_drawdown),
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "ic_ir": float(ic_ir),
        "obs_count": int(len(daily_pnl)),
        "symbols": int(len(common_symbols)),
        "margin": float(returns_mean * 1000.0),
        "margin_after_cost": float(float(np.mean(daily_pnl_net)) * 1000.0),
        "single_side_fee": float(SINGLE_SIDE_FEE),
        "round_trip_fee": float(ROUND_TRIP_FEE),
        "avg_long_funding": float(avg_long_funding),
        "avg_short_funding": float(avg_short_funding),
        "funding_annual": float(funding_annual),
        "annual_cost_drag": float(returns_annual - returns_annual_net),
        "leverage": float(leverage),
        "long_group": int(long_group),
        "short_group": int(short_group),
    }
    return curve_df, metrics

def _load_index_html() -> str:
    from importlib.resources import files

    return (files("alpha101.research") / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/")
def home() -> HTMLResponse:
    return HTMLResponse(content=_load_index_html())


@app.post("/api/backtest")
def run_backtest(payload: BacktestRequest) -> dict[str, Any]:
    expression = payload.expression.strip()
    if not expression:
        raise HTTPException(status_code=400, detail="expression cannot be empty")
    if payload.long_group is not None and payload.long_group > payload.n_quintiles:
        raise HTTPException(status_code=400, detail="long_group cannot be greater than n_quintiles")
    if payload.short_group is not None and payload.short_group > payload.n_quintiles:
        raise HTTPException(status_code=400, detail="short_group cannot be greater than n_quintiles")

    long_group = payload.long_group if payload.long_group is not None else payload.n_quintiles
    short_group = payload.short_group if payload.short_group is not None else 1
    if long_group == short_group:
        raise HTTPException(status_code=400, detail="long_group and short_group must be different")

    ctx = get_context()
    cfg = ctx["cfg"]
    engine: FastExpressionEngine = ctx["engine"]
    wide_data: pd.DataFrame = ctx["wide_data"]
    n_bars: int = ctx["n_bars"]

    try:
        factor_wide = engine.evaluate(expression)
        factor_wide.index.name = "date"
        factor_wide.columns.name = "symbol"
        factor_df = process_factor_wide_format(factor_wide)
        factor_name = "alpha_test"
        factor_df.columns = pd.MultiIndex.from_product([[factor_name], factor_df.columns])

        panel_df = pd.concat([wide_data, factor_df], axis=1).iloc[n_bars:]
        curve_df, metrics = _compute_curve_and_metrics(
            panel_df=panel_df,
            alpha_name=factor_name,
            target_col="target",
            n_quintiles=payload.n_quintiles,
            long_group=long_group,
            short_group=short_group,
            leverage=payload.leverage,
            k_bars=cfg.trade_per_k_bars,
            freq=cfg.timeframe,
            factor_agg="ewma",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "success": True,
        "expression": expression,
        "config": {
            "timeframe": cfg.timeframe,
            "trade_per_k_bars": cfg.trade_per_k_bars,
            "test_start_date": str(cfg.test_start_date),
            "test_end_date": str(cfg.test_end_date),
            "leverage": float(payload.leverage),
            "long_group": int(long_group),
            "short_group": int(short_group),
        },
        "metrics": metrics,
        "curve": curve_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    uvicorn.run("alpha101.research.server:app", host="0.0.0.0", port=8001, reload=False)
