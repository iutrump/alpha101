from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import sys
from pathlib import Path
sys.path.append(str((Path(__file__).parent.parent.parent).resolve()))
from alpha101.futures_ml.config import get_config
from alpha101.futures_ml.data import build_wide_df
from alpha101.futures_ml.alpha_sharpe import _nan_rowwise_corr, _sample_indices_after_agg
from alpha101.world_quant.Alpha101_code_1 import Alphas, process_factor_wide_format
from alpha101.world_quant.fastengine import FastExpressionEngine


app = FastAPI(title="WorldQuant FastEngine Backtest API", version="1.0.0")

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

    sampled_idx = _sample_indices_after_agg(len(panel_df), k_bars)
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

    ic_series = _nan_rowwise_corr(factor_eval_raw, target_eval_raw)
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


@app.get("/")
def home() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FastEngine Factor Backtest</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { --bg:#f4efe6; --panel:#fffaf1; --ink:#1f1b16; --soft:#6d6357; --accent:#be4f2e; --line:#d7c9b7; }
    * { box-sizing:border-box; } body { margin:0; font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:radial-gradient(circle at 20% 10%,#ffe9cb 0,transparent 30%),radial-gradient(circle at 80% 90%,#ffd9cd 0,transparent 35%),var(--bg); color:var(--ink); }
    .app { max-width:1400px; margin:24px auto; padding:0 8px; display:grid; grid-template-columns:320px minmax(0,1fr); gap:12px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px; box-shadow:0 10px 30px rgba(45,32,14,.08); }
    h1 { margin:0 0 12px; font-size:20px; } .muted { color:var(--soft); font-size:13px; }
    textarea { width:100%; min-height:108px; resize:vertical; border:1px solid var(--line); border-radius:10px; padding:12px; font-size:14px; background:#fff; color:var(--ink); }
    .row { margin-top:10px; display:flex; gap:10px; align-items:center; } input[type=number]{ width:72px; border:1px solid var(--line); border-radius:8px; padding:8px; }
    .control-row { display:grid; grid-template-columns:auto 72px auto 72px; column-gap:10px; row-gap:8px; align-items:center; }
    button { border:0; border-radius:10px; padding:10px 14px; background:var(--accent); color:#fff; font-weight:600; cursor:pointer; } button:disabled { opacity:.6; cursor:wait; }
    #runBtn { margin-left:auto; }
    #status { margin-top:10px; min-height:20px; font-size:13px; color:var(--soft); } #chart { width:100%; height:620px; }
    .metrics { margin-top:14px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:10px; padding:10px; } .metric .k { color:var(--soft); font-size:12px; } .metric .v { font-size:15px; font-weight:600; margin-top:4px; }
    @media (max-width:980px){ .app { grid-template-columns:1fr; } #chart { height:420px; } }
    @media (max-width:420px){ .control-row { grid-template-columns:auto 1fr; } }
  </style>
</head>
<body>
  <div class="app">
    <section class="panel">
      <h1>Factor Expression Backtest</h1>
      <div class="muted">Enter a factor expression and click Run to view the PnL curve and metrics.</div>
      <textarea id="expr">ts_alpha(close, low, 14)</textarea>
      <div class="row control-row">
        <label for="nq">Quintiles</label>
        <input id="nq" type="number" min="2" max="50" value="5" />
        <label for="leverage">Leverage</label>
        <input id="leverage" type="number" min="1" max="20" step="1" value="1" />
      </div>
      <div class="row control-row">
        <label for="longGroup">Long</label>
        <input id="longGroup" type="number" min="1" max="50" value="5" />
        <label for="shortGroup">Short</label>
        <input id="shortGroup" type="number" min="1" max="50" value="1" />
      </div>
      <div class="row">
        <button id="runBtn">Run</button>
      </div>
      <div id="status"></div>
      <div id="metrics" class="metrics"></div>
    </section>
    <section class="panel"><div id="chart"></div></section>
  </div>
  <script>
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const metricsEl = document.getElementById("metrics");
    const exprEl = document.getElementById("expr");
    const nqEl = document.getElementById("nq");
    const leverageEl = document.getElementById("leverage");
    const longGroupEl = document.getElementById("longGroup");
    const shortGroupEl = document.getElementById("shortGroup");
    const pct = v => `${(v * 100).toFixed(2)}%`;
    const num = v => Number(v).toFixed(4);
    function renderMetrics(m){ const items=[["Sharpe",num(m.sharpe)],["Sharpe (After Cost)",num(m.sharpe_after_cost)],["CAGR",pct(m.cagr)],["CAGR (After Cost)",pct(m.cagr_after_cost)],["Annual Returns",pct(m.returns)],["Annual Returns (After Cost)",pct(m.returns_after_cost)],["Annual Cost Drag",pct(m.annual_cost_drag)],["Funding (Annualized)",pct(m.funding_annual)],["Max Drawdown",pct(m.drawdown)],["Win Rate",pct(m.win_rate)],["Turnover",pct(m.turnover)],["Margin",num(m.margin)],["Margin (After Cost)",num(m.margin_after_cost)],["Single Fee",pct(m.single_side_fee)],["Round-Trip Fee",pct(m.round_trip_fee)],["Avg Long Funding",pct(m.avg_long_funding)],["Avg Short Funding",pct(m.avg_short_funding)],["IC Mean",num(m.ic_mean)],["IC IR",num(m.ic_ir)],["Observations",String(m.obs_count)],["Symbols",String(m.symbols)]];
      metricsEl.innerHTML=items.map(([k,v])=>`<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join(""); }
    function renderChart(curve, nQuintiles, longGroup, shortGroup){
      const x=curve.map(d=>d.date);
      const traces=[];
      for(let i=1;i<=nQuintiles;i++){
        traces.push({
          x,
          y:curve.map(d=>d[`q${i}_cum`]),
          mode:"lines",
          name:`Q${i}`,
          line:{width:1.6}
        });
      }
      traces.push({
        x,
        y:curve.map(d=>d.cum_pnl),
        mode:"lines",
        name:`Long-Short (Q${longGroup}-Q${shortGroup})`,
        line:{color:"#be4f2e",width:3}
      });
      traces.push({
        x,
        y:curve.map(d=>d.cum_pnl_net),
        mode:"lines",
        name:`Long-Short Net (Q${longGroup}-Q${shortGroup}, After Cost)`,
        line:{color:"#1d6e45",width:2.4,dash:"dot"}
      });
      Plotly.newPlot("chart",traces,{margin:{l:45,r:20,t:10,b:40},paper_bgcolor:"#fffaf1",plot_bgcolor:"#fff",xaxis:{gridcolor:"#efe4d8"},yaxis:{gridcolor:"#efe4d8",title:"Equity"}},{responsive:true,displaylogo:false});
    }
    function syncGroupInputs(){
      const nQuintiles = Number(nqEl.value) || 5;
      longGroupEl.max = String(nQuintiles);
      shortGroupEl.max = String(nQuintiles);
      if ((Number(longGroupEl.value) || 0) > nQuintiles) longGroupEl.value = String(nQuintiles);
      if ((Number(shortGroupEl.value) || 0) > nQuintiles) shortGroupEl.value = "1";
      if ((Number(longGroupEl.value) || 0) < 1) longGroupEl.value = String(nQuintiles);
      if ((Number(shortGroupEl.value) || 0) < 1) shortGroupEl.value = "1";
    }
    async function runBacktest(){
      syncGroupInputs();
      const expression=exprEl.value.trim();
      const n_quintiles=Number(nqEl.value)||5;
      const leverage=Number(leverageEl.value)||1;
      const long_group=Number(longGroupEl.value)||n_quintiles;
      const short_group=Number(shortGroupEl.value)||1;
      if(!expression){ statusEl.textContent="Please enter an expression"; return; }
      runBtn.disabled=true; statusEl.textContent="Running, please wait..."; metricsEl.innerHTML="";
      try { const resp=await fetch("/api/backtest",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({expression,n_quintiles,leverage,long_group,short_group})});
        const data=await resp.json(); if(!resp.ok){ throw new Error(data.detail || "Backtest failed"); } renderMetrics(data.metrics); renderChart(data.curve, n_quintiles, data.config.long_group, data.config.short_group);
        statusEl.textContent=`Done: ${data.config.test_start_date} ~ ${data.config.test_end_date}, ${data.curve.length} points`;
      } catch(err) { statusEl.textContent=`Failed: ${err.message}`; } finally { runBtn.disabled=false; }
    }
    nqEl.addEventListener("change", syncGroupInputs);
    syncGroupInputs();
    runBtn.addEventListener("click", runBacktest); runBacktest();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


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
    uvicorn.run("alpha101.world_quant.backtest_server:app", host="0.0.0.0", port=8001, reload=False)
