from __future__ import annotations

import threading
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from alpha101.config import get_config
from alpha101.data import FactorDataView, build_research_wide_frame
from alpha101.factors.backtesting import LongShortBacktestConfig, backtest_long_short
from alpha101.factors.operator_lib import process_factor_wide_format
from alpha101.factors.expression import FastExpressionEngine


app = FastAPI(title="Alpha101 Factor Research Server", version="1.0.0")

_context_lock = threading.Lock()
_context: dict[str, Any] | None = None
_factor_view_lock = threading.Lock()
_factor_view: dict[str, Any] | None = None
_factor_view_seq = 0


class BacktestRequest(BaseModel):
    expression: str = Field(..., min_length=1, max_length=5000)
    n_quintiles: int = Field(5, ge=2, le=50)
    leverage: float = Field(1.0, ge=1.0, le=20.0)
    long_group: int | None = Field(None, ge=1, le=50)
    short_group: int | None = Field(None, ge=1, le=50)


def _load_context() -> dict[str, Any]:
    cfg = get_config()
    wide_data = build_research_wide_frame(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        buffer=cfg.pre_buffer_candles,
    )
    engine = FastExpressionEngine(FactorDataView(wide_data))
    return {
        "cfg": cfg,
        "wide_data": wide_data,
        "engine": engine,
    }


def get_context() -> dict[str, Any]:
    global _context
    with _context_lock:
        if _context is None:
            _context = _load_context()
    return _context


def _json_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _json_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _build_factor_view(wide_data: pd.DataFrame, raw_factor: pd.DataFrame) -> dict[str, Any]:
    global _factor_view, _factor_view_seq
    symbols = [
        symbol
        for symbol in raw_factor.columns
        if all((field, symbol) in wide_data.columns for field in ("open", "high", "low", "close"))
    ]
    raw_factor = raw_factor.reindex(columns=symbols)
    ranks = raw_factor.rank(axis=1, method="min", ascending=False, na_option="keep")
    totals = raw_factor.notna().sum(axis=1)
    valid_dates = raw_factor.dropna(how="all").index
    latest_date = valid_dates[-1] if len(valid_dates) else raw_factor.index[-1]

    summary = []
    latest_raw = raw_factor.loc[latest_date]
    latest_rank = ranks.loc[latest_date]
    latest_total = int(totals.loc[latest_date])
    for symbol in symbols:
        summary.append(
            {
                "symbol": symbol,
                "date": str(latest_date),
                "raw_factor": _json_float(latest_raw.get(symbol)),
                "rank": _json_int(latest_rank.get(symbol)),
                "total": latest_total,
            }
        )
    summary.sort(key=lambda item: (item["rank"] is None, item["rank"] or 10**9, item["symbol"]))

    with _factor_view_lock:
        _factor_view_seq += 1
        run_id = _factor_view_seq
        view = {
            "run_id": run_id,
            "latest_date": str(latest_date),
            "symbols": summary,
            "wide_data": wide_data,
            "raw_factor": raw_factor,
            "ranks": ranks,
            "totals": totals,
        }
        _factor_view = view
    return {
        "run_id": run_id,
        "latest_date": str(latest_date),
        "symbols": summary,
    }


def _get_factor_view(run_id: int | None = None) -> dict[str, Any]:
    with _factor_view_lock:
        view = _factor_view
    if view is None:
        raise HTTPException(status_code=404, detail="no factor view is available; run a backtest first")
    if run_id is not None and int(run_id) != int(view["run_id"]):
        raise HTTPException(status_code=404, detail="factor view has expired; run the backtest again")
    return view


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

    try:
        factor_wide = engine.evaluate(expression)
        factor_wide.index.name = "date"
        factor_wide.columns.name = "symbol"
        factor_view = _build_factor_view(wide_data, factor_wide)
        factor_df = process_factor_wide_format(factor_wide)
        factor_name = "alpha_test"
        factor_df.columns = pd.MultiIndex.from_product([[factor_name], factor_df.columns])

        panel_df = pd.concat([wide_data, factor_df], axis=1)
        curve_df, metrics = backtest_long_short(
            panel_df=panel_df,
            factor_name=factor_name,
            target_col="target",
            config=LongShortBacktestConfig(
                n_quantiles=payload.n_quintiles,
                long_group=long_group,
                short_group=short_group,
                leverage=payload.leverage,
                k_bars=cfg.trade_per_k_bars,
                freq=cfg.timeframe,
                factor_agg="ewma",
                single_side_fee=cfg.single_side_fee,
                round_trip_fee=cfg.round_trip_fee,
            ),
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
            "single_side_fee": float(cfg.single_side_fee),
            "round_trip_fee": float(cfg.round_trip_fee),
        },
        "metrics": metrics,
        "curve": curve_df.to_dict(orient="records"),
        "factor_view": factor_view,
    }


@app.get("/api/factor-detail/{symbol}")
def factor_detail(symbol: str, run_id: int | None = None) -> dict[str, Any]:
    view = _get_factor_view(run_id)
    raw_factor: pd.DataFrame = view["raw_factor"]
    if symbol not in raw_factor.columns:
        raise HTTPException(status_code=404, detail=f"symbol not found: {symbol}")

    wide_data: pd.DataFrame = view["wide_data"]
    ranks: pd.DataFrame = view["ranks"]
    totals: pd.Series = view["totals"]
    frame = pd.DataFrame(
        {
            "open": wide_data[("open", symbol)].reindex(raw_factor.index),
            "high": wide_data[("high", symbol)].reindex(raw_factor.index),
            "low": wide_data[("low", symbol)].reindex(raw_factor.index),
            "close": wide_data[("close", symbol)].reindex(raw_factor.index),
            "raw_factor": raw_factor[symbol],
            "rank": ranks[symbol],
            "total": totals,
        }
    )
    frame = frame.dropna(how="all", subset=["open", "high", "low", "close", "raw_factor"])

    rows = []
    for dt, row in frame.iterrows():
        rows.append(
            {
                "date": str(dt),
                "open": _json_float(row["open"]),
                "high": _json_float(row["high"]),
                "low": _json_float(row["low"]),
                "close": _json_float(row["close"]),
                "raw_factor": _json_float(row["raw_factor"]),
                "rank": _json_int(row["rank"]),
                "total": _json_int(row["total"]),
            }
        )

    return {
        "run_id": int(view["run_id"]),
        "symbol": symbol,
        "latest_date": view["latest_date"],
        "rows": rows,
    }


if __name__ == "__main__":
    uvicorn.run("alpha101.research.server:app", host="0.0.0.0", port=8001, reload=False)
