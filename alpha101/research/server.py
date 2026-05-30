from __future__ import annotations

import threading
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from alpha101.config import get_config
from alpha101.data.panel import build_wide_df
from alpha101.factors.alpha_data import Alphas
from alpha101.factors.backtest import LongShortBacktestConfig, backtest_long_short
from alpha101.factors.operators import process_factor_wide_format
from alpha101.factors.expression import FastExpressionEngine


app = FastAPI(title="Alpha101 Factor Research Server", version="1.0.0")

_context_lock = threading.Lock()
_context: dict[str, Any] | None = None


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
        },
        "metrics": metrics,
        "curve": curve_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    uvicorn.run("alpha101.research.server:app", host="0.0.0.0", port=8001, reload=False)
