from __future__ import annotations

import threading
import os
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from alpha101.config import get_config
from alpha101.data import (
    FactorDataView,
    build_ashare_wide_frame_from_csv,
    build_external_factor_wide_frame,
    build_research_wide_frame,
)
from alpha101.factors.backtesting import LongShortBacktestConfig, backtest_long_short
from alpha101.factors.evaluation.external import (
    build_external_factor_curve,
    external_factor_names,
    score_external_factors,
    select_external_factors_by_pnl_similarity,
)
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


class ExternalBacktestRequest(BaseModel):
    n_quintiles: int = Field(5, ge=2, le=50)
    transaction_cost: float = Field(0.001, ge=0.0, le=0.1)
    direction: int | None = None


class ExternalSelectionRequest(BaseModel):
    n_quintiles: int = Field(5, ge=2, le=50)
    transaction_cost: float = Field(0.001, ge=0.0, le=0.1)
    max_factors: int = Field(20, ge=1, le=200)
    pnl_corr_threshold: float = Field(0.75, ge=0.0, le=0.999)
    pnl_corr_method: str = Field("pearson", pattern="^(pearson|spearman)$")
    score_column: str = Field("selected_excess_returns", min_length=1, max_length=100)


def _load_context() -> dict[str, Any]:
    cfg = get_config()
    ashare_csv = os.getenv("ALPHA101_ASHARE_CSV")
    external_csv = os.getenv("ALPHA101_EXTERNAL_FACTOR_CSV") or os.getenv("ALPHA101_EXTERNAL_FACTOR_GLOB")
    if ashare_csv:
        wide_data = build_ashare_wide_frame_from_csv(ashare_csv)
    elif external_csv:
        wide_data = build_external_factor_wide_frame(external_csv)
    else:
        wide_data = build_research_wide_frame(
            cfg.pairs,
            cfg.lookback_days,
            cfg.data_root,
            cfg.timeframe,
            test_start_date=cfg.test_start_date,
            test_end_date=cfg.test_end_date,
            buffer=cfg.pre_buffer_candles,
        )
    engine = FastExpressionEngine(FactorDataView(wide_data)) if _can_build_expression_engine(wide_data) else None
    external_wide = build_external_factor_wide_frame(external_csv) if external_csv else None
    return {
        "cfg": cfg,
        "wide_data": wide_data,
        "engine": engine,
        "external_wide": external_wide,
    }


def _can_build_expression_engine(wide_data: pd.DataFrame) -> bool:
    fields = set(wide_data.columns.get_level_values(0))
    return {"open", "high", "low", "close", "volume"}.issubset(fields)


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


def _json_safe_value(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (pd.isna(value) or value in (float("inf"), float("-inf"))):
        return None
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append({str(key): _json_safe_value(value) for key, value in record.items()})
    return rows


def _build_factor_view(wide_data: pd.DataFrame, raw_factor: pd.DataFrame, *, direction: int = 1) -> dict[str, Any]:
    global _factor_view, _factor_view_seq
    symbols = [
        symbol
        for symbol in raw_factor.columns
        if all((field, symbol) in wide_data.columns for field in ("open", "high", "low", "close"))
    ]
    if not symbols:
        symbols = list(raw_factor.columns)
    raw_factor = raw_factor.reindex(columns=symbols)
    ranks = raw_factor.rank(axis=1, method="min", ascending=int(direction) < 0, na_option="keep")
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
            "direction": int(1 if int(direction) >= 0 else -1),
            "rank_meaning": "highest factor values are selected" if int(direction) >= 0 else "lowest factor values are selected",
        }
        _factor_view = view
    return {
        "run_id": run_id,
        "latest_date": str(latest_date),
        "symbols": summary,
        "direction": int(1 if int(direction) >= 0 else -1),
        "rank_meaning": "highest factor values are selected" if int(direction) >= 0 else "lowest factor values are selected",
    }


def _get_factor_view(run_id: int | None = None) -> dict[str, Any]:
    with _factor_view_lock:
        view = _factor_view
    if view is None:
        raise HTTPException(status_code=404, detail="no factor view is available; run a backtest first")
    if run_id is not None and int(run_id) != int(view["run_id"]):
        raise HTTPException(status_code=404, detail="factor view has expired; run the backtest again")
    return view


def _get_external_wide(ctx: dict[str, Any]) -> pd.DataFrame:
    external_wide = ctx.get("external_wide")
    if external_wide is None:
        raise HTTPException(status_code=404, detail="no external factor data is configured")
    return external_wide


def _get_external_summary(ctx: dict[str, Any]) -> pd.DataFrame:
    summary = ctx.get("external_summary")
    if summary is None:
        external_wide = _get_external_wide(ctx)
        summary = score_external_factors(
            external_wide,
            n_quantiles=_default_external_quantiles(external_wide),
            min_segment_obs=1,
            directions=_external_directions(ctx),
        )
        summary = _merge_accepted_metadata(summary, ctx)
        ctx["external_summary"] = summary
    return summary


def _accepted_summary_path() -> Path | None:
    raw_path = os.getenv("ALPHA101_ACCEPTED_SUMMARY_CSV")
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.exists() else None


def _get_accepted_summary(ctx: dict[str, Any]) -> pd.DataFrame:
    accepted = ctx.get("accepted_summary")
    if accepted is not None:
        return accepted
    path = _accepted_summary_path()
    if path is None:
        frame = pd.DataFrame()
    else:
        frame = pd.read_csv(path)
        if "factor_id" in frame.columns:
            frame = frame.rename(columns={"factor_id": "factor"})
        keep = [
            column
            for column in [
                "factor",
                "rank_ic",
                "rank_icir",
                "max_corr",
                "ic",
                "icir",
                "complexity",
                "expression",
            ]
            if column in frame.columns
        ]
        frame = frame[keep].drop_duplicates("factor", keep="last") if keep else pd.DataFrame()
    ctx["accepted_summary"] = frame
    return frame


def _external_directions(ctx: dict[str, Any]) -> dict[str, int]:
    accepted = _get_accepted_summary(ctx)
    if accepted.empty or "rank_icir" not in accepted.columns:
        return {}
    directions: dict[str, int] = {}
    for _, row in accepted.iterrows():
        value = pd.to_numeric(row.get("rank_icir"), errors="coerce")
        if pd.notna(value):
            directions[str(row["factor"])] = -1 if float(value) < 0 else 1
    return directions


def _external_direction_for_factor(ctx: dict[str, Any], factor_name: str) -> int | None:
    directions = _external_directions(ctx)
    if factor_name in directions:
        return int(directions[factor_name])
    summary = ctx.get("external_summary")
    if isinstance(summary, pd.DataFrame) and not summary.empty and "direction" in summary.columns:
        match = summary[summary["factor"].astype(str) == str(factor_name)]
        if not match.empty and pd.notna(match.iloc[0].get("direction")):
            return int(match.iloc[0]["direction"])
    return None


def _merge_accepted_metadata(summary: pd.DataFrame, ctx: dict[str, Any]) -> pd.DataFrame:
    accepted = _get_accepted_summary(ctx)
    if summary.empty or accepted.empty or "factor" not in accepted.columns:
        return summary
    merged = summary.merge(accepted, on="factor", how="left", suffixes=("", "_mining"))
    if "rank_icir" in merged.columns:
        merged["direction_source"] = merged["rank_icir"].apply(
            lambda value: "accepted_summary_rank_icir" if pd.notna(value) else "research_ic_ir"
        )
    return merged


def _default_external_quantiles(external_wide: pd.DataFrame) -> int:
    target_symbols = len(external_wide["target"].columns) if "target" in external_wide.columns.get_level_values(0) else 5
    return max(2, min(5, int(target_symbols)))


def _external_metrics_for_response(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    sharpe = out.get("sharpe_ratio", out.get("sharpe_after_cost", 0.0))
    returns = out.get("returns", out.get("returns_after_cost", 0.0))
    out.setdefault("sharpe", sharpe)
    out.setdefault("sharpe_after_cost", sharpe)
    out.setdefault("sharpe_before_cost", out.get("sharpe_before_cost", sharpe))
    out.setdefault("cagr", returns)
    out.setdefault("cagr_after_cost", out.get("returns_after_cost", returns))
    out.setdefault("returns_after_cost", returns)
    out.setdefault("annual_cost_drag", out.get("returns_before_cost", returns) - out.get("returns_after_cost", returns))
    out.setdefault("funding_annual", 0.0)
    out.setdefault("margin", returns)
    out.setdefault("margin_after_cost", out.get("returns_after_cost", returns))
    out.setdefault("single_side_fee", out.get("transaction_cost", 0.0))
    out.setdefault("round_trip_fee", out.get("transaction_cost", 0.0))
    out.setdefault("avg_long_funding", 0.0)
    out.setdefault("avg_short_funding", 0.0)
    out.setdefault("symbols", 0)
    return {str(key): _json_safe_value(value) for key, value in out.items()}


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
    engine: FastExpressionEngine | None = ctx["engine"]
    wide_data: pd.DataFrame = ctx["wide_data"]
    if engine is None:
        raise HTTPException(status_code=400, detail="expression backtest is unavailable without OHLCV data")

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


@app.get("/api/external-factors")
def list_external_factors() -> dict[str, Any]:
    ctx = get_context()
    external_wide = _get_external_wide(ctx)
    summary = _get_external_summary(ctx)
    if summary.empty:
        factors = [{"factor": name} for name in external_factor_names(external_wide)]
    else:
        factors = _records(summary)
    return {"success": True, "factors": factors}


@app.post("/api/external-factor-selection")
def run_external_factor_selection(payload: ExternalSelectionRequest) -> dict[str, Any]:
    ctx = get_context()
    external_wide = _get_external_wide(ctx)
    n_quantiles = _effective_external_quantiles(external_wide, payload.n_quintiles)
    try:
        result = select_external_factors_by_pnl_similarity(
            external_wide,
            _get_external_summary(ctx),
            n_quantiles=n_quantiles,
            transaction_cost=payload.transaction_cost,
            max_factors=payload.max_factors,
            pnl_corr_threshold=payload.pnl_corr_threshold,
            pnl_corr_method=payload.pnl_corr_method,
            score_column=payload.score_column,
            directions=_external_directions(ctx),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "success": True,
        "config": {
            "mode": "external_precomputed_selection",
            "n_quintiles": int(n_quantiles),
            "transaction_cost": float(payload.transaction_cost),
            "max_factors": int(payload.max_factors),
            "pnl_corr_threshold": float(payload.pnl_corr_threshold),
            "pnl_corr_method": str(payload.pnl_corr_method),
            "score_column": str(payload.score_column),
        },
        "selected": _records(result["selected"]),
        "rejected": _records(result["rejected"]),
        "factors": _records(result["factors"]),
        "curve": _records(result["curve"]),
        "stats": {str(key): _json_safe_value(value) for key, value in result["stats"].items()},
        "similarity_pairs": [
            {str(key): _json_safe_value(value) for key, value in item.items()}
            for item in result["similarity_pairs"]
        ],
    }


@app.post("/api/external-factor-backtest/{factor_name}")
def run_external_factor_backtest(factor_name: str, payload: ExternalBacktestRequest) -> dict[str, Any]:
    ctx = get_context()
    external_wide = _get_external_wide(ctx)
    factor_names = external_factor_names(external_wide)
    if factor_name not in factor_names:
        raise HTTPException(status_code=404, detail=f"external factor not found: {factor_name}")
    n_quantiles = _effective_external_quantiles(external_wide, payload.n_quintiles)

    try:
        direction = payload.direction if payload.direction is not None else _external_direction_for_factor(ctx, factor_name)
        directions = {factor_name: direction} if direction is not None else None
        summary = score_external_factors(
            external_wide,
            factor_names=[factor_name],
            n_quantiles=n_quantiles,
            min_segment_obs=1,
            transaction_cost=payload.transaction_cost,
            directions=directions,
        )
        summary = _merge_accepted_metadata(summary, ctx)
        metrics = summary.iloc[0].to_dict()
        direction = int(metrics.get("direction", direction if direction is not None else 1))
        curve_df = build_external_factor_curve(
            external_wide,
            factor_name,
            n_quantiles=n_quantiles,
            transaction_cost=payload.transaction_cost,
            direction=direction,
        )
        raw_factor = external_wide[factor_name]
        wide_data = ctx.get("wide_data", external_wide)
        factor_view = _build_factor_view(wide_data, raw_factor, direction=direction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "success": True,
        "factor": factor_name,
        "config": {
            "mode": "external_precomputed",
            "n_quintiles": int(n_quantiles),
            "transaction_cost": float(payload.transaction_cost),
            "direction": int(1 if direction >= 0 else -1),
            "buy_group": int(n_quantiles if direction >= 0 else 1),
        },
        "metrics": _external_metrics_for_response(metrics),
        "curve": _records(curve_df),
        "factor_view": factor_view,
    }


def _effective_external_quantiles(external_wide: pd.DataFrame, requested: int) -> int:
    target_symbols = len(external_wide["target"].columns)
    return max(2, min(int(requested), int(target_symbols)))


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
            "open": _optional_symbol_series(wide_data, "open", symbol, raw_factor.index),
            "high": _optional_symbol_series(wide_data, "high", symbol, raw_factor.index),
            "low": _optional_symbol_series(wide_data, "low", symbol, raw_factor.index),
            "close": _optional_symbol_series(wide_data, "close", symbol, raw_factor.index),
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


def _optional_symbol_series(
    wide_data: pd.DataFrame,
    field: str,
    symbol: str,
    index: pd.Index,
) -> pd.Series:
    if (field, symbol) in wide_data.columns:
        return wide_data[(field, symbol)].reindex(index)
    return pd.Series(pd.NA, index=index, dtype="Float64")


if __name__ == "__main__":
    uvicorn.run("alpha101.research.server:app", host="0.0.0.0", port=8001, reload=False)
