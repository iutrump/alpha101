from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from alpha101.data import FactorDataView
from alpha101.factors.evaluation import StyleConfig, build_style_factors, residualize_style_factor
from alpha101.factors.evaluation.scoring import _score_factor_segment, _time_segment_slices, forward_returns
from alpha101.factors.expression import FastExpressionEngine
from alpha101.factors.operator_lib import process_factor_wide_format


def cross_validate_candidates(
    candidates: list[dict[str, Any]],
    wide: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    time_folds: int,
    universe_folds: int,
    corr_threshold: float,
    report_mode: str = "validation",
    specific: bool = False,
    style_config: StyleConfig | None = None,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    if report_mode not in {"validation", "final"}:
        raise ValueError("report_mode must be 'validation' or 'final'")
    close = wide["close"]
    columns = list(close.columns)
    n_quantiles = int(manifest.get("n_quantiles", 5))
    forward_periods = int(manifest.get("forward_periods", 1))
    transaction_cost = float(manifest.get("transaction_cost", 0.001))

    engine = FastExpressionEngine(FactorDataView(wide))
    target = forward_returns(close, periods=forward_periods)
    style_config = style_config or StyleConfig()
    styles = None
    if specific:
        view = FactorDataView(wide)
        styles = build_style_factors(
            view.close.reindex(columns=columns),
            view.cap.reindex(columns=columns) if view.cap is not None else None,
            view.market_return.reindex(columns=columns),
            view.volume.reindex(columns=columns),
            view.funding.reindex(columns=columns) if view.funding is not None else None,
            momentum_window=style_config.momentum_window,
            volatility_window=style_config.volatility_window,
            beta_window=style_config.beta_window,
            liquidity_window=style_config.liquidity_window,
            reversal_window=style_config.reversal_window,
            funding_window=style_config.funding_window,
        )
    time_splits = time_split_indexes(wide.index, time_folds)
    universe_splits = universe_split_columns(columns, universe_folds)

    def score(factor: pd.DataFrame, *, dates=None, cols=None) -> dict[str, Any]:
        factor_slice = factor if dates is None else factor.loc[dates]
        target_slice = target if dates is None else target.loc[dates]
        if cols is not None:
            factor_slice = factor_slice[cols]
            target_slice = target_slice[cols]
        return _score_factor_segment(
            factor_slice.to_numpy(dtype=float, copy=False),
            target_slice.to_numpy(dtype=float, copy=False),
            n_quantiles=n_quantiles,
            transaction_cost=transaction_cost,
        )

    records: list[dict[str, Any]] = []
    factor_values: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        expression = candidate["expression"]
        raw_factor = engine.evaluate(expression)
        factor = process_factor_wide_format(raw_factor).reindex(index=wide.index, columns=columns)
        if specific:
            factor = residualize_style_factor(factor, styles or {}, config=style_config)
        factor_values[expression] = factor

        full_metrics = score(factor)
        time_metrics = [
            {"fold": label, "start": str(dates[0]), "end": str(dates[-1]), **score(factor, dates=dates)}
            for label, dates in time_splits
        ]
        universe_metrics = [
            {"group": label, "n_symbols": len(cols), **score(factor, cols=cols)}
            for label, cols in universe_splits
        ]

        time_sharpes = np.asarray([m["sharpe_ratio"] for m in time_metrics], dtype=float)
        universe_sharpes = np.asarray([m["sharpe_ratio"] for m in universe_metrics], dtype=float)
        record = {
            "decision": "",
            "name": candidate["factor_name"],
            "expression": expression,
            "mode": "specific" if specific else "raw",
            "report_mode": report_mode,
            "summary_train_sharpe": candidate["train_sharpe"],
            "summary_valid_sharpe": candidate["valid_sharpe"],
            "summary_test_sharpe": candidate["test_sharpe"] if report_mode == "final" else np.nan,
            "summary_test_ic_ir": candidate["test_ic_ir"] if report_mode == "final" else np.nan,
            "summary_complexity": candidate["complexity_score"],
            "full_sharpe": full_metrics["sharpe_ratio"],
            "full_returns": full_metrics["returns"],
            "full_ic_ir": full_metrics["ic_ir"],
            "full_turnover": full_metrics["turnover"],
            "full_drawdown": full_metrics["drawdown"],
            "time_pos_folds": int(np.sum(time_sharpes > 0)),
            "time_gt1_folds": int(np.sum(time_sharpes > 1)),
            "time_min_sharpe": float(np.nanmin(time_sharpes)),
            "time_median_sharpe": float(np.nanmedian(time_sharpes)),
            "universe_pos_groups": int(np.sum(universe_sharpes > 0)),
            "universe_min_sharpe": float(np.nanmin(universe_sharpes)),
            "universe_median_sharpe": float(np.nanmedian(universe_sharpes)),
            "time_sharpes": [float(x) for x in time_sharpes],
            "universe_sharpes": [float(x) for x in universe_sharpes],
            "time_metrics": time_metrics,
            "universe_metrics": universe_metrics,
        }
        record["decision"] = validation_decision(record, universe_folds)
        records.append(record)

    corr_clusters = correlation_clusters(factor_values, corr_threshold)
    return records, corr_clusters


def wide_for_report_mode(wide: pd.DataFrame, manifest: dict[str, Any], report_mode: str) -> pd.DataFrame:
    if report_mode == "final":
        return wide
    ratios = tuple(float(value) for value in manifest.get("segment_ratios", (0.70, 0.15, 0.15)))
    valid_slice = _time_segment_slices(len(wide), ratios)["valid"]
    return wide.iloc[valid_slice]


def validation_decision(record: dict[str, Any], universe_folds: int) -> str:
    valid_ok = np.isnan(record["summary_valid_sharpe"]) or record["summary_valid_sharpe"] >= 1.0
    if (
        valid_ok
        and record["time_pos_folds"] >= 5
        and record["universe_pos_groups"] == universe_folds
        and record["time_median_sharpe"] > 1.0
        and record["universe_min_sharpe"] > 0
        and record["full_sharpe"] >= 2.0
    ):
        return "accepted_candidate"
    if (
        valid_ok
        and record["time_pos_folds"] >= 4
        and record["universe_pos_groups"] >= max(1, universe_folds - 1)
        and record["time_median_sharpe"] > 0
    ):
        return "watchlist"
    return "rejected"


def time_split_indexes(index: pd.Index, n_folds: int) -> list[tuple[str, pd.Index]]:
    arrays = np.array_split(np.arange(len(index)), n_folds)
    return [(f"T{i}", index[array]) for i, array in enumerate(arrays, start=1) if len(array)]


def universe_split_columns(columns: list[str], n_folds: int) -> list[tuple[str, list[str]]]:
    groups = {i: [] for i in range(n_folds)}
    for column in columns:
        digest = hashlib.md5(str(column).encode("utf-8")).hexdigest()
        groups[int(digest, 16) % n_folds].append(column)
    return [(f"U{i + 1}", groups[i]) for i in range(n_folds) if groups[i]]


def correlation_clusters(
    factor_values: dict[str, pd.DataFrame],
    threshold: float,
) -> list[list[str]]:
    flattened = {}
    for expression, factor in factor_values.items():
        ranked = factor.rank(axis=1, pct=True)
        flattened[expression] = pd.Series(ranked.to_numpy().ravel())
    corr = pd.DataFrame(flattened).corr(method="pearson")

    clusters: list[list[str]] = []
    seen: set[str] = set()
    for expression in factor_values:
        if expression in seen:
            continue
        members = [
            other
            for other in factor_values
            if other not in seen and abs(float(corr.loc[expression, other])) >= threshold
        ]
        seen.update(members)
        clusters.append(members)
    return clusters
