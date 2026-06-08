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
    pnl_corr_threshold: float = 0.85,
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
    pnl_values: dict[str, pd.Series] = {}
    for candidate in candidates:
        expression = candidate["expression"]
        raw_factor = engine.evaluate(expression)
        factor = process_factor_wide_format(raw_factor).reindex(index=wide.index, columns=columns)
        if specific:
            factor = residualize_style_factor(factor, styles or {}, config=style_config)
        factor_values[expression] = factor
        pnl_values[expression] = factor_pnl_series(
            factor,
            target,
            n_quantiles=n_quantiles,
            transaction_cost=transaction_cost,
        )

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
    annotate_pnl_redundancy(records, pnl_values, pnl_corr_threshold)
    return records, corr_clusters


def wide_for_report_mode(wide: pd.DataFrame, manifest: dict[str, Any], report_mode: str) -> pd.DataFrame:
    if report_mode == "final":
        return wide
    ratios = tuple(float(value) for value in manifest.get("segment_ratios", (0.70, 0.15, 0.15)))
    valid_slice = _time_segment_slices(len(wide), ratios)["valid"]
    return wide.iloc[valid_slice]


def validate_expressions_on_validation(
    expressions: list[str],
    wide: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    time_folds: int,
    universe_folds: int,
    walk_forward_folds: int,
    n_quantiles: int,
    extra_n_quantiles: tuple[int, ...] | list[int] = (),
) -> dict[str, dict[str, Any]]:
    validation_wide = wide_for_report_mode(wide, manifest, "validation")
    close = validation_wide["close"]
    columns = list(close.columns)
    forward_periods = int(manifest.get("forward_periods", 1))
    transaction_cost = float(manifest.get("transaction_cost", 0.001))
    target = forward_returns(close, periods=forward_periods)
    engine = FastExpressionEngine(FactorDataView(validation_wide))
    time_splits = time_split_indexes(validation_wide.index, time_folds)
    universe_splits = universe_split_columns(columns, universe_folds)
    walk_forward_splits = walk_forward_split_indexes(validation_wide.index, walk_forward_folds)
    quantiles = [int(n_quantiles)]
    for value in extra_n_quantiles:
        value = int(value)
        if value not in quantiles:
            quantiles.append(value)

    results: dict[str, dict[str, Any]] = {}
    for expression in expressions:
        raw_factor = engine.evaluate(expression)
        factor = process_factor_wide_format(raw_factor).reindex(index=validation_wide.index, columns=columns)
        metrics: dict[str, Any] = {}
        for quantile in quantiles:
            prefix = "cv" if quantile == int(n_quantiles) else f"cv_nq{quantile}"
            metrics.update(
                _robustness_metrics_for_quantile(
                    factor,
                    target,
                    time_splits=time_splits,
                    universe_splits=universe_splits,
                    walk_forward_splits=walk_forward_splits,
                    n_quantiles=quantile,
                    transaction_cost=transaction_cost,
                    prefix=prefix,
                )
            )
        results[expression] = metrics
    return results


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


def walk_forward_split_indexes(index: pd.Index, n_folds: int) -> list[tuple[str, pd.Index]]:
    arrays = np.array_split(np.arange(len(index)), max(1, n_folds) + 1)
    holdouts = arrays[1:] if len(arrays) > 1 else arrays
    return [(f"WF{i}", index[array]) for i, array in enumerate(holdouts, start=1) if len(array)]


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


def _robustness_metrics_for_quantile(
    factor: pd.DataFrame,
    target: pd.DataFrame,
    *,
    time_splits: list[tuple[str, pd.Index]],
    universe_splits: list[tuple[str, list[str]]],
    walk_forward_splits: list[tuple[str, pd.Index]],
    n_quantiles: int,
    transaction_cost: float,
    prefix: str,
) -> dict[str, Any]:
    def score(*, dates=None, cols=None) -> dict[str, Any]:
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

    time_metrics = [
        {"fold": label, "start": str(dates[0]), "end": str(dates[-1]), **score(dates=dates)}
        for label, dates in time_splits
    ]
    universe_metrics = [
        {"group": label, "n_symbols": len(cols), **score(cols=cols)}
        for label, cols in universe_splits
    ]
    walk_forward_metrics = [
        {"fold": label, "start": str(dates[0]), "end": str(dates[-1]), **score(dates=dates)}
        for label, dates in walk_forward_splits
    ]
    time_sharpes = np.asarray([item["sharpe_ratio"] for item in time_metrics], dtype=float)
    universe_sharpes = np.asarray([item["sharpe_ratio"] for item in universe_metrics], dtype=float)
    walk_forward_sharpes = np.asarray([item["sharpe_ratio"] for item in walk_forward_metrics], dtype=float)
    metrics = {
        f"{prefix}_time_pos_folds": int(np.sum(time_sharpes > 0)),
        f"{prefix}_time_median_sharpe": _safe_nanmedian(time_sharpes),
        f"{prefix}_time_min_sharpe": _safe_nanmin(time_sharpes),
        f"{prefix}_universe_pos_groups": int(np.sum(universe_sharpes > 0)),
        f"{prefix}_universe_median_sharpe": _safe_nanmedian(universe_sharpes),
        f"{prefix}_universe_min_sharpe": _safe_nanmin(universe_sharpes),
        f"{prefix}_walk_forward_pos_folds": int(np.sum(walk_forward_sharpes > 0)),
        f"{prefix}_walk_forward_median_sharpe": _safe_nanmedian(walk_forward_sharpes),
        f"{prefix}_walk_forward_min_sharpe": _safe_nanmin(walk_forward_sharpes),
        f"{prefix}_time_metrics": time_metrics,
        f"{prefix}_universe_metrics": universe_metrics,
        f"{prefix}_walk_forward_metrics": walk_forward_metrics,
    }
    metrics[f"{prefix}_pass"] = _robustness_pass(
        time_sharpes,
        universe_sharpes,
        walk_forward_sharpes,
    )
    return metrics


def _robustness_pass(
    time_sharpes: np.ndarray,
    universe_sharpes: np.ndarray,
    walk_forward_sharpes: np.ndarray,
) -> bool:
    time_required = max(1, int(np.ceil(len(time_sharpes) * 0.5)))
    universe_required = max(1, len(universe_sharpes) - 1)
    walk_forward_required = max(1, int(np.ceil(len(walk_forward_sharpes) * 0.5)))
    return bool(
        np.sum(time_sharpes > 0) >= time_required
        and np.sum(universe_sharpes > 0) >= universe_required
        and np.sum(walk_forward_sharpes > 0) >= walk_forward_required
        and _safe_nanmedian(time_sharpes) > 0
        and _safe_nanmedian(walk_forward_sharpes) > 0
    )


def _safe_nanmedian(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.nanmedian(finite)) if finite.size else float("nan")


def _safe_nanmin(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.nanmin(finite)) if finite.size else float("nan")


def factor_pnl_series(
    factor: pd.DataFrame,
    target: pd.DataFrame,
    *,
    n_quantiles: int,
    transaction_cost: float,
) -> pd.Series:
    factor = factor.reindex_like(target)
    valid = np.isfinite(factor.to_numpy(dtype=float, copy=False)) & np.isfinite(target.to_numpy(dtype=float, copy=False))
    factor_values = factor.to_numpy(dtype=float, copy=False)
    target_values = target.to_numpy(dtype=float, copy=False)
    pnl = np.full(factor.shape[0], np.nan, dtype=float)
    prev_long_idx: np.ndarray | None = None
    prev_short_idx: np.ndarray | None = None

    for date_idx in range(factor.shape[0]):
        mask = valid[date_idx]
        n_valid = int(mask.sum())
        if n_valid < n_quantiles:
            continue

        row_factor = factor_values[date_idx]
        row_target = target_values[date_idx]
        valid_factor = row_factor[mask]
        valid_target = row_target[mask]
        valid_idx = np.flatnonzero(mask)
        base_group_size, remainder = divmod(n_valid, n_quantiles)
        short_count = base_group_size + (1 if remainder else 0)
        long_count = base_group_size
        short_idx = np.argpartition(valid_factor, short_count - 1)[:short_count]
        long_start = n_valid - long_count
        long_idx = np.argpartition(valid_factor, long_start)[long_start:]
        long_global_idx = valid_idx[long_idx]
        short_global_idx = valid_idx[short_idx]

        gross = float(valid_target[long_idx].mean() - valid_target[short_idx].mean())
        if prev_long_idx is None or prev_short_idx is None:
            turnover = 0.0
        else:
            long_turnover = _turnover_ratio(long_global_idx, prev_long_idx)
            short_turnover = _turnover_ratio(short_global_idx, prev_short_idx)
            turnover = float((long_turnover + short_turnover) * 0.5)
        pnl[date_idx] = gross - turnover * float(transaction_cost)
        prev_long_idx = long_global_idx
        prev_short_idx = short_global_idx

    return pd.Series(pnl, index=factor.index)


def annotate_pnl_redundancy(
    records: list[dict[str, Any]],
    pnl_values: dict[str, pd.Series],
    threshold: float,
) -> None:
    kept: list[dict[str, Any]] = []
    for record in sorted(records, key=validation_sort_key):
        expression = record["expression"]
        pnl = pnl_values[expression]
        max_corr = 0.0
        nearest_expression = ""
        for kept_record in kept:
            kept_expression = kept_record["expression"]
            corr = _series_corr(pnl, pnl_values[kept_expression])
            if abs(corr) > abs(max_corr):
                max_corr = corr
                nearest_expression = kept_expression
        record["max_pnl_corr"] = float(max_corr)
        record["nearest_pnl_corr_expression"] = nearest_expression
        record["redundant_by_pnl"] = bool(nearest_expression and abs(max_corr) >= threshold)
        if not record["redundant_by_pnl"]:
            kept.append(record)


def validation_sort_key(record: dict[str, Any]) -> tuple[int, float]:
    order = {"accepted_candidate": 0, "watchlist": 1, "rejected": 2}
    return order.get(record.get("decision", "rejected"), 9), -float(record.get("time_median_sharpe", 0.0))


def _series_corr(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return float(corr) if np.isfinite(corr) else 0.0


def _turnover_ratio(current_idx: np.ndarray, prev_idx: np.ndarray) -> float:
    if current_idx.size == 0:
        return 0.0
    changed = np.setdiff1d(current_idx, prev_idx, assume_unique=False).size
    return float(changed / current_idx.size)
