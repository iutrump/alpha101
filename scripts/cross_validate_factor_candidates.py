from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha101.config import get_config
from alpha101.data import FactorDataView, build_research_wide_frame
from alpha101.factors.evaluation import StyleConfig, build_style_factors, residualize_style_factor
from alpha101.factors.evaluation.scoring import _score_factor_segment, _time_segment_slices, forward_returns
from alpha101.factors.expression import FastExpressionEngine
from alpha101.factors.operator_lib import process_factor_wide_format


NUMERIC_COLUMNS = (
    "fitness",
    "test_fitness",
    "test_sharpe",
    "test_returns",
    "test_ic_ir",
    "valid_sharpe",
    "valid_returns",
    "valid_ic_ir",
    "train_sharpe",
    "train_returns",
    "train_ic_ir",
    "turnover",
    "drawdown",
    "complexity_score",
    "expr_max_depth",
    "stability_score",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate mined factor candidates.")
    parser.add_argument("summary_csv", type=Path, nargs="?", help="Path to a factor search summary.csv")
    parser.add_argument("--config", type=Path, default=Path("configs/alpha101.json"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--time-folds", type=int, default=6)
    parser.add_argument("--universe-folds", type=int, default=3)
    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--timeframe", type=str, default=None, help="Override config timeframe, e.g. 1h or 4h.")
    parser.add_argument(
        "--report-mode",
        choices=["validation", "final"],
        default="validation",
        help="validation excludes held-out test data; final may report test metrics for frozen candidates.",
    )
    parser.add_argument(
        "--specific",
        action="store_true",
        help="Validate style-neutral residual factors instead of raw factors.",
    )
    parser.add_argument("--momentum-window", type=int, default=20)
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--beta-window", type=int, default=60)
    parser.add_argument("--liquidity-window", type=int, default=20)
    parser.add_argument("--reversal-window", type=int, default=5)
    parser.add_argument("--funding-window", type=int, default=3)
    parser.add_argument("--style-min-count", type=int, default=8)
    parser.add_argument(
        "--forward-periods",
        type=int,
        action="append",
        default=None,
        help="Override forecast horizon in bars. Can be passed multiple times.",
    )
    parser.add_argument(
        "--expression",
        action="append",
        default=None,
        help="Manual expression to validate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--expressions-file",
        type=Path,
        default=None,
        help="CSV or text file of manual expressions. CSV columns: name,expression. Text: one expression per line.",
    )
    args = parser.parse_args()

    summary_path = args.summary_csv
    if summary_path is None and not args.expression and args.expressions_file is None:
        raise SystemExit("Provide summary_csv, --expression, or --expressions-file")
    manifest_path = args.manifest or (summary_path.with_name("manifest.json") if summary_path else None)
    out_dir = args.out_dir or (summary_path.parent if summary_path else Path("factor_validation_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path) if manifest_path else {}
    candidates: list[dict[str, Any]] = []
    if summary_path is not None:
        rows = _load_success_rows(summary_path)
        candidates.extend(_select_candidates(rows, max_candidates=args.max_candidates))
    candidates.extend(_load_manual_candidates(args.expression or [], args.expressions_file))
    candidates = _dedupe_candidates(candidates)

    cfg = get_config(args.config)
    if args.timeframe is not None:
        cfg.timeframe = args.timeframe
    wide = build_research_wide_frame(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        buffer=cfg.pre_buffer_candles,
    )
    if "date_end" in manifest:
        run_end = pd.to_datetime(manifest["date_end"], utc=True)
        wide = wide.loc[wide.index <= run_end]

    forward_periods_values = args.forward_periods or [int(manifest.get("forward_periods", 1))]
    for forward_periods in forward_periods_values:
        run_manifest = {**manifest, "forward_periods": int(forward_periods), "report_mode": args.report_mode}
        eval_wide = _wide_for_report_mode(wide, run_manifest, args.report_mode)
        records, corr_clusters = cross_validate(
            candidates,
            eval_wide,
            manifest=run_manifest,
            time_folds=args.time_folds,
            universe_folds=args.universe_folds,
            corr_threshold=args.corr_threshold,
            report_mode=args.report_mode,
            specific=args.specific,
            style_config=StyleConfig(
                momentum_window=args.momentum_window,
                volatility_window=args.volatility_window,
                beta_window=args.beta_window,
                liquidity_window=args.liquidity_window,
                reversal_window=args.reversal_window,
                funding_window=args.funding_window,
                min_count=args.style_min_count,
            ),
        )

        suffix_parts = []
        if len(forward_periods_values) > 1:
            suffix_parts.append(f"fp{forward_periods}")
        if args.specific:
            suffix_parts.append("specific")
        if args.report_mode == "final":
            suffix_parts.append("final")
        suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)
        csv_path = out_dir / f"cross_validation_candidates{suffix}.csv"
        md_path = out_dir / f"cross_validation_report{suffix}.md"
        _write_candidates_csv(csv_path, records)
        _write_report(md_path, summary_path, wide, records, corr_clusters, run_manifest)
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")


def cross_validate(
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
    time_splits = _time_splits(wide.index, time_folds)
    universe_splits = _universe_splits(columns, universe_folds)

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
        record["decision"] = _decision(record, universe_folds)
        records.append(record)

    corr_clusters = _correlation_clusters(factor_values, corr_threshold)
    return records, corr_clusters


def _select_candidates(rows: list[dict[str, Any]], *, max_candidates: int) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if _robust_summary_candidate(row):
            selected.setdefault(row["expression"], row)
    for row in sorted(rows, key=_summary_sort_key)[:max_candidates]:
        if row["complexity_score"] <= 8:
            selected.setdefault(row["expression"], row)
    return list(selected.values())


def _load_manual_candidates(expressions: list[str], path: Path | None) -> list[dict[str, Any]]:
    candidates = []
    for idx, expression in enumerate(expressions, start=1):
        candidates.append(_manual_candidate(f"manual_{idx:03d}", expression))
    if path is None:
        return candidates

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            for idx, row in enumerate(csv.DictReader(handle), start=1):
                expression = (row.get("expression") or "").strip()
                if not expression:
                    continue
                name = (row.get("name") or row.get("factor_name") or f"manual_file_{idx:03d}").strip()
                candidates.append(_manual_candidate(name, expression))
        return candidates

    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            expression = line.strip()
            if not expression or expression.startswith("#"):
                continue
            candidates.append(_manual_candidate(f"manual_file_{idx:03d}", expression))
    return candidates


def _manual_candidate(name: str, expression: str) -> dict[str, Any]:
    return {
        "factor_name": name,
        "expression": expression,
        "train_sharpe": np.nan,
        "valid_sharpe": np.nan,
        "test_sharpe": np.nan,
        "test_ic_ir": np.nan,
        "complexity_score": np.nan,
    }


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        deduped.setdefault(candidate["expression"], candidate)
    return list(deduped.values())


def _wide_for_report_mode(wide: pd.DataFrame, manifest: dict[str, Any], report_mode: str) -> pd.DataFrame:
    if report_mode == "final":
        return wide
    ratios = tuple(float(value) for value in manifest.get("segment_ratios", (0.70, 0.15, 0.15)))
    valid_slice = _time_segment_slices(len(wide), ratios)["valid"]
    return wide.iloc[valid_slice]


def _robust_summary_candidate(row: dict[str, Any]) -> bool:
    return (
        row["valid_sharpe"] > 1
        and row["train_sharpe"] > 0
        and row["valid_ic_ir"] > 0.1
        and row["complexity_score"] <= 6
    )


def _decision(record: dict[str, Any], universe_folds: int) -> str:
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


def _summary_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    valid_sharpe = _nan_to_neg_inf(row.get("valid_sharpe", np.nan))
    train_sharpe = _nan_to_neg_inf(row.get("train_sharpe", np.nan))
    complexity = row.get("complexity_score", np.nan)
    complexity_sort = -float(complexity) if np.isfinite(complexity) else float("-inf")
    return (-valid_sharpe, -train_sharpe, -complexity_sort)


def _nan_to_neg_inf(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("-inf")


def _load_success_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "success":
                continue
            for column in NUMERIC_COLUMNS:
                value = row.get(column, "")
                row[column] = float(value) if value != "" else np.nan
            rows.append(row)
    return rows


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _time_splits(index: pd.Index, n_folds: int) -> list[tuple[str, pd.Index]]:
    arrays = np.array_split(np.arange(len(index)), n_folds)
    return [(f"T{i}", index[array]) for i, array in enumerate(arrays, start=1) if len(array)]


def _universe_splits(columns: list[str], n_folds: int) -> list[tuple[str, list[str]]]:
    groups = {i: [] for i in range(n_folds)}
    for column in columns:
        digest = hashlib.md5(str(column).encode("utf-8")).hexdigest()
        groups[int(digest, 16) % n_folds].append(column)
    return [(f"U{i + 1}", groups[i]) for i in range(n_folds) if groups[i]]


def _correlation_clusters(
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


def _write_candidates_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "decision",
        "mode",
        "report_mode",
        "name",
        "expression",
        "summary_train_sharpe",
        "summary_valid_sharpe",
        "summary_test_sharpe",
        "summary_test_ic_ir",
        "summary_complexity",
        "full_sharpe",
        "full_returns",
        "full_ic_ir",
        "full_turnover",
        "full_drawdown",
        "time_pos_folds",
        "time_gt1_folds",
        "time_min_sharpe",
        "time_median_sharpe",
        "universe_pos_groups",
        "universe_min_sharpe",
        "universe_median_sharpe",
        "time_sharpes",
        "universe_sharpes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=_record_sort_key):
            row = {field: record[field] for field in fields}
            row["time_sharpes"] = ";".join(f"{value:.3f}" for value in record["time_sharpes"])
            row["universe_sharpes"] = ";".join(f"{value:.3f}" for value in record["universe_sharpes"])
            writer.writerow(row)


def _write_report(
    path: Path,
    summary_path: Path | None,
    wide: pd.DataFrame,
    records: list[dict[str, Any]],
    corr_clusters: list[list[str]],
    manifest: dict[str, Any],
) -> None:
    expression_to_record = {record["expression"]: record for record in records}
    lines = [
        "# Cross Validation Report",
        "",
        f"Source: `{summary_path}`" if summary_path else "Source: manual expressions",
        f"Report mode: `{manifest.get('report_mode', 'validation')}`",
        (
            f"Data: {wide.index.min()} to {wide.index.max()}, "
            f"{len(wide)} bars, {wide['close'].shape[1]} symbols, "
            f"forward_periods={int(manifest.get('forward_periods', 1))}"
        ),
        "",
        "## Decisions",
    ]
    for record in sorted(records, key=_record_sort_key):
        test_part = (
            f"test_sh={record['summary_test_sharpe']:.3f} "
            if record.get("report_mode") == "final"
            else ""
        )
        lines.append(
            "- "
            f"{record['decision']} | {record['name']} "
            f"| mode={record.get('mode', 'raw')} "
            f"valid_sh={record['summary_valid_sharpe']:.3f} "
            f"{test_part}"
            f"| time_pos={record['time_pos_folds']} "
            f"time_med={record['time_median_sharpe']:.3f} "
            f"time_min={record['time_min_sharpe']:.3f} "
            f"| universe_pos={record['universe_pos_groups']} "
            f"universe_min={record['universe_min_sharpe']:.3f} "
            f"| `{record['expression']}`"
        )

    lines.extend(["", "## High-Correlation Clusters"])
    for cluster in corr_clusters:
        if len(cluster) <= 1:
            continue
        parts = []
        for expression in cluster:
            record = expression_to_record[expression]
            summary = f"valid_sh={record['summary_valid_sharpe']:.3f}"
            if record.get("report_mode") == "final":
                summary += f", test_sh={record['summary_test_sharpe']:.3f}"
            parts.append(
                f"{record['name']} ({record['decision']}, "
                f"{summary})"
            )
        lines.append("- " + "; ".join(parts))
    path.write_text("\n".join(lines), encoding="utf-8")


def _record_sort_key(record: dict[str, Any]) -> tuple[int, float]:
    order = {"accepted_candidate": 0, "watchlist": 1, "rejected": 2}
    return order.get(record["decision"], 9), -float(record["time_median_sharpe"])


if __name__ == "__main__":
    main()
