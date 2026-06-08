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
from alpha101.data import build_research_wide_frame
from alpha101.factors.evaluation import StyleConfig
from alpha101.factors.validation import cross_validate_candidates, wide_for_report_mode


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
    parser.add_argument("--pnl-corr-threshold", type=float, default=0.85)
    parser.add_argument(
        "--robust-max-complexity",
        type=float,
        default=10.0,
        help="Max complexity for summary rows that already pass robust train/valid filters.",
    )
    parser.add_argument(
        "--candidate-max-complexity",
        type=float,
        default=12.0,
        help="Max complexity for top summary rows added as validation candidates.",
    )
    parser.add_argument("--timeframe", type=str, default=None, help="Override config timeframe, e.g. 1h or 4h.")
    parser.add_argument(
        "--report-mode",
        choices=["validation", "final"],
        default="validation",
        help="validation excludes held-out test data; final may report test metrics for frozen candidates.",
    )
    parser.add_argument(
        "--final-evaluate",
        action="store_true",
        help="Alias for --report-mode final; use only after candidate expressions and selection rules are frozen.",
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
    if args.final_evaluate:
        args.report_mode = "final"

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
        candidates.extend(
            _select_candidates(
                rows,
                max_candidates=args.max_candidates,
                robust_max_complexity=args.robust_max_complexity,
                candidate_max_complexity=args.candidate_max_complexity,
            )
        )
    candidates.extend(_load_manual_candidates(args.expression or [], args.expressions_file))
    candidates = _dedupe_candidates(candidates)
    candidate_source = _candidate_source_metadata(summary_path, args.expression or [], args.expressions_file)

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
        run_manifest = {
            **manifest,
            "forward_periods": int(forward_periods),
            "report_mode": args.report_mode,
            "test_used_for_selection": False,
            "final_evaluation": args.report_mode == "final",
            "candidate_source": candidate_source,
            "candidate_count": len(candidates),
            "pnl_corr_threshold": args.pnl_corr_threshold,
            "robust_max_complexity": args.robust_max_complexity,
            "candidate_max_complexity": args.candidate_max_complexity,
        }
        eval_wide = wide_for_report_mode(wide, run_manifest, args.report_mode)
        records, corr_clusters = cross_validate_candidates(
            candidates,
            eval_wide,
            manifest=run_manifest,
            time_folds=args.time_folds,
            universe_folds=args.universe_folds,
            corr_threshold=args.corr_threshold,
            pnl_corr_threshold=args.pnl_corr_threshold,
            report_mode=args.report_mode,
            specific=args.specific,
            progress_bar=True,
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
        manifest_out_path = out_dir / f"cross_validation_manifest{suffix}.json"
        _write_candidates_csv(csv_path, records)
        _write_report(md_path, summary_path, wide, records, corr_clusters, run_manifest)
        manifest_out_path.write_text(json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")
        print(f"Wrote {manifest_out_path}")


def _select_candidates(
    rows: list[dict[str, Any]],
    *,
    max_candidates: int,
    robust_max_complexity: float = 10.0,
    candidate_max_complexity: float = 12.0,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if _robust_summary_candidate(row, max_complexity=robust_max_complexity):
            selected.setdefault(row["expression"], row)
    for row in sorted(rows, key=_summary_sort_key)[:max_candidates]:
        if row["complexity_score"] <= candidate_max_complexity:
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


def _candidate_source_metadata(
    summary_path: Path | None,
    expressions: list[str],
    expressions_file: Path | None,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    if summary_path is not None:
        sources.append(_file_source_metadata("summary_csv", summary_path))
    if expressions_file is not None:
        sources.append(_file_source_metadata("expressions_file", expressions_file))
    if expressions:
        payload = "\n".join(expressions).encode("utf-8")
        sources.append(
            {
                "type": "manual_expressions",
                "count": len(expressions),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {"sources": sources}


def _file_source_metadata(source_type: str, path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "type": source_type,
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _robust_summary_candidate(row: dict[str, Any], *, max_complexity: float) -> bool:
    return (
        row["valid_sharpe"] > 1
        and row["train_sharpe"] > 0
        and row["valid_ic_ir"] > 0.1
        and row["complexity_score"] <= max_complexity
    )

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
        "max_pnl_corr",
        "nearest_pnl_corr_expression",
        "redundant_by_pnl",
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
        f"Final evaluation: `{bool(manifest.get('final_evaluation', False))}`",
        f"Test used for selection: `{bool(manifest.get('test_used_for_selection', False))}`",
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
            f"| pnl_corr={record['max_pnl_corr']:.3f} "
            f"redundant={record['redundant_by_pnl']} "
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


def _record_sort_key(record: dict[str, Any]) -> tuple[int, int, float]:
    order = {"accepted_candidate": 0, "watchlist": 1, "rejected": 2}
    redundancy = 1 if record.get("redundant_by_pnl", False) else 0
    return order.get(record["decision"], 9), redundancy, -float(record["time_median_sharpe"])


if __name__ == "__main__":
    main()
