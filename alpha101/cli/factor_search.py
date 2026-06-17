from __future__ import annotations

import argparse
import random

import numpy as np

from alpha101.config import get_config
from alpha101.data import build_ashare_wide_frame_from_csv, build_research_wide_frame
from alpha101.factors.search import FactorSearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Factor Search Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/alpha101*.json")
    parser.add_argument("--strategy", type=str, default="genetic", choices=["genetic"])
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--forward-periods", type=int, default=1)
    parser.add_argument("--timeframe", type=str, default=None, help="Override config timeframe, e.g. 1h or 4h.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "process", "serial"])
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel workers for batch expression evaluation.")
    parser.add_argument("--profile", action="store_true", help="Print per-generation timing breakdown.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible factor generation.")
    parser.add_argument("--seed-expression", action="append", default=None, help="Expression to seed into the initial GP population.")
    parser.add_argument("--seed-expressions-file", type=str, default=None, help="Text file with one seed expression per line.")
    parser.add_argument("--max-complexity", type=float, default=36.0, help="Reject generated expressions above this complexity.")
    parser.add_argument("--complexity-penalty", type=float, default=0.0, help="Penalty applied to GP selection fitness per complexity point.")
    parser.add_argument("--diversity-penalty", type=float, default=0.0, help="Penalty for duplicate expression families within a generation.")
    parser.add_argument("--min-valid-sharpe", type=float, default=None, help="Optional validation Sharpe floor for GP selection.")
    parser.add_argument("--min-valid-ic-ir", type=float, default=None, help="Optional validation IC IR floor for GP selection.")
    parser.add_argument("--validation-failure-penalty", type=float, default=0.0, help="Penalty when validation floors are not met.")
    parser.add_argument("--train-valid-gap-penalty", type=float, default=0.0, help="Penalty per Sharpe point of train-valid gap.")
    parser.add_argument("--validation-interval", type=int, default=5, help="Run elite robustness validation every N generations; <=0 disables it.")
    parser.add_argument("--validation-time-folds", type=int, default=4)
    parser.add_argument("--validation-universe-folds", type=int, default=3)
    parser.add_argument("--validation-walk-forward-folds", type=int, default=4)
    parser.add_argument(
        "--validation-extra-n-quantiles",
        type=int,
        nargs="*",
        default=[10],
        help="Extra quantile counts checked during elite validation, e.g. 10.",
    )
    parser.add_argument("--cv-failure-penalty", type=float, default=0.5, help="Penalty per failed elite robustness validation set.")
    parser.add_argument(
        "--pnl-dedupe-interval",
        type=int,
        default=1,
        help="Run search-visible PnL correlation dedupe every N generations; <=0 disables it.",
    )
    parser.add_argument(
        "--pnl-corr-threshold",
        type=float,
        default=0.85,
        help="Max allowed absolute PnL correlation among candidates kept for GA selection.",
    )
    parser.add_argument(
        "--pnl-redundancy-penalty",
        type=float,
        default=999.0,
        help="Selection fitness penalty applied to PnL-redundant candidates.",
    )
    parser.add_argument(
        "--exclude-fields",
        nargs="*",
        default=None,
        help="Data fields to exclude from random generation, e.g. --exclude-fields cap funding.",
    )
    parser.add_argument(
        "--ashare-csv",
        action="append",
        default=None,
        help="A-share long CSV path. Repeat for multiple files. Uses direct target if present.",
    )
    parser.add_argument("--ashare-date-col", default="date")
    parser.add_argument("--ashare-symbol-col", default="code")
    parser.add_argument("--ashare-cap-col", default="market_cap")
    parser.add_argument("--ashare-target-col", default="label_5d")
    parser.add_argument("--ashare-tradeable-col", default=None)
    parser.add_argument(
        "--ashare-include-vwap",
        action="store_true",
        help="Expose typical-price VWAP for A-share experiments. Disabled by default.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    seed_expressions = _load_seed_expressions(args.seed_expression or [], args.seed_expressions_file)

    cfg = get_config(args.config)
    if args.timeframe is not None:
        cfg.timeframe = args.timeframe
    wide_data = _load_wide_data_from_args(args, cfg)
    search_engine = FactorSearchEngine(
        wide_data,
        output_dir=args.output_dir or str(cfg.output_dir),
        timeframe=cfg.timeframe,
        n_quantiles=args.n_quantiles,
        forward_periods=args.forward_periods,
        transaction_cost=cfg.round_trip_fee,
        segment_ratios=cfg.search_segment_ratios,
        seed=args.seed,
        exclude_fields=args.exclude_fields,
        seed_expressions=seed_expressions,
        max_complexity=args.max_complexity,
        complexity_penalty=args.complexity_penalty,
        diversity_penalty=args.diversity_penalty,
        min_valid_sharpe=args.min_valid_sharpe,
        min_valid_ic_ir=args.min_valid_ic_ir,
        validation_failure_penalty=args.validation_failure_penalty,
        train_valid_gap_penalty=args.train_valid_gap_penalty,
        validation_interval=args.validation_interval,
        validation_time_folds=args.validation_time_folds,
        validation_universe_folds=args.validation_universe_folds,
        validation_walk_forward_folds=args.validation_walk_forward_folds,
        validation_extra_n_quantiles=args.validation_extra_n_quantiles,
        cv_failure_penalty=args.cv_failure_penalty,
        pnl_dedupe_interval=args.pnl_dedupe_interval,
        pnl_corr_threshold=args.pnl_corr_threshold,
        pnl_redundancy_penalty=args.pnl_redundancy_penalty,
    )
    search_engine.save_manifest(cli_args=vars(args))

    search_engine.genetic_search(
        population_size=args.population,
        n_generations=args.generations,
        backend=args.backend,
        max_workers=args.n_jobs,
        profile=args.profile,
    )
    search_engine.summarize_results()


def _load_seed_expressions(expressions: list[str], path: str | None) -> list[str]:
    out = [expr.strip() for expr in expressions if expr.strip()]
    if path is None:
        return out
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            expr = line.strip()
            if expr and not expr.startswith("#"):
                out.append(expr)
    return out


def _load_wide_data_from_args(args, cfg):
    if getattr(args, "ashare_csv", None):
        return build_ashare_wide_frame_from_csv(
            args.ashare_csv,
            date_col=args.ashare_date_col,
            symbol_col=args.ashare_symbol_col,
            cap_col=args.ashare_cap_col or None,
            target_col=args.ashare_target_col or None,
            tradeable_col=args.ashare_tradeable_col or None,
            include_vwap=bool(args.ashare_include_vwap),
        )
    return build_research_wide_frame(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        buffer=cfg.pre_buffer_candles,
    )


if __name__ == "__main__":
    main()
