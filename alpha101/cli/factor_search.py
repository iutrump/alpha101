from __future__ import annotations

import argparse
import random

import numpy as np

from alpha101.config import get_config
from alpha101.data import build_research_wide_frame
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
    parser.add_argument(
        "--exclude-fields",
        nargs="*",
        default=None,
        help="Data fields to exclude from random generation, e.g. --exclude-fields cap funding.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    seed_expressions = _load_seed_expressions(args.seed_expression or [], args.seed_expressions_file)

    cfg = get_config(args.config)
    if args.timeframe is not None:
        cfg.timeframe = args.timeframe
    wide_data = build_research_wide_frame(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        buffer=cfg.pre_buffer_candles,
    )
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


if __name__ == "__main__":
    main()
