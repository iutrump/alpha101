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
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "process", "serial"])
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel workers for batch expression evaluation.")
    parser.add_argument("--profile", action="store_true", help="Print per-generation timing breakdown.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible factor generation.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    cfg = get_config(args.config)
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


if __name__ == "__main__":
    main()
