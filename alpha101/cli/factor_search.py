from __future__ import annotations

import argparse

from alpha101.config import get_config
from alpha101.data.panel import build_wide_df
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
    parser.add_argument("--expression-backend", type=str, default="pandas", choices=["pandas", "polars"])
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel workers for batch expression evaluation.")
    parser.add_argument("--profile", action="store_true", help="Print per-generation timing breakdown.")
    args = parser.parse_args()

    cfg = get_config(args.config)
    wide_data = build_wide_df(
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
        expression_backend=args.expression_backend,
    )

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
