from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from alpha101.config import get_config
from alpha101.data import FactorDataView, build_research_wide_frame
from alpha101.factors.backtesting import LongShortBacktestConfig, backtest_long_short
from alpha101.factors.expression import FastExpressionEngine
from alpha101.factors.operator_lib import process_factor_wide_format


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Alpha101-style factor expression.")
    parser.add_argument("-f", "--file", type=str, required=False)
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--long-group", type=int, default=None)
    parser.add_argument("--short-group", type=int, default=1)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--factor-agg", type=str, default="ewma", choices=["ewma", "mean", "std"])
    parser.add_argument("expression", type=str, nargs="?", default=None)
    return parser.parse_args()


METRIC_LABELS = [
    ("sharpe", "Sharpe"),
    ("sharpe_after_cost", "Sharpe After Cost"),
    ("cagr", "CAGR"),
    ("cagr_after_cost", "CAGR After Cost"),
    ("returns", "Annual Returns"),
    ("returns_after_cost", "Annual Returns After Cost"),
    ("annual_cost_drag", "Annual Cost Drag"),
    ("funding_annual", "Funding Annualized"),
    ("drawdown", "Max Drawdown"),
    ("win_rate", "Win Rate"),
    ("turnover", "Turnover"),
    ("margin", "Margin x1000"),
    ("margin_after_cost", "Margin After Cost x1000"),
    ("ic_mean", "IC Mean"),
    ("ic_std", "IC Std"),
    ("ic_ir", "IC IR"),
    ("obs_count", "Observations"),
    ("symbols", "Symbols"),
    ("single_side_fee", "Single Side Fee"),
    ("round_trip_fee", "Round Trip Fee"),
    ("avg_long_funding", "Avg Long Funding"),
    ("avg_short_funding", "Avg Short Funding"),
    ("leverage", "Leverage"),
    ("long_group", "Long Group"),
    ("short_group", "Short Group"),
]

PERCENT_METRICS = {
    "cagr",
    "cagr_after_cost",
    "returns",
    "returns_after_cost",
    "annual_cost_drag",
    "funding_annual",
    "drawdown",
    "win_rate",
    "turnover",
    "single_side_fee",
    "round_trip_fee",
    "avg_long_funding",
    "avg_short_funding",
}


def format_metric_value(key: str, value: Any) -> str:
    if isinstance(value, float):
        if key in PERCENT_METRICS:
            return f"{value * 100:>10.3f}%"
        return f"{value:>11.4f}"
    return f"{value:>11}"


def print_metrics(metrics: dict[str, Any]) -> None:
    rows = [(label, format_metric_value(key, metrics[key])) for key, label in METRIC_LABELS if key in metrics]
    label_width = max(len(label) for label, _ in rows)
    print("\nBacktest metrics:")
    print("-" * (label_width + 16))
    for label, value in rows:
        print(f"{label:<{label_width}} : {value}")
    print("-" * (label_width + 16))


def main() -> None:
    args = parse_args()
    cfg = get_config()
    print(f"test start from {cfg.test_start_date} {cfg.test_end_date}")
    wide_data = build_research_wide_frame(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        buffer=cfg.pre_buffer_candles,
    )
    engine = FastExpressionEngine(FactorDataView(wide_data))
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            fast_expression = f.read()
    else:
        fast_expression = args.expression

    if not fast_expression:
        print("Error: No expression provided")
        print("Usage: alpha101-expression <expression>")
        raise SystemExit(1)

    long_group = args.long_group if args.long_group is not None else args.n_quantiles
    if long_group == args.short_group:
        raise SystemExit("Error: --long-group and --short-group must be different")
    if long_group < 1 or long_group > args.n_quantiles:
        raise SystemExit("Error: --long-group must be between 1 and --n-quantiles")
    if args.short_group < 1 or args.short_group > args.n_quantiles:
        raise SystemExit("Error: --short-group must be between 1 and --n-quantiles")

    result = engine.evaluate(fast_expression)
    print("Raw factor tail:")
    print(result.tail())

    result.index.name = "date"
    result.columns.name = "symbol"
    result = process_factor_wide_format(result)
    factor_name = "alpha_test"
    result.columns = pd.MultiIndex.from_product([[factor_name], result.columns])

    df = pd.concat([wide_data, result], axis=1)
    curve_df, metrics = backtest_long_short(
        panel_df=df,
        factor_name=factor_name,
        target_col="target",
        config=LongShortBacktestConfig(
            n_quantiles=args.n_quantiles,
            long_group=long_group,
            short_group=args.short_group,
            leverage=args.leverage,
            k_bars=cfg.trade_per_k_bars,
            freq=cfg.timeframe,
            factor_agg=args.factor_agg,
            single_side_fee=cfg.single_side_fee,
            round_trip_fee=cfg.round_trip_fee,
        ),
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
