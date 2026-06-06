from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from alpha101.config import get_config
from alpha101.data import FactorDataView, build_research_wide_frame, sample_indices_after_agg
from alpha101.factors.backtesting import LongShortBacktestConfig, backtest_long_short
from alpha101.factors.backtesting.simulation import forward_compound_returns
from alpha101.factors.evaluation import (
    StyleConfig,
    adf_wide,
    build_style_factors,
    factor_style_exposures,
    residualize_style_factor,
    strategy_market_exposure,
    summarize_exposures,
)
from alpha101.factors.expression import FastExpressionEngine
from alpha101.factors.operator_lib import process_factor_wide_format


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Alpha101-style factor expression.")
    parser.add_argument("-f", "--file", type=str, required=False)
    parser.add_argument("--n-quantiles", type=int, default=None)
    parser.add_argument("--long-group", type=int, default=None)
    parser.add_argument("--short-group", type=int, default=None)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--factor-agg", type=str, default="ewma", choices=["ewma", "mean", "std"])
    parser.add_argument("--adf", action="store_true", help="Run per-symbol ADF tests for the factor.")
    parser.add_argument("--adf-lag", type=int, default=1)
    parser.add_argument("--exposure", action="store_true", help="Report size/momentum/volatility exposures.")
    parser.add_argument(
        "--specific",
        action="store_true",
        help="Backtest the residual factor after neutralizing size/momentum/volatility.",
    )
    parser.add_argument("--momentum-window", type=int, default=20)
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--beta-window", type=int, default=60)
    parser.add_argument("--liquidity-window", type=int, default=20)
    parser.add_argument("--reversal-window", type=int, default=5)
    parser.add_argument("--funding-window", type=int, default=3)
    parser.add_argument("--style-min-count", type=int, default=8)
    parser.add_argument(
        "--market-beta",
        action="store_true",
        help="Report strategy pnl_net exposure to forward market returns.",
    )
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
    ("drawdown_before_cost", "Max Drawdown Before Cost"),
    ("win_rate", "Win Rate"),
    ("win_rate_after_cost", "Win Rate After Cost"),
    ("profit_loss_ratio", "Profit/Loss Ratio"),
    ("profit_loss_ratio_after_cost", "Profit/Loss Ratio After Cost"),
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
    "drawdown_before_cost",
    "win_rate",
    "win_rate_after_cost",
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


def print_adf_summary(metrics: dict[str, Any]) -> None:
    print("\nADF summary:")
    print("-" * 42)
    print(f"Symbols Tested      : {metrics['symbols_tested']:>11}")
    print(f"ADF Stat Mean       : {metrics['adf_stat_mean']:>11.4f}")
    print(f"ADF Stat Median     : {metrics['adf_stat_median']:>11.4f}")
    print(f"Approx P Mean       : {metrics['pvalue_mean']:>11.4f}")
    print(f"Approx P Median     : {metrics['pvalue_median']:>11.4f}")
    print(f"Reject 5% Ratio     : {metrics['reject_5pct_ratio'] * 100:>10.3f}%")
    print("-" * 42)


def print_exposure_summary(summary: dict[str, dict[str, float]], *, title: str = "Style exposure summary") -> None:
    print(f"\n{title}:")
    print("-" * 78)
    print(f"{'Style':<14} {'Mean':>11} {'Median':>11} {'Std':>11} {'Abs Mean':>11} {'T-stat':>11} {'Obs':>6}")
    for name, values in summary.items():
        print(
            f"{name:<14}"
            f" {values['mean']:>11.4f}"
            f" {values['median']:>11.4f}"
            f" {values['std']:>11.4f}"
            f" {values['abs_mean']:>11.4f}"
            f" {values['t_stat']:>11.4f}"
            f" {values['obs_count']:>6}"
        )
    print("-" * 78)


def print_strategy_market_exposure(metrics: dict[str, Any]) -> None:
    print("\nStrategy market exposure:")
    print("-" * 42)
    print(f"Alpha               : {metrics['alpha']:>11.6f}")
    print(f"Market Beta         : {metrics['beta']:>11.4f}")
    print(f"R-squared           : {metrics['r2']:>11.4f}")
    print(f"Correlation         : {metrics['corr']:>11.4f}")
    print(f"Beta T-stat         : {metrics['t_stat']:>11.4f}")
    print(f"Observations        : {metrics['obs_count']:>11}")
    print("-" * 42)


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
    view = FactorDataView(wide_data)
    engine = FastExpressionEngine(view)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            fast_expression = f.read()
    else:
        fast_expression = args.expression

    if not fast_expression:
        print("Error: No expression provided")
        print("Usage: alpha101-expression <expression>")
        raise SystemExit(1)

    n_quantiles = args.n_quantiles if args.n_quantiles is not None else cfg.n_quantiles
    short_group = args.short_group if args.short_group is not None else cfg.short_group
    long_group = args.long_group if args.long_group is not None else cfg.long_group
    long_group = long_group if long_group is not None else n_quantiles
    if long_group == short_group:
        raise SystemExit("Error: long_group and short_group must be different")
    if n_quantiles < 2:
        raise SystemExit("Error: n_quantiles must be at least 2")
    if long_group < 1 or long_group > n_quantiles:
        raise SystemExit("Error: long_group must be between 1 and n_quantiles")
    if short_group < 1 or short_group > n_quantiles:
        raise SystemExit("Error: short_group must be between 1 and n_quantiles")

    result = engine.evaluate(fast_expression)
    print("Raw factor tail:")
    print(result.tail())

    result.index.name = "date"
    result.columns.name = "symbol"
    result = process_factor_wide_format(result)

    style_config = StyleConfig(
        momentum_window=args.momentum_window,
        volatility_window=args.volatility_window,
        beta_window=args.beta_window,
        liquidity_window=args.liquidity_window,
        reversal_window=args.reversal_window,
        funding_window=args.funding_window,
        min_count=args.style_min_count,
    )
    styles = None

    if args.adf:
        print_adf_summary(adf_wide(result, max_lag=args.adf_lag))

    if args.exposure or args.specific:
        styles = build_style_factors(
            view.close.reindex(columns=result.columns),
            view.cap.reindex(columns=result.columns) if view.cap is not None else None,
            view.market_return.reindex(columns=result.columns),
            view.volume.reindex(columns=result.columns),
            view.funding.reindex(columns=result.columns) if view.funding is not None else None,
            momentum_window=args.momentum_window,
            volatility_window=args.volatility_window,
            beta_window=args.beta_window,
            liquidity_window=args.liquidity_window,
            reversal_window=args.reversal_window,
            funding_window=args.funding_window,
        )

    if args.exposure:
        exposures, _ = factor_style_exposures(result, styles, config=style_config)
        print_exposure_summary(summarize_exposures(exposures), title="Raw factor style exposure")

    if args.specific:
        result = residualize_style_factor(result, styles, config=style_config)
        print("Specific residual factor tail:")
        print(result.tail())
        if args.exposure:
            exposures, _ = factor_style_exposures(result, styles, config=style_config)
            print_exposure_summary(
                summarize_exposures(exposures),
                title="Specific residual factor style exposure",
            )

    factor_name = "alpha_test"
    result.columns = pd.MultiIndex.from_product([[factor_name], result.columns])

    df = pd.concat([wide_data, result], axis=1)
    curve_df, metrics = backtest_long_short(
        panel_df=df,
        factor_name=factor_name,
        target_col="target",
        config=LongShortBacktestConfig(
            n_quantiles=n_quantiles,
            long_group=long_group,
            short_group=short_group,
            leverage=args.leverage,
            k_bars=cfg.trade_per_k_bars,
            freq=cfg.timeframe,
            factor_agg=args.factor_agg,
            single_side_fee=cfg.single_side_fee,
            round_trip_fee=cfg.round_trip_fee,
        ),
    )
    if args.exposure or args.market_beta:
        market_forward = forward_compound_returns(
            view.market_return.iloc[:, :1].shift(-1).to_numpy(dtype=float, copy=False),
            cfg.trade_per_k_bars,
        )
        sampled_idx = sample_indices_after_agg(len(wide_data), cfg.trade_per_k_bars)
        print_strategy_market_exposure(
            strategy_market_exposure(
                curve_df["pnl_net"].to_numpy(dtype=float, copy=False),
                market_forward[sampled_idx, 0],
            )
        )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
