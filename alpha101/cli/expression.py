from __future__ import annotations

import argparse

import pandas as pd

from alpha101.config import get_config
from alpha101.data.panel import build_wide_df
from alpha101.factors.alpha_data import Alphas
from alpha101.factors.expression import FastExpressionEngine
from alpha101.factors.operator_lib import process_factor_wide_format


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Alpha101-style factor expression.")
    parser.add_argument("-f", "--file", type=str, required=False)
    parser.add_argument("expression", type=str, nargs="?", default=None)
    return parser.parse_args()


def main() -> None:
    cfg = get_config()
    print(f"test start from {cfg.test_start_date} {cfg.test_end_date}")
    wide_data = build_wide_df(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        buffer=cfg.pre_buffer_candles,
    )
    engine = FastExpressionEngine(Alphas(wide_data))
    args = parse_args()
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            fast_expression = f.read()
    else:
        fast_expression = args.expression

    if not fast_expression:
        print("Error: No expression provided")
        print("Usage: alpha101-expression <expression>")
        raise SystemExit(1)

    result = engine.evaluate(fast_expression)
    print(result)

    result.index.name = "date"
    result.columns.name = "symbol"
    result = process_factor_wide_format(result)
    result.columns = pd.MultiIndex.from_product([["alpha_test"], result.columns])

    n_bars = int(
        cfg.pre_buffer_candles
        * pd.Timedelta("1d").total_seconds()
        // pd.Timedelta(cfg.timeframe).total_seconds()
    )
    df = pd.concat([wide_data, result], axis=1).iloc[n_bars:]
    print(df.tail())


if __name__ == "__main__":
    main()
