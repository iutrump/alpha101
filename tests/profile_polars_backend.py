from __future__ import annotations

import argparse
import gc
import math
import statistics
import time

import numpy as np
import pandas as pd

from alpha101.data import FactorDataView
from alpha101.factors.expression.polars_runtime import (
    alpha_polars_fields,
    evaluate_polars_expression,
    polars_result_to_pandas,
)
from alpha101.factors.expression.runtime import alpha_fields, evaluate_expression


PROFILE_EXPRESSIONS = {
    "rank": "rank(close)",
    "zscore": "zscore(close)",
    "ts_mean": "ts_mean(close, 20)",
    "ts_delta": "ts_delta(close, 5)",
    "ts_max_min": "ts_max(high, 20) - ts_min(low, 20)",
    "ts_rank": "ts_rank(close, 20)",
    "ts_corr": "ts_corr(close, volume, 20)",
    "ts_decay": "ts_decay_linear(close, 20)",
    "ts_slope": "ts_slope(close, 20)",
    "pipeline": "rank(ts_mean(close, 20) - ts_mean(open, 20))",
}


def make_profile_wide_data(rows: int, symbols: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    cols = [f"S{i:04d}" for i in range(symbols)]
    base = 100 + rng.normal(0, 1, size=(rows, symbols)).cumsum(axis=0)
    close = pd.DataFrame(base, index=dates, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0]).add(rng.normal(0, 0.1, size=(rows, symbols)))
    high = pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()) + rng.random((rows, symbols)), index=dates, columns=cols)
    low = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()) - rng.random((rows, symbols)), index=dates, columns=cols)
    volume = pd.DataFrame(rng.lognormal(mean=8, sigma=0.4, size=(rows, symbols)), index=dates, columns=cols)
    vwap = (high + low + close) / 3
    cap = close * rng.uniform(1e6, 1e9, size=(1, symbols))
    funding = pd.DataFrame(rng.normal(0, 0.0001, size=(rows, symbols)), index=dates, columns=cols)
    target = close.pct_change(fill_method=None).shift(-1)
    return pd.concat(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "vwap": vwap,
            "cap": cap,
            "funding": funding,
            "target": target,
        },
        axis=1,
    )


def run_profile(rows: int, symbols: int, repeats: int, warmups: int) -> list[dict]:
    wide_data = make_profile_wide_data(rows, symbols)
    view = FactorDataView(wide_data)
    pandas_fields = alpha_fields(view)
    polars_fields = alpha_polars_fields(view)

    results = []
    for name, expression in PROFILE_EXPRESSIONS.items():
        for _ in range(warmups):
            evaluate_expression(expression, pandas_fields)
            polars_result_to_pandas(evaluate_polars_expression(expression, polars_fields))

        pandas_times = _time_repeats(lambda: evaluate_expression(expression, pandas_fields), repeats)
        polars_times = _time_repeats(
            lambda: polars_result_to_pandas(evaluate_polars_expression(expression, polars_fields)),
            repeats,
        )

        pandas_result = evaluate_expression(expression, pandas_fields)
        polars_result = polars_result_to_pandas(evaluate_polars_expression(expression, polars_fields))
        precision = _precision_stats(pandas_result, polars_result)
        pandas_ms = statistics.median(pandas_times) * 1000
        polars_ms = statistics.median(polars_times) * 1000
        results.append(
            {
                "name": name,
                "pandas_ms": pandas_ms,
                "polars_ms": polars_ms,
                "speedup": pandas_ms / polars_ms if polars_ms else math.inf,
                **precision,
            }
        )
    return results


def _time_repeats(fn, repeats: int) -> list[float]:
    times = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    return times


def _precision_stats(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    right = right.reindex(index=left.index, columns=left.columns)
    left_values = left.to_numpy(dtype=float, copy=False)
    right_values = right.to_numpy(dtype=float, copy=False)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if not valid.any():
        return {"max_abs_diff": math.nan, "mean_abs_diff": math.nan, "valid_ratio": 0.0}
    diff = np.abs(left_values[valid] - right_values[valid])
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "valid_ratio": float(valid.mean()),
    }


def print_results(results: list[dict], rows: int, symbols: int, repeats: int) -> None:
    print(f"rows={rows} symbols={symbols} cells={rows * symbols} repeats={repeats}")
    print(
        f"{'operator':<12} {'pandas_ms':>10} {'polars_ms':>10} {'speedup':>9} "
        f"{'max_diff':>12} {'mean_diff':>12} {'valid':>8}"
    )
    for item in results:
        print(
            f"{item['name']:<12} {item['pandas_ms']:>10.3f} {item['polars_ms']:>10.3f} "
            f"{item['speedup']:>9.3f} {item['max_abs_diff']:>12.3g} "
            f"{item['mean_abs_diff']:>12.3g} {item['valid_ratio']:>8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile pandas vs polars expression operators.")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--symbols", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()

    results = run_profile(
        rows=args.rows,
        symbols=args.symbols,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    print_results(results, rows=args.rows, symbols=args.symbols, repeats=args.repeats)


if __name__ == "__main__":
    main()
