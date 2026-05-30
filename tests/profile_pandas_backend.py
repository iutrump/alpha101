from __future__ import annotations

import argparse
import gc
import statistics
import time

import numpy as np
import pandas as pd

from alpha101.data import FactorDataView
from alpha101.factors.evaluation import forward_returns_array, score_factor_cross_section_array
from alpha101.factors.expression.runtime import alpha_fields, evaluate_expression
from alpha101.factors.operator_lib import process_factor_wide_format


PROFILE_EXPRESSIONS = {
    "ts_alpha": "ts_alpha(close, volume, 14)",
    "ts_arg_max": "ts_arg_max(close, 14)",
    "ts_arg_min": "ts_arg_min(close, 14)",
    "ts_beta": "ts_beta(close, volume, 14)",
    "ts_corr": "ts_corr(close, volume, 20)",
    "ts_covariance": "ts_covariance(close, volume, 20)",
    "ts_decay": "ts_decay_linear(close, 21)",
    "ts_delay": "ts_delay(close, 5)",
    "ts_delta": "ts_delta(close, 5)",
    "ts_drawdown": "ts_drawdown(close, 14)",
    "ts_ema": "ts_ema(close, 12)",
    "ts_kurtosis": "ts_kurtosis(close, 20)",
    "ts_max": "ts_max(high, 20)",
    "ts_mean": "ts_mean(close, 20)",
    "ts_min": "ts_min(low, 20)",
    "ts_pos": "ts_pos(close, 14)",
    "ts_product": "ts_product(close / 100, 10)",
    "ts_r2": "ts_r2(close, volume, 14)",
    "ts_rank": "ts_rank(close, 20)",
    "ts_resid": "ts_resid(close, volume, 12)",
    "ts_skewness": "ts_skewness(close, 20)",
    "ts_slope": "ts_slope(close, 24)",
    "ts_std_dev": "ts_std_dev(close, 21)",
    "ts_sum": "ts_sum(volume, 20)",
    "ts_zscore": "ts_zscore(close, 14)",
    "rank": "rank(close)",
    "scale": "scale(close)",
    "winsorize": "winsorize(close)",
    "zscore": "zscore(close)",
    "if_else": "if_else(close > open, close, open)",
    "trade_when": "trade_when(close > open, close)",
    "process": "process_factor_wide_format(close)",
    "pipeline": "rank(ts_mean(close, 20) - ts_mean(open, 20))",
}


def make_profile_wide_data(rows: int, symbols: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    cols = [f"S{i:04d}" for i in range(symbols)]
    base = 100 + rng.normal(0, 1, size=(rows, symbols)).cumsum(axis=0)
    close = pd.DataFrame(base, index=dates, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0]).add(rng.normal(0, 0.1, size=(rows, symbols)))
    high = pd.DataFrame(
        np.maximum(open_.to_numpy(), close.to_numpy()) + rng.random((rows, symbols)),
        index=dates,
        columns=cols,
    )
    low = pd.DataFrame(
        np.minimum(open_.to_numpy(), close.to_numpy()) - rng.random((rows, symbols)),
        index=dates,
        columns=cols,
    )
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


def run_operator_profile(rows: int, symbols: int, repeats: int, warmups: int) -> list[dict]:
    wide_data = make_profile_wide_data(rows, symbols)
    fields = alpha_fields(FactorDataView(wide_data))
    results = []
    for name, expression in PROFILE_EXPRESSIONS.items():
        for _ in range(warmups):
            evaluate_expression(expression, fields)
        times = _time_repeats(lambda: evaluate_expression(expression, fields), repeats)
        results.append({"name": name, "pandas_ms": statistics.median(times) * 1000})
    return results


def run_pipeline_profile(rows: int, symbols: int, repeats: int, warmups: int) -> list[dict]:
    wide_data = make_profile_wide_data(rows, symbols)
    view = FactorDataView(wide_data)
    fields = alpha_fields(view)
    close = wide_data["close"]
    target = forward_returns_array(close.to_numpy(dtype=float, copy=False), periods=1)
    results = []

    for name, expression in PROFILE_EXPRESSIONS.items():
        for _ in range(warmups):
            _run_pipeline_once(expression, fields, target)

        stage_times = [_run_pipeline_once(expression, fields, target) for _ in range(repeats)]
        results.append(
            {
                "name": name,
                "eval_ms": _median_stage(stage_times, "eval") * 1000,
                "preprocess_ms": _median_stage(stage_times, "preprocess") * 1000,
                "to_numpy_ms": _median_stage(stage_times, "to_numpy") * 1000,
                "score_ms": _median_stage(stage_times, "score") * 1000,
                "total_ms": _median_stage(stage_times, "total") * 1000,
            }
        )
    return results


def _run_pipeline_once(expression: str, fields: dict, target: np.ndarray) -> dict:
    start = time.perf_counter()
    factor = evaluate_expression(expression, fields)
    after_eval = time.perf_counter()
    processed = process_factor_wide_format(factor)
    after_preprocess = time.perf_counter()
    values = processed.to_numpy(dtype=float, copy=False)
    after_to_numpy = time.perf_counter()
    score_factor_cross_section_array(values, target[:, : values.shape[1]], n_quantiles=5)
    end = time.perf_counter()
    return {
        "eval": after_eval - start,
        "preprocess": after_preprocess - after_eval,
        "to_numpy": after_to_numpy - after_preprocess,
        "score": end - after_to_numpy,
        "total": end - start,
    }


def _median_stage(stage_times: list[dict], key: str) -> float:
    return statistics.median(item[key] for item in stage_times)


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


def print_operator_results(results: list[dict], rows: int, symbols: int, repeats: int) -> None:
    print(f"operator profile: rows={rows} symbols={symbols} cells={rows * symbols} repeats={repeats}")
    print(f"{'operator':<12} {'pandas_ms':>10}")
    for item in results:
        print(f"{item['name']:<12} {item['pandas_ms']:>10.3f}")


def print_pipeline_results(results: list[dict], rows: int, symbols: int, repeats: int) -> None:
    print(f"pipeline profile: rows={rows} symbols={symbols} cells={rows * symbols} repeats={repeats}")
    print(
        f"{'operator':<12} {'eval_ms':>10} {'prep_ms':>10} {'numpy_ms':>10} "
        f"{'score_ms':>10} {'total_ms':>10}"
    )
    for item in results:
        print(
            f"{item['name']:<12} {item['eval_ms']:>10.3f} {item['preprocess_ms']:>10.3f} "
            f"{item['to_numpy_ms']:>10.3f} {item['score_ms']:>10.3f} {item['total_ms']:>10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile pandas expression and scoring stages.")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--symbols", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--mode", choices=["operators", "pipeline", "all"], default="all")
    args = parser.parse_args()

    if args.mode in {"operators", "all"}:
        operator_results = run_operator_profile(args.rows, args.symbols, args.repeats, args.warmups)
        print_operator_results(operator_results, args.rows, args.symbols, args.repeats)
    if args.mode == "all":
        print()
    if args.mode in {"pipeline", "all"}:
        pipeline_results = run_pipeline_profile(args.rows, args.symbols, args.repeats, args.warmups)
        print_pipeline_results(pipeline_results, args.rows, args.symbols, args.repeats)


if __name__ == "__main__":
    main()
