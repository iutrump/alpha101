from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from alpha101.factors.runtime import alpha_fields, build_eval_env, evaluate_expression


_FAST_WORKER_ENV_BASE = None


def _init_fast_worker(fields: dict) -> None:
    global _FAST_WORKER_ENV_BASE
    _FAST_WORKER_ENV_BASE = build_eval_env(fields)


def _result_is_all_nan(result) -> bool:
    try:
        return bool(np.asarray(result.isnull()).all())
    except Exception:
        return False


def _prepare_factor_result(result: pd.DataFrame) -> pd.DataFrame:
    from alpha101.factors.operators import process_factor_wide_format

    result.index.name = "date"
    result.columns.name = "symbol"
    return process_factor_wide_format(result)


def _eval_one_worker(item):
    factor_name, expr = item
    try:
        env = dict(_FAST_WORKER_ENV_BASE)
        from alpha101.factors.runtime import normalize_expression_code

        lines = [line.strip() for line in normalize_expression_code(expr).split(";") if line.strip()]
        if not lines:
            raise ValueError("Expression cannot be empty")
        for line in lines[:-1]:
            var, sub_expr = line.split("=", 1)
            env[var.strip()] = eval(sub_expr.strip(), {}, env)

        result = eval(lines[-1], {}, env)
        if result is None:
            return factor_name, None, expr, f"Warning: {factor_name} returned None"
        if _result_is_all_nan(result):
            return factor_name, None, expr, f"Warning: {factor_name} returned all NaN"
        return factor_name, _prepare_factor_result(result), expr, None
    except Exception as exc:
        return factor_name, None, expr, f"Error evaluating {factor_name}: {exc}"


class FastExpressionEngine:
    def __init__(self, alpha_instance):
        self.alpha = alpha_instance

    def _build_env(self):
        return build_eval_env(alpha_fields(self.alpha))

    def evaluate(self, fast_code: str):
        return evaluate_expression(fast_code, alpha_fields(self.alpha))

    def evaluate_batch(
        self,
        expressions: Dict[str, str],
        progress_bar: bool = True,
        backend: str = "process",
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        results = {}
        items = list(expressions.items())

        if backend not in {"process", "serial"}:
            print(f"Unknown backend '{backend}', fallback to 'process'.")
            backend = "process"

        if backend == "serial" or len(items) <= 1:
            iterator = tqdm(items, desc="Calculating factors") if progress_bar else items
            for factor_name, expr in iterator:
                name, result, raw_expr, err = self._eval_one(factor_name, expr)
                if err:
                    self._print_eval_error(err, raw_expr)
                    continue
                results[name] = result
        else:
            if max_workers is None:
                max_workers = max(1, min(len(items), os.cpu_count() or 1))
            pbar = tqdm(total=len(items), desc="Calculating factors (parallel)") if progress_bar else None
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_fast_worker,
                initargs=(alpha_fields(self.alpha),),
            ) as executor:
                future_map = {executor.submit(_eval_one_worker, item): item for item in items}
                for future in as_completed(future_map):
                    name, raw_expr = future_map[future]
                    try:
                        name, result, raw_expr, err = future.result()
                    except Exception as exc:
                        err = f"Error evaluating {name}: {exc}"
                        result = None
                    if err:
                        self._print_eval_error(err, raw_expr)
                    elif result is not None:
                        results[name] = result
                    if pbar is not None:
                        pbar.update(1)
            if pbar is not None:
                pbar.close()

        if not results:
            print("No factors were successfully evaluated")
            return pd.DataFrame()

        concatenated_results = []
        for factor_name, factor_data in results.items():
            factor_data.columns = pd.MultiIndex.from_product([[factor_name], factor_data.columns])
            concatenated_results.append(factor_data)
        return pd.concat(concatenated_results, axis=1)

    def _eval_one(self, factor_name: str, expr: str):
        try:
            result = self.evaluate(expr)
            if result is None:
                return factor_name, None, expr, f"Warning: {factor_name} returned None"
            if _result_is_all_nan(result):
                return factor_name, None, expr, f"Warning: {factor_name} returned all NaN"
            return factor_name, _prepare_factor_result(result), expr, None
        except Exception as exc:
            return factor_name, None, expr, f"Error evaluating {factor_name}: {exc}"

    @staticmethod
    def _print_eval_error(err: str, raw_expr: str) -> None:
        print(err)
        if not err.startswith("Warning:"):
            print(f"Expression: {raw_expr}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Alpha101-style factor expression.")
    parser.add_argument("-f", "--file", type=str, required=False)
    parser.add_argument("expression", type=str, nargs="?", default=None)
    return parser.parse_args()


def main() -> None:
    from alpha101.config import get_config
    from alpha101.data.panel import build_wide_df
    from alpha101.factors.alpha_data import Alphas
    from alpha101.factors.operators import process_factor_wide_format

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
        print("Usage: python -m alpha101.factors.expression_engine <expression>")
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
