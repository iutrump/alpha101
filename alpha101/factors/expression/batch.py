from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

import pandas as pd
from tqdm import tqdm

from alpha101.factors.expression.runtime import alpha_fields
from alpha101.factors.expression.worker import (
    eval_one_worker,
    init_worker,
    prepare_factor_result,
    result_is_all_nan,
)


def evaluate_batch(
    engine,
    expressions: Dict[str, str],
    *,
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
            name, result, raw_expr, err = eval_one_serial(engine, factor_name, expr)
            if err:
                print_eval_error(err, raw_expr)
                continue
            results[name] = result
    else:
        if max_workers is None:
            max_workers = max(1, min(len(items), os.cpu_count() or 1))
        pbar = tqdm(total=len(items), desc="Calculating factors (parallel)") if progress_bar else None
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(alpha_fields(engine.alpha),),
        ) as executor:
            future_map = {executor.submit(eval_one_worker, item): item for item in items}
            for future in as_completed(future_map):
                name, raw_expr = future_map[future]
                try:
                    name, result, raw_expr, err = future.result()
                except Exception as exc:
                    err = f"Error evaluating {name}: {exc}"
                    result = None
                if err:
                    print_eval_error(err, raw_expr)
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


def eval_one_serial(engine, factor_name: str, expr: str):
    try:
        result = engine.evaluate(expr)
        if result is None:
            return factor_name, None, expr, f"Warning: {factor_name} returned None"
        if result_is_all_nan(result):
            return factor_name, None, expr, f"Warning: {factor_name} returned all NaN"
        return factor_name, prepare_factor_result(result), expr, None
    except Exception as exc:
        return factor_name, None, expr, f"Error evaluating {factor_name}: {exc}"


def print_eval_error(err: str, raw_expr: str) -> None:
    print(err)
    if not err.startswith("Warning:"):
        print(f"Expression: {raw_expr}")
