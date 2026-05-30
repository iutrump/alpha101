from __future__ import annotations

import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from concurrent.futures import Executor
from typing import Dict

import pandas as pd
from tqdm import tqdm

from alpha101.data.alpha_view import Alphas
from alpha101.factors.expression import FastExpressionEngine
from alpha101.factors.expression.runtime import alpha_fields
from alpha101.factors.generation import FactorGenerator
from alpha101.factors.evaluation import score_factor_cross_section
from alpha101.factors.search.results import SearchResultStore
from alpha101.factors.search.strategies import genetic_search
from alpha101.factors.search.worker import evaluate_search_item, init_search_worker


class FactorSearchEngine:
    """Search Alpha101-style factor expressions against a wide OHLCV panel."""

    def __init__(
        self,
        wide_data: pd.DataFrame,
        output_dir: str = "factor_search_results",
        timeframe: str = "1d",
        n_quantiles: int = 5,
        forward_periods: int = 1,
        expression_backend: str = "pandas",
    ):
        if expression_backend not in {"pandas", "polars"}:
            raise ValueError("expression_backend must be either 'pandas' or 'polars'")
        if expression_backend == "polars":
            from alpha101.factors.operator_lib.polars.utils import require_polars

            require_polars()
        self.wide_data = wide_data
        self.alpha_obj = Alphas(wide_data)
        self.engine = FastExpressionEngine(self.alpha_obj)
        self.generator = FactorGenerator()
        self.results = SearchResultStore(output_dir, timeframe)
        self.n_quantiles = n_quantiles
        self.forward_periods = forward_periods
        self.expression_backend = expression_backend
        self.evaluation_cache: dict[str, dict] = {}
        self.seen_expressions: set[str] = set()
        self.max_complexity = 36.0
        self.min_obs = 30
        self._metrics_executor: ProcessPoolExecutor | None = None

    def normalize_expression(self, expr: str) -> str:
        try:
            root = self.generator._parse_expr_to_ast(expr)
            root = self.generator._sanitize_ast(root)
            root = self.generator._simplify_ast(root)
            return self.generator._ast_to_string(root)
        except Exception:
            return re.sub(r"\s+", "", expr)

    def new_random_expression(self, max_attempts: int = 50) -> str:
        for _ in range(max_attempts):
            expr = self.normalize_expression(self.generator.generate_random_factor())
            if expr not in self.seen_expressions:
                self.seen_expressions.add(expr)
                return expr
        expr = self.normalize_expression(self.generator.generate_random_factor())
        self.seen_expressions.add(expr)
        return expr

    def evaluate_factor(self, factor_name: str, factor_expr: str) -> Dict:
        try:
            factor_expr = self.normalize_expression(factor_expr)
            cached = self.evaluation_cache.get(factor_expr)
            if cached is not None:
                return self._clone_metrics(cached, factor_name, factor_expr)

            self._validate_expression(factor_expr)
            result = self.engine.evaluate(factor_expr)
            if result is None or result.isnull().all().all():
                raise ValueError("All NaN result")
            result.index.name = "date"
            result.columns.name = "symbol"

            metrics = self._score_factor(result)
            if metrics["obs_count"] < self.min_obs:
                raise ValueError(f"Insufficient observations: {metrics['obs_count']} < {self.min_obs}")

            complexity = self.generator.calculate_complexity(factor_expr)
            metrics.update(
                {
                    "factor_name": factor_name,
                    "expression": factor_expr,
                    "status": "success",
                    "complexity_score": complexity["complexity_score"],
                    "expr_max_depth": complexity["max_depth"],
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.evaluation_cache[factor_expr] = metrics
            return metrics
        except Exception as exc:
            failed = {
                "factor_name": factor_name,
                "expression": factor_expr,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "fitness": -999.0,
            }
            self.evaluation_cache[self.normalize_expression(factor_expr)] = failed
            return failed

    def evaluate_factors_batch(
        self,
        named_expressions: dict[str, str],
        *,
        backend: str = "auto",
        max_workers: int | None = None,
        progress_bar: bool = True,
    ) -> dict[str, Dict]:
        results: dict[str, Dict] = {}
        to_evaluate: dict[str, str] = {}

        for factor_name, factor_expr in named_expressions.items():
            norm_expr = self.normalize_expression(factor_expr)
            cached = self.evaluation_cache.get(norm_expr)
            if cached is not None:
                results[factor_name] = self._clone_metrics(cached, factor_name, norm_expr)
                continue
            try:
                self._validate_expression(norm_expr)
            except Exception as exc:
                failed = self._failed_metrics(factor_name, norm_expr, exc)
                self.evaluation_cache[norm_expr] = failed
                results[factor_name] = failed
                continue
            to_evaluate[factor_name] = norm_expr

        if not to_evaluate:
            return results

        resolved_backend = self._resolve_batch_backend(backend, len(to_evaluate), max_workers)
        batch_results = self._evaluate_metrics_batch(
            to_evaluate,
            backend=resolved_backend,
            max_workers=max_workers,
            progress_bar=progress_bar,
        )

        for factor_name, factor_expr in to_evaluate.items():
            try:
                metrics, error, error_traceback = batch_results[factor_name]
                if error:
                    raise RuntimeError(error_traceback or error)
                if metrics is None:
                    raise ValueError("Evaluation returned no metrics")

                complexity = self.generator.calculate_complexity(factor_expr)
                metrics.update(
                    {
                        "factor_name": factor_name,
                        "expression": factor_expr,
                        "status": "success",
                        "complexity_score": complexity["complexity_score"],
                        "expr_max_depth": complexity["max_depth"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                self.evaluation_cache[factor_expr] = metrics
                results[factor_name] = metrics
            except Exception as exc:
                failed = self._failed_metrics(factor_name, factor_expr, exc)
                self.evaluation_cache[factor_expr] = failed
                results[factor_name] = failed

        return results

    def _evaluate_metrics_batch(
        self,
        named_expressions: dict[str, str],
        *,
        backend: str,
        max_workers: int | None,
        progress_bar: bool,
    ) -> dict[str, tuple[dict | None, str | None, str | None]]:
        fields = alpha_fields(self.alpha_obj)
        close = self.wide_data["close"]
        items = list(named_expressions.items())

        if backend == "serial" or len(items) <= 1:
            init_search_worker(fields, close, self.n_quantiles, self.forward_periods, self.min_obs, self.expression_backend)
            iterator = tqdm(items, desc="Evaluating factors") if progress_bar else items
            return {
                name: (metrics, error, error_traceback)
                for name, metrics, error, error_traceback in (evaluate_search_item(item) for item in iterator)
            }

        if max_workers is None:
            max_workers = max(1, min(len(items), _cpu_count()))

        results: dict[str, tuple[dict | None, str | None, str | None]] = {}
        pbar = tqdm(total=len(items), desc="Evaluating factors (metrics)") if progress_bar else None
        if self._metrics_executor is not None:
            self._collect_metric_results(self._metrics_executor, items, results, pbar)
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_search_worker,
                initargs=(fields, close, self.n_quantiles, self.forward_periods, self.min_obs, self.expression_backend),
            ) as executor:
                self._collect_metric_results(executor, items, results, pbar)
        if pbar is not None:
            pbar.close()
        return results

    def genetic_search(
        self,
        population_size: int = 50,
        n_generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        backend: str = "auto",
        max_workers: int | None = None,
        profile: bool = False,
    ) -> dict | None:
        with self._metrics_worker_pool(backend=backend, max_workers=max_workers):
            return genetic_search(
                self,
                population_size=population_size,
                n_generations=n_generations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                backend=backend,
                max_workers=max_workers,
                profile=profile,
            )

    def save_batch_results(self, results: list[Dict], batch_name: str) -> None:
        self.results.save_batch(results, batch_name)

    def summarize_results(self) -> None:
        self.results.summarize(self.normalize_expression)

    def _validate_expression(self, factor_expr: str) -> None:
        is_valid, reason = self.generator.is_semantically_valid(factor_expr)
        if not is_valid:
            raise ValueError(f"Semantic invalid: {reason}")
        if not self.generator.filter_by_complexity(factor_expr, max_complexity=self.max_complexity):
            raise ValueError(f"Complexity too high (> {self.max_complexity})")

    def _score_factor(self, factor_df: pd.DataFrame, *, preprocess: bool = True) -> dict:
        return score_factor_cross_section(
            factor_df,
            self.wide_data["close"],
            n_quantiles=self.n_quantiles,
            forward_periods=self.forward_periods,
            preprocess=preprocess,
        )

    @staticmethod
    def _clone_metrics(metrics: dict, factor_name: str, expression: str) -> dict:
        out = dict(metrics)
        out["factor_name"] = factor_name
        out["expression"] = expression
        return out

    @staticmethod
    def _resolve_batch_backend(backend: str, n_items: int, max_workers: int | None) -> str:
        if backend in {"process", "serial"}:
            return backend
        if max_workers == 1 or n_items <= 1:
            return "serial"
        if sys.platform == "win32":
            return "serial"
        return "process"

    @contextmanager
    def _metrics_worker_pool(self, *, backend: str, max_workers: int | None):
        resolved_backend = self._resolve_batch_backend(backend, 2, max_workers)
        if resolved_backend != "process":
            yield
            return

        fields = alpha_fields(self.alpha_obj)
        close = self.wide_data["close"]
        worker_count = max_workers if max_workers is not None else _cpu_count()
        self._metrics_executor = ProcessPoolExecutor(
            max_workers=max(1, int(worker_count)),
            initializer=init_search_worker,
            initargs=(fields, close, self.n_quantiles, self.forward_periods, self.min_obs, self.expression_backend),
        )
        try:
            yield
        finally:
            self._metrics_executor.shutdown(wait=True)
            self._metrics_executor = None

    @staticmethod
    def _collect_metric_results(
        executor: Executor,
        items: list[tuple[str, str]],
        results: dict[str, tuple[dict | None, str | None, str | None]],
        pbar,
    ) -> None:
        future_map = {executor.submit(evaluate_search_item, item): item[0] for item in items}
        for future in as_completed(future_map):
            factor_name = future_map[future]
            try:
                name, metrics, error, error_traceback = future.result()
                results[name] = (metrics, error, error_traceback)
            except Exception:
                results[factor_name] = (None, "Worker failed", traceback.format_exc())
            if pbar is not None:
                pbar.update(1)

    @staticmethod
    def _failed_metrics(factor_name: str, factor_expr: str, exc: Exception) -> dict:
        return {
            "factor_name": factor_name,
            "expression": factor_expr,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "fitness": -999.0,
        }


def _cpu_count() -> int:
    try:
        import os

        return os.cpu_count() or 1
    except Exception:
        return 1
