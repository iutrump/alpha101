from __future__ import annotations

import re
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from concurrent.futures import Executor
from typing import Dict

import pandas as pd
from tqdm import tqdm

from alpha101.config import get_config
from alpha101.data import FactorDataView
from alpha101.factors.expression import FastExpressionEngine
from alpha101.factors.expression.runtime import alpha_fields
from alpha101.factors.generation import FactorGenerator
from alpha101.factors.evaluation import forward_returns, periods_per_year, score_factor_search
from alpha101.factors.evaluation.scoring import _time_segment_slices
from alpha101.factors.operator_lib import process_factor_wide_format
from alpha101.factors.search.results import SearchResultStore
from alpha101.factors.search.strategies import genetic_search
from alpha101.factors.search.strategies import apply_pnl_redundancy_penalty
from alpha101.factors.search.worker import evaluate_search_item, init_search_worker
from alpha101.factors.validation import factor_pnl_series, validate_expressions_on_validation


class FactorSearchEngine:
    """Search Alpha101-style factor expressions against a wide OHLCV panel."""

    def __init__(
        self,
        wide_data: pd.DataFrame,
        output_dir: str = "factor_search_results",
        timeframe: str = "1d",
        n_quantiles: int = 5,
        forward_periods: int = 1,
        transaction_cost: float | None = None,
        segment_ratios: tuple[float, float, float] | list[float] | None = None,
        seed: int | None = None,
        exclude_fields: tuple[str, ...] | list[str] | None = None,
        seed_expressions: tuple[str, ...] | list[str] | None = None,
        max_complexity: float = 36.0,
        complexity_penalty: float = 0.0,
        diversity_penalty: float = 0.0,
        min_valid_sharpe: float | None = None,
        min_valid_ic_ir: float | None = None,
        validation_failure_penalty: float = 0.0,
        train_valid_gap_penalty: float = 0.0,
        validation_interval: int = 5,
        validation_time_folds: int = 4,
        validation_universe_folds: int = 3,
        validation_walk_forward_folds: int = 4,
        validation_extra_n_quantiles: tuple[int, ...] | list[int] = (10,),
        cv_failure_penalty: float = 0.5,
        pnl_dedupe_interval: int = 1,
        pnl_corr_threshold: float = 0.85,
        pnl_redundancy_penalty: float = 999.0,
    ):
        self.wide_data = wide_data
        self.alpha_obj = FactorDataView(wide_data)
        self.engine = FastExpressionEngine(self.alpha_obj)
        self.generator = FactorGenerator(seed=seed)
        self.exclude_fields = tuple(exclude_fields or ())
        if self.exclude_fields:
            self.generator.data_fields = [
                field for field in self.generator.data_fields if field not in set(self.exclude_fields)
            ]
            if not self.generator.data_fields:
                raise ValueError("exclude_fields removed all generator data fields")
        self.results = SearchResultStore(output_dir, timeframe)
        self.timeframe = timeframe
        self.n_quantiles = n_quantiles
        self.forward_periods = forward_periods
        self.annualization = periods_per_year(timeframe, forward_periods)
        self.seed = seed
        cfg = get_config()
        self.transaction_cost = float(cfg.round_trip_fee if transaction_cost is None else transaction_cost)
        raw_segment_ratios = cfg.search_segment_ratios if segment_ratios is None else segment_ratios
        self.segment_ratios = tuple(float(value) for value in raw_segment_ratios)
        self.evaluation_cache: dict[str, dict] = {}
        self.seen_expressions: set[str] = set()
        self.seed_expressions = tuple(seed_expressions or ())
        self.max_complexity = float(max_complexity)
        self.complexity_penalty = float(complexity_penalty)
        self.diversity_penalty = float(diversity_penalty)
        self.min_valid_sharpe = None if min_valid_sharpe is None else float(min_valid_sharpe)
        self.min_valid_ic_ir = None if min_valid_ic_ir is None else float(min_valid_ic_ir)
        self.validation_failure_penalty = float(validation_failure_penalty)
        self.train_valid_gap_penalty = float(train_valid_gap_penalty)
        self.validation_interval = int(validation_interval)
        self.validation_time_folds = int(validation_time_folds)
        self.validation_universe_folds = int(validation_universe_folds)
        self.validation_walk_forward_folds = int(validation_walk_forward_folds)
        self.validation_extra_n_quantiles = tuple(int(value) for value in validation_extra_n_quantiles)
        self.cv_failure_penalty = float(cv_failure_penalty)
        self.pnl_dedupe_interval = int(pnl_dedupe_interval)
        self.pnl_corr_threshold = float(pnl_corr_threshold)
        self.pnl_redundancy_penalty = float(pnl_redundancy_penalty)
        self.min_obs = 30
        self._metrics_executor: ProcessPoolExecutor | None = None
        self._pnl_dedupe_cache: dict[str, pd.Series] = {}

    def save_manifest(self, *, cli_args: dict | None = None) -> None:
        manifest = {
            "created_at": datetime.now().isoformat(),
            "git": _git_context(),
            "output_dir": str(self.results.output_dir),
            "seed": self.seed,
            "exclude_fields": self.exclude_fields,
            "seed_expressions": self.seed_expressions,
            "max_complexity": self.max_complexity,
            "complexity_penalty": self.complexity_penalty,
            "diversity_penalty": self.diversity_penalty,
            "min_valid_sharpe": self.min_valid_sharpe,
            "min_valid_ic_ir": self.min_valid_ic_ir,
            "validation_failure_penalty": self.validation_failure_penalty,
            "train_valid_gap_penalty": self.train_valid_gap_penalty,
            "validation_interval": self.validation_interval,
            "validation_time_folds": self.validation_time_folds,
            "validation_universe_folds": self.validation_universe_folds,
            "validation_walk_forward_folds": self.validation_walk_forward_folds,
            "validation_extra_n_quantiles": self.validation_extra_n_quantiles,
            "cv_failure_penalty": self.cv_failure_penalty,
            "pnl_dedupe_interval": self.pnl_dedupe_interval,
            "pnl_corr_threshold": self.pnl_corr_threshold,
            "pnl_redundancy_penalty": self.pnl_redundancy_penalty,
            "n_quantiles": self.n_quantiles,
            "forward_periods": self.forward_periods,
            "timeframe": self.timeframe,
            "annualization": self.annualization,
            "transaction_cost": self.transaction_cost,
            "segment_ratios": self.segment_ratios,
            "symbols": list(self.wide_data["close"].columns),
            "date_start": str(self.wide_data.index.min()),
            "date_end": str(self.wide_data.index.max()),
            "rows": int(len(self.wide_data)),
            "cli_args": cli_args or {},
        }
        self.results.save_manifest(manifest)

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

    def initial_population(self, population_size: int) -> list[str]:
        expressions: list[str] = []
        for raw_expr in self.seed_expressions:
            expr = self.normalize_expression(raw_expr)
            if expr in self.seen_expressions:
                continue
            self._validate_expression(expr)
            self.seen_expressions.add(expr)
            expressions.append(expr)
            if len(expressions) >= population_size:
                return expressions

        while len(expressions) < population_size:
            expressions.append(self.new_random_expression())
        return expressions

    def selection_fitness(self, metrics: dict) -> float:
        raw = float(metrics.get("fitness", -999.0))
        if metrics.get("status") != "success":
            return raw
        complexity = float(metrics.get("complexity_score", 0.0) or 0.0)
        valid_sharpe = float(metrics.get("valid_sharpe", 0.0) or 0.0)
        valid_ic_ir = float(metrics.get("valid_ic_ir", 0.0) or 0.0)
        train_sharpe = float(metrics.get("train_sharpe", 0.0) or 0.0)
        train_valid_gap = max(0.0, train_sharpe - valid_sharpe)
        validation_pass = True
        if self.min_valid_sharpe is not None and valid_sharpe < self.min_valid_sharpe:
            validation_pass = False
        if self.min_valid_ic_ir is not None and valid_ic_ir < self.min_valid_ic_ir:
            validation_pass = False
        metrics["train_valid_sharpe_gap"] = train_valid_gap
        metrics["validation_pass"] = validation_pass
        fitness = raw - self.complexity_penalty * complexity
        fitness -= self.train_valid_gap_penalty * train_valid_gap
        if not validation_pass:
            fitness -= self.validation_failure_penalty
        return fitness

    def validate_elite(self, elite: list[dict], generation: int) -> int:
        if self.validation_interval <= 0 or generation % self.validation_interval != 0:
            return 0
        candidates = [
            individual
            for individual in elite
            if individual.get("metrics") and individual["metrics"].get("status") == "success"
        ]
        if not candidates:
            return 0
        expressions = [individual["expression"] for individual in candidates]
        manifest = {
            "forward_periods": self.forward_periods,
            "transaction_cost": self.transaction_cost,
            "segment_ratios": self.segment_ratios,
        }
        validation_metrics = validate_expressions_on_validation(
            expressions,
            self.wide_data,
            manifest=manifest,
            time_folds=self.validation_time_folds,
            universe_folds=self.validation_universe_folds,
            walk_forward_folds=self.validation_walk_forward_folds,
            n_quantiles=self.n_quantiles,
            extra_n_quantiles=self.validation_extra_n_quantiles,
            progress_bar=True,
        )
        for individual in candidates:
            metrics = individual["metrics"]
            expr_metrics = validation_metrics.get(individual["expression"], {})
            metrics.update(expr_metrics)
            penalty = self.cv_failure_penalty * self._cv_failure_count(expr_metrics)
            metrics["cv_failure_penalty"] = penalty
            base_fitness = individual.get("base_fitness", individual.get("fitness", -999.0))
            individual["fitness"] = float(base_fitness) - penalty
            metrics["selection_fitness"] = individual["fitness"]
        return len(candidates)

    def _cv_failure_count(self, metrics: dict) -> int:
        count = 0
        if metrics.get("cv_pass") is False:
            count += 1
        for quantile in self.validation_extra_n_quantiles:
            if metrics.get(f"cv_nq{int(quantile)}_pass") is False:
                count += 1
        return count

    def dedupe_population_by_pnl(self, population: list[dict], generation: int) -> dict[str, int]:
        if self.pnl_dedupe_interval <= 0 or generation % self.pnl_dedupe_interval != 0:
            return {"checked": 0, "kept": 0, "redundant": 0}
        candidates = [
            individual
            for individual in population
            if individual.get("metrics") and individual["metrics"].get("status") == "success"
        ]
        if len(candidates) <= 1:
            return {"checked": len(candidates), "kept": len(candidates), "redundant": 0}

        pnl_values: dict[str, pd.Series] = {}
        for individual in tqdm(candidates, desc=f"PNL dedupe generation {generation}"):
            expression = individual["expression"]
            try:
                pnl_values[expression] = self._search_visible_pnl(expression)
            except Exception as exc:
                individual["metrics"]["search_pnl_dedupe_error"] = str(exc)
        return apply_pnl_redundancy_penalty(
            population,
            pnl_values,
            threshold=self.pnl_corr_threshold,
            penalty=self.pnl_redundancy_penalty,
        )

    def _search_visible_pnl(self, expression: str) -> pd.Series:
        cached = self._pnl_dedupe_cache.get(expression)
        if cached is not None:
            return cached
        raw_factor = self.engine.evaluate(expression)
        factor = process_factor_wide_format(raw_factor).reindex(
            index=self.wide_data.index,
            columns=self.wide_data["close"].columns,
        )
        target = forward_returns(self.wide_data["close"], periods=self.forward_periods)
        slices = _time_segment_slices(len(self.wide_data), self.segment_ratios)
        visible_slice = slice(slices["train"].start, slices["valid"].stop)
        factor = factor.iloc[visible_slice]
        target = target.iloc[visible_slice]
        pnl = factor_pnl_series(
            factor,
            target,
            n_quantiles=self.n_quantiles,
            transaction_cost=self.transaction_cost,
        )
        self._pnl_dedupe_cache[expression] = pnl
        return pnl

    def _cache_search_visible_pnl(self, expression: str, metrics: dict) -> None:
        values = metrics.pop("_search_visible_pnl_values", None)
        if values is None:
            return
        self._pnl_dedupe_cache[expression] = pd.Series(values, dtype=float)

    def expression_family(self, expr: str) -> str:
        norm = self.normalize_expression(expr)
        norm = re.sub(r"\b\d+(?:\.\d+)?\b", "#", norm)
        return re.sub(r"\s+", "", norm)

    def evaluate_factor(self, factor_name: str, factor_expr: str) -> Dict:
        try:
            factor_expr = self.normalize_expression(factor_expr)
            cached = self.evaluation_cache.get(factor_expr)
            if cached is not None:
                return self._clone_metrics(cached, factor_name, factor_expr)

            self._validate_expression(factor_expr)
            result = self._evaluate_expression(factor_expr)
            if result is None or self._result_is_all_nan(result):
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
                self._cache_search_visible_pnl(factor_expr, metrics)

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
        fields = self._expression_fields()
        close = self.wide_data["close"]
        items = list(named_expressions.items())

        if backend == "serial" or len(items) <= 1:
            init_search_worker(
                fields,
                close,
                self.n_quantiles,
                self.forward_periods,
                self.min_obs,
                self.segment_ratios,
                self.transaction_cost,
                self.annualization,
            )
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
                initargs=(
                    fields,
                    close,
                    self.n_quantiles,
                    self.forward_periods,
                    self.min_obs,
                    self.segment_ratios,
                    self.transaction_cost,
                    self.annualization,
                ),
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
        return score_factor_search(
            factor_df,
            self.wide_data["close"],
            n_quantiles=self.n_quantiles,
            forward_periods=self.forward_periods,
            preprocess=preprocess,
            min_segment_obs=self.min_obs,
            segment_ratios=self.segment_ratios,
            transaction_cost=self.transaction_cost,
            annualization=self.annualization,
        )

    def _evaluate_expression(self, factor_expr: str):
        return self.engine.evaluate(factor_expr)

    def _expression_fields(self) -> dict:
        return alpha_fields(self.alpha_obj)

    def _result_is_all_nan(self, result) -> bool:
        return result.isnull().all().all()

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

        fields = self._expression_fields()
        close = self.wide_data["close"]
        worker_count = max_workers if max_workers is not None else _cpu_count()
        self._metrics_executor = ProcessPoolExecutor(
            max_workers=max(1, int(worker_count)),
            initializer=init_search_worker,
            initargs=(
                fields,
                close,
                self.n_quantiles,
                self.forward_periods,
                self.min_obs,
                self.segment_ratios,
                self.transaction_cost,
            ),
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


def _git_context() -> dict[str, str | None]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "branch": run_git(["branch", "--show-current"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": run_git(["status", "--short"]),
    }
