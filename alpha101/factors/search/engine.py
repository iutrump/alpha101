from __future__ import annotations

import re
import traceback
from datetime import datetime
from typing import Dict

import pandas as pd

from alpha101.factors.alpha_data import Alphas
from alpha101.factors.evaluation import score_factor_cross_section
from alpha101.factors.expression_engine import FastExpressionEngine
from alpha101.factors.generation import FactorGenerator
from alpha101.factors.search.results import SearchResultStore
from alpha101.factors.search.strategies import genetic_search


class FactorSearchEngine:
    """Search Alpha101-style factor expressions against a wide OHLCV panel."""

    def __init__(
        self,
        wide_data: pd.DataFrame,
        output_dir: str = "factor_search_results",
        timeframe: str = "1d",
        n_quantiles: int = 5,
        forward_periods: int = 1,
    ):
        self.wide_data = wide_data
        self.alpha_obj = Alphas(wide_data)
        self.engine = FastExpressionEngine(self.alpha_obj)
        self.generator = FactorGenerator()
        self.results = SearchResultStore(output_dir, timeframe)
        self.n_quantiles = n_quantiles
        self.forward_periods = forward_periods
        self.evaluation_cache: dict[str, dict] = {}
        self.seen_expressions: set[str] = set()
        self.max_complexity = 36.0
        self.min_obs = 30

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

    def genetic_search(
        self,
        population_size: int = 50,
        n_generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
    ) -> dict | None:
        return genetic_search(
            self,
            population_size=population_size,
            n_generations=n_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
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

    def _score_factor(self, factor_df: pd.DataFrame) -> dict:
        return score_factor_cross_section(
            factor_df,
            self.wide_data["close"],
            n_quantiles=self.n_quantiles,
            forward_periods=self.forward_periods,
        )

    @staticmethod
    def _clone_metrics(metrics: dict, factor_name: str, expression: str) -> dict:
        out = dict(metrics)
        out["factor_name"] = factor_name
        out["expression"] = expression
        return out
