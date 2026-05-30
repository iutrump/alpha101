from __future__ import annotations

import argparse
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from alpha101.config import get_config
from alpha101.data.panel import build_wide_df
from alpha101.factors.operators import Alphas
from alpha101.factors.factor_generator import FactorGenerator, FactorLibrary
from alpha101.factors.expression_engine import FastExpressionEngine
from alpha101.factors.evaluation import score_factor_cross_section


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
        self.output_dir = Path(output_dir) / timeframe
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_quantiles = n_quantiles
        self.forward_periods = forward_periods
        self.all_results: list[dict] = []
        self.evaluation_cache: dict[str, dict] = {}
        self.seen_expressions: set[str] = set()
        self.max_complexity = 36.0
        self.min_obs = 30

    def _normalize_expression(self, expr: str) -> str:
        try:
            root = self.generator._parse_expr_to_ast(expr)
            root = self.generator._sanitize_ast(root)
            root = self.generator._simplify_ast(root)
            return self.generator._ast_to_string(root)
        except Exception:
            return re.sub(r"\s+", "", expr)

    def _clone_metrics(self, metrics: dict, factor_name: str, expression: str) -> dict:
        out = dict(metrics)
        out["factor_name"] = factor_name
        out["expression"] = expression
        return out

    def _deduplicate_factor_pairs(self, factors: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        unique: list[tuple[str, str]] = []
        local_seen: set[str] = set()
        for name, expr in factors:
            norm_expr = self._normalize_expression(expr)
            if norm_expr in local_seen or norm_expr in self.seen_expressions:
                continue
            local_seen.add(norm_expr)
            self.seen_expressions.add(norm_expr)
            unique.append((name, norm_expr))
        return unique

    def _new_random_expression(self, max_attempts: int = 50) -> str:
        for _ in range(max_attempts):
            expr = self._normalize_expression(self.generator.generate_random_factor())
            if expr not in self.seen_expressions:
                self.seen_expressions.add(expr)
                return expr
        expr = self._normalize_expression(self.generator.generate_random_factor())
        self.seen_expressions.add(expr)
        return expr

    def _score_factor(self, factor_df: pd.DataFrame) -> dict:
        return score_factor_cross_section(
            factor_df,
            self.wide_data["close"],
            n_quantiles=self.n_quantiles,
            forward_periods=self.forward_periods,
        )

    def evaluate_factor(self, factor_name: str, factor_expr: str) -> Dict:
        try:
            factor_expr = self._normalize_expression(factor_expr)
            cached = self.evaluation_cache.get(factor_expr)
            if cached is not None:
                return self._clone_metrics(cached, factor_name, factor_expr)

            is_valid, reason = self.generator.is_semantically_valid(factor_expr)
            if not is_valid:
                raise ValueError(f"Semantic invalid: {reason}")
            if not self.generator.filter_by_complexity(factor_expr, max_complexity=self.max_complexity):
                raise ValueError(f"Complexity too high (> {self.max_complexity})")

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
            self.evaluation_cache[self._normalize_expression(factor_expr)] = failed
            return failed

    def random_search(self, n_factors: int = 1000, batch_size: int = 100) -> int:
        results = []
        successful_count = 0
        for i in tqdm(range(n_factors), desc="Random Search"):
            metrics = self.evaluate_factor(f"random_{i:04d}", self._new_random_expression())
            results.append(metrics)
            successful_count += int(metrics["status"] == "success")
            if (i + 1) % batch_size == 0:
                self._save_batch_results(results, f"random_batch_{i + 1}")
                results = []
        if results:
            self._save_batch_results(results, "random_batch_final")
        return successful_count

    def template_search(self, n_templates: int | None = None, batch_size: int = 10) -> int:
        templates = FactorLibrary.ALPHA101_PATTERNS
        if n_templates is not None:
            templates = templates[:n_templates]
        template_factors = self._deduplicate_factor_pairs(
            self.generator.generate_template_based_factors(templates)
        )

        results = []
        successful_count = 0
        for i, (factor_name, factor_expr) in enumerate(tqdm(template_factors, desc="Template Search")):
            metrics = self.evaluate_factor(factor_name, factor_expr)
            results.append(metrics)
            successful_count += int(metrics["status"] == "success")
            if (i + 1) % batch_size == 0:
                self._save_batch_results(results, f"template_batch_{i + 1}")
                results = []
        if results:
            self._save_batch_results(results, "template_batch_final")
        return successful_count

    def grid_search(self, batch_size: int = 10) -> int:
        grid_factors = self._deduplicate_factor_pairs(self.generator.generate_grid_search_factors())
        results = []
        successful_count = 0
        for i, (factor_name, factor_expr) in enumerate(tqdm(grid_factors, desc="Grid Search")):
            metrics = self.evaluate_factor(factor_name, factor_expr)
            results.append(metrics)
            successful_count += int(metrics["status"] == "success")
            if (i + 1) % batch_size == 0:
                self._save_batch_results(results, f"grid_batch_{i + 1}")
                results = []
        if results:
            self._save_batch_results(results, "grid_batch_final")
        return successful_count

    def genetic_search(
        self,
        population_size: int = 50,
        n_generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
    ) -> dict | None:
        population = [
            {"expression": self._new_random_expression(), "fitness": None, "metrics": None}
            for _ in range(population_size)
        ]
        best_overall = None
        for gen in range(n_generations):
            for idx, individual in enumerate(tqdm(population, desc=f"Generation {gen + 1}")):
                if individual["fitness"] is not None:
                    continue
                metrics = self.evaluate_factor(f"gen{gen}_individual_{idx:04d}", individual["expression"])
                individual["metrics"] = metrics
                individual["fitness"] = metrics.get("fitness", -999.0)

            population.sort(key=lambda x: x["fitness"], reverse=True)
            if best_overall is None or population[0]["fitness"] > best_overall["fitness"]:
                best_overall = population[0].copy()
            self._save_batch_results([ind["metrics"] for ind in population if ind["metrics"]], f"genetic_gen_{gen + 1}")

            elite_count = max(1, population_size // 5)
            new_population = population[:elite_count]
            next_seen = {ind["expression"] for ind in new_population}
            while len(new_population) < population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                if np.random.random() < crossover_rate:
                    child_expr = self.generator.crossover_expressions(parent1["expression"], parent2["expression"])
                else:
                    child_expr = parent1["expression"]
                if np.random.random() < mutation_rate:
                    child_expr = self.generator.mutate_expression(child_expr)
                child_expr = self._normalize_expression(child_expr)
                if child_expr in next_seen:
                    child_expr = self._new_random_expression()
                next_seen.add(child_expr)
                new_population.append({"expression": child_expr, "fitness": None, "metrics": None})
            population = new_population[:population_size]

        if best_overall:
            self._save_batch_results([best_overall["metrics"]], "genetic_best_overall")
        return best_overall

    def _tournament_select(self, population: list[dict], tournament_size: int = 3) -> dict:
        tournament = np.random.choice(population, size=min(tournament_size, len(population)), replace=False)
        return max(tournament, key=lambda x: x["fitness"])

    def _save_batch_results(self, results: List[Dict], batch_name: str) -> None:
        results = [r for r in results if r]
        if not results:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"{batch_name}_{timestamp}.csv"
        json_path = self.output_dir / f"{batch_name}_{timestamp}.json"
        pd.DataFrame(results).to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        self.all_results.extend(results)

    def summarize_results(self) -> None:
        if not self.all_results:
            print("No results to summarize")
            return
        df = pd.DataFrame(self.all_results)
        success_df = df[df["status"] == "success"].copy()
        if success_df.empty:
            print("No successful factors found")
            return
        success_df["expression_norm"] = success_df["expression"].map(self._normalize_expression)
        success_df = success_df.sort_values("fitness", ascending=False).drop_duplicates("expression_norm")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"summary_{timestamp}.csv"
        success_df.to_csv(summary_path, index=False)

        print(f"Total factors evaluated: {len(df)}")
        print(f"Successful factors: {len(success_df)}")
        print("Top factors:")
        cols = ["factor_name", "fitness", "sharpe_ratio", "ic_ir", "returns", "expression"]
        print(success_df[cols].head(10).to_string(index=False))
        print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Factor Search Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/alpha101*.json")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "template", "grid", "genetic", "all"])
    parser.add_argument("--n-factors", type=int, default=100)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--forward-periods", type=int, default=1)
    args = parser.parse_args()

    cfg = get_config(args.config)
    wide_data = build_wide_df(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
        buffer=cfg.pre_buffer_candles,
    )
    search_engine = FactorSearchEngine(
        wide_data,
        output_dir=args.output_dir or str(cfg.output_dir),
        timeframe=cfg.timeframe,
        n_quantiles=args.n_quantiles,
        forward_periods=args.forward_periods,
    )

    if args.strategy == "random":
        search_engine.random_search(n_factors=args.n_factors)
    elif args.strategy == "template":
        search_engine.template_search()
    elif args.strategy == "grid":
        search_engine.grid_search()
    elif args.strategy == "genetic":
        search_engine.genetic_search(population_size=args.population, n_generations=args.generations)
    elif args.strategy == "all":
        search_engine.random_search(n_factors=args.n_factors)
        search_engine.template_search()
        search_engine.grid_search()

    search_engine.summarize_results()


if __name__ == "__main__":
    main()
