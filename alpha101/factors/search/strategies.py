from __future__ import annotations

import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def genetic_search(
    search_engine,
    *,
    population_size: int = 50,
    n_generations: int = 10,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.5,
    backend: str = "auto",
    max_workers: int | None = None,
    profile: bool = False,
) -> dict | None:
    search_started = time.perf_counter()
    population = [
        {"expression": expression, "fitness": None, "metrics": None}
        for expression in search_engine.initial_population(population_size)
    ]
    init_elapsed = time.perf_counter() - search_started
    best_overall = None
    for gen in range(n_generations):
        gen_started = time.perf_counter()
        pending_started = time.perf_counter()
        pending = {
            f"gen{gen}_individual_{idx:04d}": individual["expression"]
            for idx, individual in enumerate(population)
            if individual["fitness"] is None
        }
        pending_elapsed = time.perf_counter() - pending_started

        evaluate_started = time.perf_counter()
        batch_metrics = search_engine.evaluate_factors_batch(
            pending,
            backend=backend,
            max_workers=max_workers,
            progress_bar=True,
        )
        evaluate_elapsed = time.perf_counter() - evaluate_started

        scoring_started = time.perf_counter()
        for idx, individual in enumerate(tqdm(population, desc=f"Scoring generation {gen + 1}")):
            if individual["fitness"] is not None:
                continue
            metrics = batch_metrics[f"gen{gen}_individual_{idx:04d}"]
            individual["metrics"] = metrics
            individual["base_fitness"] = search_engine.selection_fitness(metrics)
            individual["fitness"] = individual["base_fitness"]
            metrics["selection_fitness"] = individual["base_fitness"]
        scoring_elapsed = time.perf_counter() - scoring_started

        rank_started = time.perf_counter()
        apply_diversity_penalty(population, search_engine)
        population.sort(key=lambda x: x["fitness"], reverse=True)
        elite_count = max(1, population_size // 5)
        validation_started = time.perf_counter()
        validated_elite = search_engine.validate_elite(population[:elite_count], gen + 1)
        if validated_elite:
            population.sort(key=lambda x: x["fitness"], reverse=True)
        validation_elapsed = time.perf_counter() - validation_started
        pnl_dedupe_started = time.perf_counter()
        pnl_dedupe_stats = search_engine.dedupe_population_by_pnl(population, gen + 1)
        if pnl_dedupe_stats["checked"]:
            population.sort(key=lambda x: x["fitness"], reverse=True)
        pnl_dedupe_elapsed = time.perf_counter() - pnl_dedupe_started
        if best_overall is None or population[0]["fitness"] > best_overall["fitness"]:
            best_overall = population[0].copy()
        rank_elapsed = time.perf_counter() - rank_started

        save_started = time.perf_counter()
        search_engine.save_batch_results(
            [ind["metrics"] for ind in population if ind["metrics"]],
            f"genetic_gen_{gen + 1}",
        )
        save_elapsed = time.perf_counter() - save_started

        reproduce_started = time.perf_counter()
        new_population = population[:elite_count]
        next_seen = {ind["expression"] for ind in new_population}
        while len(new_population) < population_size:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            if np.random.random() < crossover_rate:
                child_expr = search_engine.generator.crossover_expressions(
                    parent1["expression"],
                    parent2["expression"],
                )
            else:
                child_expr = parent1["expression"]
            if np.random.random() < mutation_rate:
                child_expr = search_engine.generator.mutate_expression(child_expr)
            child_expr = search_engine.normalize_expression(child_expr)
            if child_expr in next_seen:
                child_expr = search_engine.new_random_expression()
            next_seen.add(child_expr)
            new_population.append({"expression": child_expr, "fitness": None, "metrics": None})
        population = new_population[:population_size]
        reproduce_elapsed = time.perf_counter() - reproduce_started
        gen_elapsed = time.perf_counter() - gen_started

        if profile:
            print(
                "[profile] "
                f"generation={gen + 1} "
                f"pending={pending_elapsed:.3f}s "
                f"evaluate={evaluate_elapsed:.3f}s "
                f"score_assign={scoring_elapsed:.3f}s "
                f"rank={rank_elapsed:.3f}s "
                f"elite_validation={validation_elapsed:.3f}s "
                f"pnl_dedupe={pnl_dedupe_elapsed:.3f}s "
                f"save={save_elapsed:.3f}s "
                f"reproduce={reproduce_elapsed:.3f}s "
                f"total={gen_elapsed:.3f}s "
                f"evaluated={len(pending)} "
                f"validated_elite={validated_elite} "
                f"pnl_dedupe_checked={pnl_dedupe_stats['checked']} "
                f"pnl_dedupe_redundant={pnl_dedupe_stats['redundant']} "
                f"cache_size={len(search_engine.evaluation_cache)}"
            )

    if profile:
        print(f"[profile] init_population={init_elapsed:.3f}s total_search={time.perf_counter() - search_started:.3f}s")
    return best_overall


def tournament_select(population: list[dict], tournament_size: int = 3) -> dict:
    tournament = np.random.choice(population, size=min(tournament_size, len(population)), replace=False)
    return max(tournament, key=lambda x: x["fitness"])


def apply_diversity_penalty(population: list[dict], search_engine) -> None:
    penalty = getattr(search_engine, "diversity_penalty", 0.0)
    if penalty <= 0:
        return

    family_counts: dict[str, int] = {}
    for individual in population:
        family = search_engine.expression_family(individual["expression"])
        family_counts[family] = family_counts.get(family, 0) + 1

    for individual in population:
        metrics = individual.get("metrics")
        if metrics is None:
            metrics = {}
            individual["metrics"] = metrics
        family = search_engine.expression_family(individual["expression"])
        duplicate_count = max(0, family_counts.get(family, 1) - 1)
        if individual.get("fitness") is None:
            continue
        base_fitness = individual.get("base_fitness")
        if base_fitness is None:
            base_fitness = individual["fitness"]
            individual["base_fitness"] = base_fitness
        adjusted = float(base_fitness) - penalty * duplicate_count
        individual["fitness"] = adjusted
        metrics["selection_fitness"] = adjusted
        metrics["family_duplicate_count"] = duplicate_count


def apply_pnl_redundancy_penalty(
    population: list[dict],
    pnl_values: dict[str, pd.Series],
    *,
    threshold: float,
    penalty: float,
) -> dict[str, int]:
    candidates = [
        individual
        for individual in population
        if individual.get("metrics")
        and individual["metrics"].get("status") == "success"
        and individual.get("expression") in pnl_values
        and individual.get("fitness") is not None
    ]
    candidates.sort(key=lambda individual: individual["fitness"], reverse=True)

    kept: list[dict] = []
    redundant = 0
    for individual in candidates:
        metrics = individual["metrics"]
        expression = individual["expression"]
        pnl = pnl_values[expression]
        max_corr = 0.0
        nearest_expression = ""
        for kept_individual in kept:
            kept_expression = kept_individual["expression"]
            corr = _series_corr(pnl, pnl_values[kept_expression])
            if abs(corr) > abs(max_corr):
                max_corr = corr
                nearest_expression = kept_expression

        is_redundant = bool(nearest_expression and abs(max_corr) >= threshold)
        metrics["search_max_pnl_corr"] = float(max_corr)
        metrics["search_nearest_pnl_corr_expression"] = nearest_expression
        metrics["search_redundant_by_pnl"] = is_redundant
        metrics["search_pnl_corr_threshold"] = float(threshold)
        if is_redundant:
            redundant += 1
            metrics["search_pnl_redundancy_penalty"] = float(penalty)
            metrics["pre_pnl_dedupe_fitness"] = float(individual["fitness"])
            individual["fitness"] = float(individual["fitness"]) - float(penalty)
            metrics["selection_fitness"] = individual["fitness"]
        else:
            metrics["search_pnl_redundancy_penalty"] = 0.0
            kept.append(individual)

    return {"checked": len(candidates), "kept": len(kept), "redundant": redundant}


def _series_corr(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(aligned) < 3:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return float(corr) if np.isfinite(corr) else 0.0
