from __future__ import annotations

import time

import numpy as np
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
                f"save={save_elapsed:.3f}s "
                f"reproduce={reproduce_elapsed:.3f}s "
                f"total={gen_elapsed:.3f}s "
                f"evaluated={len(pending)} "
                f"validated_elite={validated_elite} "
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
