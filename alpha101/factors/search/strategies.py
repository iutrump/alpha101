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
        {"expression": search_engine.new_random_expression(), "fitness": None, "metrics": None}
        for _ in range(population_size)
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
            individual["fitness"] = metrics.get("fitness", -999.0)
        scoring_elapsed = time.perf_counter() - scoring_started

        rank_started = time.perf_counter()
        population.sort(key=lambda x: x["fitness"], reverse=True)
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
        elite_count = max(1, population_size // 5)
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
                f"save={save_elapsed:.3f}s "
                f"reproduce={reproduce_elapsed:.3f}s "
                f"total={gen_elapsed:.3f}s "
                f"evaluated={len(pending)} "
                f"cache_size={len(search_engine.evaluation_cache)}"
            )

    if best_overall:
        save_started = time.perf_counter()
        search_engine.save_batch_results([best_overall["metrics"]], "genetic_best_overall")
        if profile:
            print(f"[profile] save_best={time.perf_counter() - save_started:.3f}s")
    if profile:
        print(f"[profile] init_population={init_elapsed:.3f}s total_search={time.perf_counter() - search_started:.3f}s")
    return best_overall


def tournament_select(population: list[dict], tournament_size: int = 3) -> dict:
    tournament = np.random.choice(population, size=min(tournament_size, len(population)), replace=False)
    return max(tournament, key=lambda x: x["fitness"])
