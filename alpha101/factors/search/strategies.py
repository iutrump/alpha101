from __future__ import annotations

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
) -> dict | None:
    population = [
        {"expression": search_engine.new_random_expression(), "fitness": None, "metrics": None}
        for _ in range(population_size)
    ]
    best_overall = None
    for gen in range(n_generations):
        pending = {
            f"gen{gen}_individual_{idx:04d}": individual["expression"]
            for idx, individual in enumerate(population)
            if individual["fitness"] is None
        }
        batch_metrics = search_engine.evaluate_factors_batch(
            pending,
            backend=backend,
            max_workers=max_workers,
            progress_bar=True,
        )
        for idx, individual in enumerate(tqdm(population, desc=f"Scoring generation {gen + 1}")):
            if individual["fitness"] is not None:
                continue
            metrics = batch_metrics[f"gen{gen}_individual_{idx:04d}"]
            individual["metrics"] = metrics
            individual["fitness"] = metrics.get("fitness", -999.0)

        population.sort(key=lambda x: x["fitness"], reverse=True)
        if best_overall is None or population[0]["fitness"] > best_overall["fitness"]:
            best_overall = population[0].copy()
        search_engine.save_batch_results(
            [ind["metrics"] for ind in population if ind["metrics"]],
            f"genetic_gen_{gen + 1}",
        )

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

    if best_overall:
        search_engine.save_batch_results([best_overall["metrics"]], "genetic_best_overall")
    return best_overall


def tournament_select(population: list[dict], tournament_size: int = 3) -> dict:
    tournament = np.random.choice(population, size=min(tournament_size, len(population)), replace=False)
    return max(tournament, key=lambda x: x["fitness"])
