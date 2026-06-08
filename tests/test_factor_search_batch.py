from __future__ import annotations

import numpy as np

from alpha101.factors.search import FactorSearchEngine
from alpha101.factors.search.strategies import apply_diversity_penalty
from tests.test_expression_runtime import make_wide_data


def test_factor_search_batch_evaluates_population():
    search_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2)
    search_engine.min_obs = 1

    results = search_engine.evaluate_factors_batch(
        {
            "alpha_a": "rank(ts_mean(close, 2))",
            "alpha_b": "zscore(ts_delta(vwap, 1))",
        },
        backend="serial",
        progress_bar=False,
    )

    assert set(results) == {"alpha_a", "alpha_b"}
    assert all(metrics["status"] == "success" for metrics in results.values())
    assert all("fitness" in metrics for metrics in results.values())


def test_factor_search_seed_reproduces_first_expression():
    first_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, seed=7)
    first_expression = first_engine.new_random_expression()

    second_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, seed=7)

    assert first_expression == second_engine.new_random_expression()


def test_factor_search_can_exclude_generation_fields():
    search_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, seed=11, exclude_fields=["cap", "funding"])

    assert "cap" not in search_engine.generator.data_fields
    assert "funding" not in search_engine.generator.data_fields


def test_factor_search_initial_population_uses_seed_expressions():
    search_engine = FactorSearchEngine(
        make_wide_data(),
        n_quantiles=2,
        seed=11,
        seed_expressions=["-ts_mean(zscore(volume), 2)"],
    )

    population = search_engine.initial_population(3)

    assert population[0] == "-ts_mean(zscore(volume), 2)"
    assert len(population) == 3


def test_factor_search_selection_fitness_penalizes_complexity():
    search_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, complexity_penalty=0.1)

    fitness = search_engine.selection_fitness(
        {
            "status": "success",
            "fitness": 2.0,
            "complexity_score": 3.0,
        }
    )

    assert np.isclose(fitness, 1.7)


def test_factor_search_selection_fitness_penalizes_validation_failure():
    search_engine = FactorSearchEngine(
        make_wide_data(),
        n_quantiles=2,
        min_valid_sharpe=1.0,
        min_valid_ic_ir=0.2,
        validation_failure_penalty=0.5,
        train_valid_gap_penalty=0.1,
    )
    metrics = {
        "status": "success",
        "fitness": 2.0,
        "train_sharpe": 2.0,
        "valid_sharpe": 0.5,
        "valid_ic_ir": 0.1,
    }

    fitness = search_engine.selection_fitness(metrics)

    assert np.isclose(fitness, 1.35)
    assert metrics["validation_pass"] is False
    assert np.isclose(metrics["train_valid_sharpe_gap"], 1.5)


def test_factor_search_selection_fitness_ignores_test_metrics():
    search_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, min_valid_sharpe=0.0)
    base_metrics = {
        "status": "success",
        "fitness": 2.0,
        "train_sharpe": 1.0,
        "valid_sharpe": 1.0,
        "valid_ic_ir": 0.5,
    }
    with_test_metrics = {**base_metrics, "test_sharpe": -100.0, "test_ic_ir": -100.0}

    assert np.isclose(
        search_engine.selection_fitness(base_metrics),
        search_engine.selection_fitness(with_test_metrics),
    )


def test_diversity_penalty_recomputes_from_base_fitness():
    search_engine = FactorSearchEngine(make_wide_data(), n_quantiles=2, diversity_penalty=0.25)
    population = [
        {
            "expression": "rank(ts_mean(close, 2))",
            "base_fitness": 2.0,
            "fitness": 2.0,
            "metrics": {},
        },
        {
            "expression": "rank(ts_mean(close, 3))",
            "base_fitness": 1.0,
            "fitness": 1.0,
            "metrics": {},
        },
    ]

    apply_diversity_penalty(population, search_engine)
    apply_diversity_penalty(population, search_engine)

    assert np.isclose(population[0]["fitness"], 1.75)
    assert np.isclose(population[1]["fitness"], 0.75)

    apply_diversity_penalty(population[:1], search_engine)

    assert np.isclose(population[0]["fitness"], 2.0)
    assert population[0]["metrics"]["family_duplicate_count"] == 0


def test_factor_search_validates_elite_on_interval():
    search_engine = FactorSearchEngine(
        make_wide_data(),
        n_quantiles=2,
        segment_ratios=[0.25, 0.50, 0.25],
        validation_interval=2,
        validation_time_folds=2,
        validation_universe_folds=2,
        validation_walk_forward_folds=2,
        validation_extra_n_quantiles=[3],
        cv_failure_penalty=0.25,
    )
    elite = [
        {
            "expression": "rank(close)",
            "base_fitness": 1.0,
            "fitness": 1.0,
            "metrics": {"status": "success", "fitness": 1.0},
        }
    ]

    assert search_engine.validate_elite(elite, generation=1) == 0
    assert "cv_time_pos_folds" not in elite[0]["metrics"]

    assert search_engine.validate_elite(elite, generation=2) == 1
    assert "cv_time_pos_folds" in elite[0]["metrics"]
    assert "cv_walk_forward_pos_folds" in elite[0]["metrics"]
    assert "cv_nq3_time_pos_folds" in elite[0]["metrics"]
    assert "test_sharpe" not in elite[0]["metrics"]
