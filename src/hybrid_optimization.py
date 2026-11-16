"""SAGE: STM-Aware Genetic Exploration with feasible grid-scale local search.

One EvaluationBudget covers initialization, the genetic stage, every local
proposal, and diversification. Neutral moves can traverse discrete plateaus;
the immutable global incumbent is kept by EvaluationBudget.
"""

from __future__ import annotations

import numpy as np

from src.advanced_optimization import _evaluate, _ga_parameters, _ga_search, _setup
from src.optimization import integer_parameter, real_parameter


def sage_hybrid(objective, geometry, max_evaluations, *, seed=0, population_size=40,
                genetic_fraction=.2, local_restart_after=600):
    """Run GA then single/group-satellite local search, using exact counted calls.

    genetic_fraction=0 is an ablation: initialize one informed formation, then
    spend the remainder on local search. No old candidate is injected.
    Diversification after stagnation starts from the global best, replaces 10%
    of satellites and evaluates the perturbed formation once, even if worse.
    """
    genetic_fraction = real_parameter(genetic_fraction, "genetic_fraction", 0, 1)
    local_restart_after = integer_parameter(local_restart_after, "local_restart_after")
    parameters = _ga_parameters(population_size, 2, 1 if population_size == 2 else 2,
                                .9, "satellite", 2., .15, 20)
    budget, rng, seed = _setup(objective, geometry, max_evaluations, seed)
    genetic_budget = max(1, int(budget.maximum * genetic_fraction))
    _ga_search(budget, geometry, rng, genetic_budget, parameters)
    state = geometry.from_chromosome(budget.best_chromosome)
    current_score = budget.best_fitness
    stagnant = 0
    while budget.remaining:
        if stagnant >= local_restart_after:
            state = geometry.from_chromosome(budget.best_chromosome)
            count = max(1, geometry.n_satellites // 10)
            selected = rng.choice(geometry.n_satellites, count, replace=False)
            state[selected] = geometry.sample(rng, count)
            current_score = _evaluate(budget, geometry, state)
            stagnant = 0
            continue
        child = state.copy()
        count = min(geometry.n_satellites, 1 + int(rng.random() < .08))
        cells = float(rng.choice([.5, 1., 2., 4.]))
        for satellite in rng.choice(geometry.n_satellites, count, replace=False):
            if rng.random() < .15:
                child[satellite] = geometry.sample(rng, 1)[0]
            else:
                child[satellite] = geometry.move(child[satellite], rng, cells=cells)
        previous_best = budget.best_fitness
        fitness = _evaluate(budget, geometry, child)
        if fitness <= current_score:
            state, current_score = child, fitness
        stagnant = stagnant + 1 if budget.best_fitness >= previous_best else 0
    return budget.finish("sage_hybrid", seed)
