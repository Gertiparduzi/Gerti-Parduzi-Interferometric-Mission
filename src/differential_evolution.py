"""Independent generational DE/rand/1/bin, with reflected box constraints.

Mathematical reference: Storn & Price (1997), doi:10.1023/A:1008202821328.
Donors always come from the frozen current generation. Equal-fitness trials
are accepted, permitting neutral movement on ESA's quantized plateaus.
"""

from __future__ import annotations

import numpy as np

from src.optimization import (
    Bounds, EvaluationBudget, Objective, OptimizationResult, Vector,
    integer_parameter, real_parameter, reflect_unit,
)


def donor_indices(target: int, population_size: int, generator: np.random.Generator) -> np.ndarray:
    """Return three distinct indexes excluding the target, without retry loops."""
    indexes = generator.choice(population_size - 1, size=3, replace=False)
    return indexes + (indexes >= target)


def binomial_crossover(
    target: Vector, mutant: Vector, crossover_rate: float, generator: np.random.Generator,
) -> Vector:
    """At least one coordinate is taken from the mutant, including when CR=0."""
    selected = generator.random(len(target)) < crossover_rate
    selected[generator.integers(len(target))] = True
    return np.where(selected, mutant, target)


def differential_evolution(
    objective: Objective,
    bounds: Bounds,
    max_evaluations: int,
    *,
    seed: int = 0,
    population_size: int = 40,
    F: float = 0.7,
    CR: float = 0.9,
) -> OptimizationResult:
    """Minimize with DE/rand/1/bin; initialization is part of the call budget.

    NP must be >=4; F must be in (0, 2] and CR in [0, 1]. If the positive
    budget is smaller than NP, return the best of that many initial samples.
    """
    seed = integer_parameter(seed, "seed", minimum=0)
    population_size = integer_parameter(population_size, "population_size", minimum=4)
    F = real_parameter(F, "F", 0.0, 2.0)
    CR = real_parameter(CR, "CR", 0.0, 1.0)
    if F == 0.0:
        raise ValueError("F must be strictly positive.")
    budget = EvaluationBudget(objective, bounds, max_evaluations)
    generator = np.random.default_rng(seed)
    active_size = min(population_size, budget.maximum)
    population = generator.random((active_size, budget.space.dimension))
    scores = np.array([budget.evaluate(candidate) for candidate in population])

    while budget.remaining:
        next_population = population.copy()
        next_scores = scores.copy()
        for target in range(active_size):
            a, b, c = donor_indices(target, active_size, generator)
            mutant = reflect_unit(population[a] + F * (population[b] - population[c]))
            trial = binomial_crossover(population[target], mutant, CR, generator)
            fitness = budget.evaluate(trial)
            if fitness <= scores[target]:
                next_population[target] = trial
                next_scores[target] = fitness
            if not budget.remaining:
                return budget.finish("differential_evolution", seed)
        population, scores = next_population, next_scores
    return budget.finish("differential_evolution", seed)
