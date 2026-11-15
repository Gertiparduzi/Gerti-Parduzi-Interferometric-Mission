"""Real-coded GA with elitism and optional satellite-coherent blend crossover.

Tournament selection, arithmetic blending and Gaussian mutation are standard
ideas. The implementation is independent. For ESA, a single blend coefficient
is shared by a satellite's six components, respecting component-block storage.
This grouping is a design hypothesis, not a claim of a new scientific method.
"""

from __future__ import annotations

import numpy as np

from src.optimization import (
    Bounds, EvaluationBudget, Objective, OptimizationResult, Vector,
    integer_parameter, real_parameter, reflect_unit,
)


def tournament_select(scores: Vector, size: int, generator: np.random.Generator) -> int:
    """Sample without replacement; choose the smallest objective in the sample."""
    candidates = generator.choice(len(scores), size=size, replace=False)
    return int(candidates[np.argmin(scores[candidates])])


def blend_crossover(
    first: Vector, second: Vector, alpha: float, generator: np.random.Generator,
    n_satellites: int | None = None,
) -> Vector:
    """Blend unit-coordinate parents; reflect extrapolated children into bounds.

    With N satellites, weights are [w_1..w_N] repeated six times. Thus indices
    i, N+i, ..., 5N+i share a weight. With no grouping, weights are per gene.
    """
    size = len(first) if n_satellites is None else n_satellites
    weights = generator.uniform(-alpha, 1.0 + alpha, size=size)
    if n_satellites is not None:
        weights = np.tile(weights, 6)
    return reflect_unit(weights * first + (1.0 - weights) * second)


def gaussian_mutation(
    child: Vector, probability: float, sigma: float, generator: np.random.Generator,
) -> Vector:
    """Per-gene mutation; sigma is a fraction of a variable's entire interval."""
    mutated = child.copy()
    if probability == 0.0 or sigma == 0.0:
        return mutated
    selected = generator.random(len(child)) < probability
    mutated[selected] += generator.normal(0.0, sigma, size=int(np.count_nonzero(selected)))
    return reflect_unit(mutated)


def genetic_algorithm(
    objective: Objective,
    bounds: Bounds,
    max_evaluations: int,
    *,
    seed: int = 0,
    population_size: int = 40,
    tournament_size: int = 2,
    crossover_rate: float = 0.9,
    crossover_alpha: float = 0.25,
    mutation_probability: float | None = None,
    mutation_sigma: float = 0.05,
    elite_count: int = 1,
    n_satellites: int | None = None,
) -> OptimizationResult:
    """Minimize with a generational GA and exactly the requested call budget.

    Initialization counts against the budget. A budget smaller than population
    size performs initialization only. Elites reuse stored fitness, never new
    calls. A final partial generation stops immediately at the budget limit.
    """
    seed = integer_parameter(seed, "seed", minimum=0)
    population_size = integer_parameter(population_size, "population_size", minimum=2)
    tournament_size = integer_parameter(tournament_size, "tournament_size")
    elite_count = integer_parameter(elite_count, "elite_count", minimum=0)
    if tournament_size > population_size:
        raise ValueError("tournament_size cannot exceed population_size.")
    if elite_count >= population_size:
        raise ValueError("elite_count must be smaller than population_size.")
    crossover_rate = real_parameter(crossover_rate, "crossover_rate", 0.0, 1.0)
    crossover_alpha = real_parameter(crossover_alpha, "crossover_alpha", 0.0, 1.0)
    mutation_sigma = real_parameter(mutation_sigma, "mutation_sigma", 0.0, 1.0)
    budget = EvaluationBudget(objective, bounds, max_evaluations)
    dimension = budget.space.dimension
    if mutation_probability is None:
        mutation_probability = 1.0 / dimension
    mutation_probability = real_parameter(mutation_probability, "mutation_probability", 0.0, 1.0)
    if n_satellites is not None:
        n_satellites = integer_parameter(n_satellites, "n_satellites")
        if dimension != 6 * n_satellites:
            raise ValueError("Satellite grouping requires exactly 6*N variables.")

    generator = np.random.default_rng(seed)
    active_size = min(population_size, budget.maximum)
    population = generator.random((active_size, dimension))
    scores = np.array([budget.evaluate(candidate) for candidate in population])

    while budget.remaining:
        order = np.argsort(scores, kind="stable")
        next_population = np.empty_like(population)
        next_scores = np.empty_like(scores)
        next_population[:elite_count] = population[order[:elite_count]]
        next_scores[:elite_count] = scores[order[:elite_count]]
        for index in range(elite_count, active_size):
            first = population[tournament_select(scores, tournament_size, generator)]
            second = population[tournament_select(scores, tournament_size, generator)]
            if generator.random() < crossover_rate:
                child = blend_crossover(first, second, crossover_alpha, generator, n_satellites)
            else:
                child = first.copy()
            child = gaussian_mutation(child, mutation_probability, mutation_sigma, generator)
            next_population[index] = child
            next_scores[index] = budget.evaluate(child)
            if not budget.remaining:
                return budget.finish("genetic_algorithm", seed)
        population, scores = next_population, next_scores
    return budget.finish("genetic_algorithm", seed)
