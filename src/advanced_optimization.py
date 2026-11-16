"""Independent, budgeted STM-informed random search, GA and DE.

The baseline implementations are unchanged. Every selection decision uses the
provided scalar objective through the existing EvaluationBudget. Geometry is
only used to propose candidates; it never supplies a surrogate fitness.
"""

from __future__ import annotations

import numpy as np

from src.genetic_algorithm import tournament_select
from src.optimization import EvaluationBudget, integer_parameter, real_parameter
from src.stm_initialization import STMGeometry


def _setup(objective, geometry, max_evaluations, seed):
    if not isinstance(geometry, STMGeometry):
        raise ValueError("geometry must be an STMGeometry.")
    seed = integer_parameter(seed, "seed", minimum=0)
    return EvaluationBudget(objective, geometry.bounds, max_evaluations), np.random.default_rng(seed), seed


def _evaluate(budget, geometry, latent):
    return budget.evaluate(geometry.unit_vector(latent))


def satellite_crossover(first, second, generator, *, kind="satellite"):
    """Exchange complete six-component states, or use a controlled ablation."""
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 6:
        raise ValueError("Parents must have equal (N, 6) shapes.")
    if kind == "satellite":
        mask = generator.random((len(first), 1)) < .5
        return np.where(mask, second, first)
    if kind == "blend":
        # Convex combinations keep each satellite inside the feasible polytope.
        weights = generator.random((len(first), 1))
        return weights * first + (1 - weights) * second
    raise ValueError("crossover must be satellite or blend.")


def satellite_binomial_crossover(target, mutant, crossover_rate, generator):
    """DE crossover with one forced whole satellite (even at CR=0)."""
    if target.shape != mutant.shape or target.ndim != 2 or target.shape[1] != 6:
        raise ValueError("Target and mutant must have equal (N, 6) shapes.")
    crossover_rate = real_parameter(crossover_rate, "crossover_rate", 0, 1)
    mask = generator.random(len(target)) < crossover_rate
    mask[generator.integers(len(target))] = True
    return np.where(mask[:, None], mutant, target)


def informed_random_search(objective, geometry, max_evaluations, *, seed=0, informed_fraction=1.0):
    """Ablate initialization alone; optional uniform official-box samples."""
    informed_fraction = real_parameter(informed_fraction, "informed_fraction", 0, 1)
    budget, rng, seed = _setup(objective, geometry, max_evaluations, seed)
    while budget.remaining:
        informed = informed_fraction == 1 or (informed_fraction > 0 and rng.random() < informed_fraction)
        if informed:
            _evaluate(budget, geometry, geometry.sample(rng))
        else:
            budget.evaluate(rng.random(budget.space.dimension))
    return budget.finish("stm_random_search", seed)


def _ga_parameters(population_size, tournament_size, elite_count, crossover_rate,
                   crossover, mutation_cells, replacement_probability, restart_generations):
    population_size = integer_parameter(population_size, "population_size", minimum=2)
    tournament_size = integer_parameter(tournament_size, "tournament_size")
    elite_count = integer_parameter(elite_count, "elite_count")
    restart_generations = integer_parameter(restart_generations, "restart_generations")
    if tournament_size > population_size or elite_count >= population_size:
        raise ValueError("Require tournament_size <= population_size and elite_count < population_size.")
    if crossover not in ("satellite", "blend"):
        raise ValueError("crossover must be satellite or blend.")
    return dict(population_size=population_size, tournament_size=tournament_size, elite_count=elite_count,
                crossover_rate=real_parameter(crossover_rate, "crossover_rate", 0, 1), crossover=crossover,
                mutation_cells=real_parameter(mutation_cells, "mutation_cells", .001, 100),
                replacement_probability=real_parameter(replacement_probability, "replacement_probability", 0, 1),
                restart_generations=restart_generations)


def _ga_search(budget, geometry, rng, stop_at, parameters):
    """Internal GA stage sharing its budget with optional subsequent local search."""
    p = parameters
    size = min(p['population_size'], stop_at - budget.evaluations)
    population = np.array([geometry.sample(rng) for _ in range(size)])
    scores = np.array([_evaluate(budget, geometry, row) for row in population])
    stagnant = 0
    while budget.evaluations < stop_at:
        previous_best = budget.best_fitness
        order = np.argsort(scores, kind="stable")
        if stagnant >= p['restart_generations']:
            for index in order[max(p['elite_count'], size // 2):]:
                population[index] = geometry.sample(rng)
                scores[index] = _evaluate(budget, geometry, population[index])
                if budget.evaluations == stop_at:
                    return
            stagnant = 0
        order = np.argsort(scores, kind="stable")
        next_population, next_scores = population.copy(), scores.copy()
        next_population[:p['elite_count']] = population[order[:p['elite_count']]]
        next_scores[:p['elite_count']] = scores[order[:p['elite_count']]]
        cells = p['mutation_cells'] * (1 + min(stagnant, 10) / 10)
        for index in range(p['elite_count'], size):
            first = population[tournament_select(scores, p['tournament_size'], rng)]
            second = population[tournament_select(scores, p['tournament_size'], rng)]
            child = (satellite_crossover(first, second, rng, kind=p['crossover'])
                     if rng.random() < p['crossover_rate'] else first.copy())
            # Mutate at least one satellite, occasionally a small group.
            count = min(geometry.n_satellites, 1 + int(rng.random() < .25))
            for satellite in rng.choice(geometry.n_satellites, count, replace=False):
                if rng.random() < p['replacement_probability']:
                    child[satellite] = geometry.sample(rng, 1)[0]
                else:
                    child[satellite] = geometry.move(child[satellite], rng, cells=cells)
            next_population[index] = child
            next_scores[index] = _evaluate(budget, geometry, child)
            if budget.evaluations == stop_at:
                return
        population, scores = next_population, next_scores
        stagnant = stagnant + 1 if budget.best_fitness >= previous_best else 0


def informed_genetic_algorithm(objective, geometry, max_evaluations, *, seed=0,
                               population_size=40, tournament_size=2, elite_count=2,
                               crossover_rate=.9, crossover="satellite", mutation_cells=2.,
                               replacement_probability=.15, restart_generations=20):
    """Elitist GA, coherent crossover, feasible mutations and partial restarts."""
    parameters = _ga_parameters(population_size, tournament_size, elite_count, crossover_rate,
                                crossover, mutation_cells, replacement_probability, restart_generations)
    budget, rng, seed = _setup(objective, geometry, max_evaluations, seed)
    _ga_search(budget, geometry, rng, budget.maximum, parameters)
    return budget.finish("stm_genetic_algorithm", seed)


def informed_differential_evolution(objective, geometry, max_evaluations, *, seed=0,
                                    population_size=40, F=.5, CR=.6, adaptive=True,
                                    restart_generations=20):
    """Current-to-pbest/1 DE with satellite crossover and optional F/CR adaptation.

    On each trial, with probability .1 propose a fresh F in [.2,.9] and CR in
    [.1,.9]; retain them only on acceptance. Select pbest uniformly in the best
    20% (at least two). Partial restarts retain the current best. This is an
    independent variant using standard DE ideas, not a full JADE/jDE replica.
    """
    population_size = integer_parameter(population_size, "population_size", minimum=4)
    restart_generations = integer_parameter(restart_generations, "restart_generations")
    F = real_parameter(F, "F", .001, 2)
    CR = real_parameter(CR, "CR", 0, 1)
    if not isinstance(adaptive, bool):
        raise ValueError("adaptive must be a boolean.")
    budget, rng, seed = _setup(objective, geometry, max_evaluations, seed)
    size = min(population_size, budget.maximum)
    population = np.array([geometry.sample(rng) for _ in range(size)])
    scores = np.array([_evaluate(budget, geometry, row) for row in population])
    factors, rates = np.full(size, F), np.full(size, CR)
    stagnant = 0
    while budget.remaining:
        previous_best = budget.best_fitness
        order = np.argsort(scores, kind="stable")
        if stagnant >= restart_generations:
            for index in order[max(1, size * 3 // 4):]:
                population[index] = geometry.sample(rng)
                scores[index] = _evaluate(budget, geometry, population[index])
                factors[index], rates[index] = F, CR
                if not budget.remaining:
                    return budget.finish("stm_differential_evolution", seed)
            stagnant = 0
            order = np.argsort(scores, kind="stable")
        next_population, next_scores = population.copy(), scores.copy()
        next_factors, next_rates = factors.copy(), rates.copy()
        for target in range(size):
            f = rng.uniform(.2, .9) if adaptive and rng.random() < .1 else factors[target]
            cr = rng.uniform(.1, .9) if adaptive and rng.random() < .1 else rates[target]
            best = int(rng.choice(order[:max(2, int(np.ceil(.2 * size)))]))
            donors = np.array([i for i in range(size) if i != target and i != best])
            a, b = rng.choice(donors, 2, replace=False)
            mutant = geometry.repair(population[target] + f * (population[best] - population[target])
                                     + f * (population[a] - population[b]))
            trial = satellite_binomial_crossover(population[target], mutant, cr, rng)
            fitness = _evaluate(budget, geometry, trial)
            if fitness <= scores[target]:
                next_population[target], next_scores[target] = trial, fitness
                next_factors[target], next_rates[target] = f, cr
            if not budget.remaining:
                return budget.finish("stm_differential_evolution", seed)
        population, scores = next_population, next_scores
        factors, rates = next_factors, next_rates
        stagnant = stagnant + 1 if budget.best_fitness >= previous_best else 0
    return budget.finish("stm_differential_evolution", seed)
