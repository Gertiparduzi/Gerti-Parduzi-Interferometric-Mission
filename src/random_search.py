"""Independent uniform sampling baseline; every draw is one objective call."""

from __future__ import annotations

import numpy as np

from src.optimization import (
    Bounds, EvaluationBudget, Objective, OptimizationResult, integer_parameter,
)


def random_search(
    objective: Objective,
    bounds: Bounds,
    max_evaluations: int,
    *,
    seed: int = 0,
) -> OptimizationResult:
    """Minimize a scalar objective with exactly max_evaluations valid samples.

    A positive budget is required. A local PCG64 generator leaves global NumPy
    random state untouched. Retain the first candidate on equal-fitness ties.
    """
    seed = integer_parameter(seed, "seed", minimum=0)
    budget = EvaluationBudget(objective, bounds, max_evaluations)
    generator = np.random.default_rng(seed)
    while budget.remaining:
        budget.evaluate(generator.random(budget.space.dimension))
    return budget.finish("random_search", seed)
