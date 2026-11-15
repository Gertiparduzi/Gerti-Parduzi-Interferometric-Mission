"""Shared contracts, bounds and exact objective-call accounting for minimizers.

Algorithms operate on unit coordinates so a mutation scale has the same meaning
for position bounds [-1, 1] and velocity bounds [-10, 10]. The objective receives
the original physical decision vector, in its original component order.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]
Bounds = tuple[ArrayLike, ArrayLike]
Objective = Callable[[Vector], float]


def integer_parameter(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def real_parameter(value: float, name: str, lower: float, upper: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in [{lower}, {upper}].")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} is not representable as float64.") from error
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value


def checked_vector(values: ArrayLike, name: str) -> Vector:
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf" or any(
            isinstance(value, (bool, np.bool_)) for value in np.asarray(values, dtype=object).flat
        ):
            raise ValueError(f"{name} must contain real numeric values.")
        with np.errstate(over="ignore", invalid="ignore"):
            vector = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional numeric vector.") from error
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite nonempty one-dimensional vector.")
    return vector


class SearchSpace:
    """Validated box bounds; fixed coordinates are allowed."""

    def __init__(self, bounds: Bounds) -> None:
        try:
            if len(bounds) != 2:
                raise ValueError("Bounds must contain lower and upper vectors.")
            self.lower = checked_vector(bounds[0], "Lower bounds")
            self.upper = checked_vector(bounds[1], "Upper bounds")
        except TypeError as error:
            raise ValueError("Bounds must contain lower and upper vectors.") from error
        if self.lower.shape != self.upper.shape or np.any(self.lower > self.upper):
            raise ValueError("Bounds must have equal dimensions and lower <= upper.")
        with np.errstate(over="ignore", invalid="ignore"):
            self.width = self.upper - self.lower
        if not np.all(np.isfinite(self.width)):
            raise ValueError("Bound widths must be representable as float64.")
        self.dimension = len(self.lower)

    def to_physical(self, unit_vector: Vector) -> Vector:
        if unit_vector.shape != self.lower.shape or not np.all(np.isfinite(unit_vector)):
            raise ValueError("Unit vector has invalid shape or nonfinite values.")
        if np.any(unit_vector < 0.0) or np.any(unit_vector > 1.0):
            raise ValueError("Unit coordinates must be in [0, 1].")
        # The final clip only prevents a floating-point rounding overshoot.
        return np.clip(self.lower + unit_vector * self.width, self.lower, self.upper)


def reflect_unit(values: Vector) -> Vector:
    """Reflect any finite coordinates into [0, 1], without boundary pile-ups."""
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot reflect nonfinite coordinates.")
    remainder = np.remainder(values, 2.0)
    return np.where(remainder <= 1.0, remainder, 2.0 - remainder)


@dataclass(frozen=True)
class HistoryPoint:
    evaluations: int
    best_fitness: float


@dataclass(frozen=True)
class OptimizationResult:
    algorithm: str
    seed: int
    best_chromosome: Vector
    best_fitness: float
    evaluations: int
    best_found_at: int
    zero_fitness_evaluations: int
    history: tuple[HistoryPoint, ...]

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "best_chromosome": self.best_chromosome.tolist(),
            "best_fitness": self.best_fitness,
            "evaluations": self.evaluations,
            "best_found_at": self.best_found_at,
            "zero_fitness_evaluations": self.zero_fitness_evaluations,
            "history": [{"evaluations": point.evaluations,
                         "best_fitness": point.best_fitness} for point in self.history],
        }


class ObjectiveEvaluationError(RuntimeError):
    """An attempted objective call failed or returned a nonfinite/nonreal scalar."""


class EvaluationBudget:
    """One gate for all objective calls; sparse best-so-far history is O(B)."""

    def __init__(self, objective: Objective, bounds: Bounds, max_evaluations: int) -> None:
        if not callable(objective):
            raise ValueError("objective must be callable and return one finite real scalar.")
        self.maximum = integer_parameter(max_evaluations, "max_evaluations")
        self.space = SearchSpace(bounds)
        self.objective = objective
        self.evaluations = 0
        self.best_fitness = float("inf")
        self.best_chromosome: Vector | None = None
        self.best_found_at = 0
        self.zero_fitness_evaluations = 0
        self._history: list[HistoryPoint] = []

    @property
    def remaining(self) -> int:
        return self.maximum - self.evaluations

    def evaluate(self, unit_vector: Vector) -> float:
        if not self.remaining:
            raise RuntimeError("Objective evaluation budget exhausted.")
        candidate = self.space.to_physical(unit_vector)
        # Passing a read-only vector catches accidental mutation by an objective.
        candidate.flags.writeable = False
        self.evaluations += 1
        try:
            value = self.objective(candidate)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError("The objective must return a real scalar, not [fitness].")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("The objective returned NaN or infinity.")
        except Exception as error:
            raise ObjectiveEvaluationError(
                f"Objective evaluation {self.evaluations} failed: {error}"
            ) from error
        if value == 0.0:
            self.zero_fitness_evaluations += 1
        if self.best_chromosome is None or value < self.best_fitness:
            self.best_fitness = value
            self.best_chromosome = candidate.copy()
            self.best_found_at = self.evaluations
            self._history.append(HistoryPoint(self.evaluations, value))
        return value

    def finish(self, algorithm: str, seed: int) -> OptimizationResult:
        if self.best_chromosome is None:
            raise RuntimeError("No candidate was evaluated.")
        history = list(self._history)
        if history[-1].evaluations != self.evaluations:
            history.append(HistoryPoint(self.evaluations, self.best_fitness))
        best = self.best_chromosome.copy()
        best.flags.writeable = False
        return OptimizationResult(
            algorithm, seed, best, self.best_fitness, self.evaluations,
            self.best_found_at, self.zero_fitness_evaluations, tuple(history),
        )
