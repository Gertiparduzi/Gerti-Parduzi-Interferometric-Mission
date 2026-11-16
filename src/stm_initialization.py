"""Independent STM-informed sampling; the ESA objective is never replaced.

For one satellite, C stacks the normalized decision bounds and the three
position maps (later maps divided by ESA's *single* inflation factor).
The sufficient survival region is |C x| < 1. With C = U S V.T, x = W z and
W = V S^-1, C W has orthonormal columns. This removes the large dynamical
anisotropy without a normal-equations inverse or a truncated pseudoinverse.

Sampling a spherical direction and scaling it to a polytope chord is an
informed distribution, NOT a claim of uniform sampling of that polytope.
Restricting search to surviving satellites is a heuristic subset of the
official search space; it does not change the official bounds or objective.
"""

from __future__ import annotations

import numpy as np

from src.chromosome import chromosome_bounds
from src.optimization import SearchSpace, checked_vector, integer_parameter, real_parameter


class STMGeometry:
    """Six-dimensional satellite geometry, shared by all N satellite blocks."""

    def __init__(self, stms, n_satellites, inflation_factor, *, margin=0.995, grid_size=21):
        self.n_satellites = integer_parameter(n_satellites, "n_satellites")
        self.grid_size = integer_parameter(grid_size, "grid_size", minimum=2)
        self.margin = real_parameter(margin, "margin", 0.01, 0.999999)
        self.inflation_factor = real_parameter(inflation_factor, "inflation_factor", 0.01, 100.0)
        raw = np.asarray(stms)
        if raw.shape != (3, 6, 6) or raw.dtype.kind not in "iuf":
            raise ValueError("stms must be a real (3, 6, 6) array.")
        self.stms = np.array(raw, dtype=float, copy=True)
        if not np.all(np.isfinite(self.stms)):
            raise ValueError("stms must be finite.")
        if not np.allclose(self.stms[0], np.eye(6), atol=1e-14, rtol=0):
            raise ValueError("The first STM must be the identity.")
        self.observation_maps = self.stms[:, :3, :].copy()
        self.observation_maps[1:] /= self.inflation_factor
        self.constraints = np.concatenate((np.diag([1., 1., 1., .1, .1, .1]),
                                           self.observation_maps.reshape(-1, 6)))
        _, singular, vt = np.linalg.svd(self.constraints, full_matrices=False)
        self.singular_values = singular
        self.condition_number = float(singular[0] / singular[-1])
        if not np.isfinite(self.condition_number) or self.condition_number > 1e10:
            raise ValueError("STM constraint system is too ill-conditioned for reliable sampling.")
        self.to_state = vt.T / singular
        self.to_latent = singular[:, None] * vt
        self.whitened_constraints = self.constraints @ self.to_state
        self.whitened_observations = self.whitened_constraints[6:]
        residual = np.max(np.abs(self.whitened_constraints.T @ self.whitened_constraints - np.eye(6)))
        if residual > 1e-6:
            raise ValueError("SVD whitening failed its numerical residual check.")
        self.whitening_residual = float(residual)
        self.bounds = chromosome_bounds(self.n_satellites)
        self.space = SearchSpace(self.bounds)

    @classmethod
    def from_evaluator(cls, evaluator, *, margin=0.995):
        problem = evaluator.udp
        return cls(problem.stms, evaluator.n_sat, problem.inflation_factor,
                   margin=margin, grid_size=problem.grid_size)

    @staticmethod
    def _rows(values):
        raw = np.asarray(values)
        if raw.ndim != 2 or raw.shape[1] != 6 or raw.shape[0] == 0 or raw.dtype.kind not in "iuf":
            raise ValueError("Satellite coordinates must be a nonempty real (N, 6) array.")
        rows = np.array(raw, dtype=float, copy=True)
        if not np.all(np.isfinite(rows)):
            raise ValueError("Satellite coordinates must be finite.")
        return rows

    def repair(self, latent):
        """Radially contract individual satellites into the sufficient region."""
        rows = self._rows(latent)
        extent = np.max(np.abs(rows @ self.whitened_constraints.T), axis=1)
        rows *= (self.margin / np.maximum(self.margin, extent))[:, None]
        return rows

    def sample(self, generator, count=None):
        """Draw independent, spread-out feasible satellite states in latent coordinates."""
        count = self.n_satellites if count is None else integer_parameter(count, "count")
        directions = generator.normal(size=(count, 6))
        extent = np.max(np.abs(directions @ self.whitened_constraints.T), axis=1)
        # A zero normal vector is possible in principle; give it a finite direction.
        zero = extent == 0
        directions[zero, 0] = 1.0
        extent[zero] = np.max(np.abs(self.whitened_constraints[:, 0]))
        radii = generator.random(count) ** (1.0 / 6.0)
        return directions * (self.margin * radii / extent)[:, None]

    def to_chromosome(self, latent):
        rows = self._rows(latent)
        if rows.shape[0] != self.n_satellites:
            raise ValueError("Formation has the wrong number of satellites.")
        states = rows @ self.to_state.T
        if np.max(np.abs(states @ self.constraints.T)) > 1.0 + 1e-12:
            raise ValueError("Formation lies outside the survival region; call repair explicitly.")
        chromosome = states.T.reshape(-1).copy()
        if np.any(chromosome < self.space.lower) or np.any(chromosome > self.space.upper):
            raise ValueError("Formation exceeds official decision bounds.")
        return chromosome

    def from_chromosome(self, chromosome):
        vector = checked_vector(chromosome, "chromosome")
        if vector.shape != self.space.lower.shape:
            raise ValueError("Chromosome has the wrong dimension.")
        if np.any(vector < self.space.lower) or np.any(vector > self.space.upper):
            raise ValueError("Chromosome exceeds official decision bounds.")
        return vector.reshape(6, self.n_satellites).T @ self.to_latent.T

    def unit_vector(self, latent):
        return (self.to_chromosome(latent) - self.space.lower) / self.space.width

    def move(self, state, generator, *, cells=2.0):
        """Feasible line move with an observation displacement measured in grid cells.

        A random direction is whitened. The chord limits enforce every linear
        constraint; the step window is scaled by the largest observation change.
        No objective is evaluated here. cells=None samples the entire chord.
        """
        row = checked_vector(state, "state")
        if row.shape != (6,):
            raise ValueError("state must contain six latent coordinates.")
        if cells is not None:
            cells = real_parameter(cells, "cells", 0.001, 1000.0)
        q = self.whitened_constraints
        current = q @ row
        if np.max(np.abs(current)) > self.margin + 1e-10:
            raise ValueError("Line moves must start inside the margin.")
        direction = generator.normal(size=6)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return row.copy()
        direction /= norm
        slopes = q @ direction
        nonzero = np.abs(slopes) > 1e-14
        ends1 = (-self.margin - current[nonzero]) / slopes[nonzero]
        ends2 = (self.margin - current[nonzero]) / slopes[nonzero]
        lower = float(np.max(np.minimum(ends1, ends2)))
        upper = float(np.min(np.maximum(ends1, ends2)))
        if cells is not None:
            rate = float(np.max(np.abs(self.whitened_observations @ direction)))
            window = cells * (2.0 / self.grid_size) / max(rate, 1e-14)
            lower, upper = max(lower, -window), min(upper, window)
        # Roundoff at the boundary may produce a tiny reversed interval.
        if lower > upper:
            if lower - upper > 1e-9:
                raise ValueError("Empty feasible line interval.")
            return self.repair(row[None, :])[0]
        moved = row + generator.uniform(lower, upper) * direction
        return self.repair(moved[None, :])[0]

    def diagnostics(self):
        return dict(condition_number=self.condition_number,
                    singular_values=self.singular_values.tolist(),
                    whitening_residual=self.whitening_residual, margin=self.margin)
