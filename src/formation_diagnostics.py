"""Read-only observation diagnostics, isolated from every optimization decision.

Propagation uses the official operation ordering. Grid support is independently
counted from pair differences only for auditing and plotting, never as the
optimizer's objective. The unchanged ESA UDP remains authoritative.
"""

from __future__ import annotations

import numpy as np

from src.chromosome import validate_chromosome


PLANES = ((0, 1), (0, 2), (1, 2))


def observe_formation(evaluator, chromosome):
    validate_chromosome(chromosome, evaluator.n_sat)
    problem = evaluator.udp
    states = np.asarray(chromosome, dtype=float).reshape(6, evaluator.n_sat).T
    observations = []
    for epoch, stm in enumerate(problem.stms):
        positions = (stm @ (states * problem.scaling_factor).T).T[:, :3] / problem.scaling_factor
        if epoch:
            positions = positions / problem.inflation_factor
        surviving = np.all((positions > -1) & (positions < 1), axis=1)
        cells = (positions[surviving] * problem.grid_size / 2 + problem.grid_size // 2).astype(int)
        support_counts = []
        for first, second in PLANES:
            projection = {(int(cell[first]), int(cell[second])) for cell in cells}
            differences = {(a[0] - b[0], a[1] - b[1]) for a in projection for b in projection}
            support_counts.append(len(differences))
        # Match ESA's per-plane divisions, then sum. Tiny roundoff is expected.
        denominator_side = 2 * problem.grid_size - 1
        fill = [count / denominator_side / denominator_side for count in support_counts]
        observations.append(dict(epoch=epoch, time=epoch * evaluator.config['period'],
                                 positions=positions.tolist(), surviving=surviving.tolist(),
                                 survivor_count=int(np.sum(surviving)), cells=cells.tolist(),
                                 support_counts=support_counts, plane_fill_factors=fill,
                                 sum_fill_factors=sum(fill)))
    return observations
