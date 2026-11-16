import unittest

import numpy as np

from src.evaluator import ESAEvaluator
from src.formation_diagnostics import observe_formation
from src.stm_initialization import STMGeometry


class TestFormationDiagnostics(unittest.TestCase):
    def test_pair_support_matches_official_on_feasible_and_cropped_formations(self):
        for instance in ('beginner', 'intermediate', 'hard'):
            evaluator = ESAEvaluator(instance)
            geometry = STMGeometry.from_evaluator(evaluator)
            rng = np.random.default_rng(45)
            candidates = [geometry.to_chromosome(geometry.sample(rng)) for _ in range(3)]
            candidates.append(geometry.space.to_physical(rng.random(6 * evaluator.n_sat)))
            for x in candidates:
                before = x.copy()
                observations = observe_formation(evaluator, x)
                self.assertAlmostEqual(evaluator.evaluate(x),
                                       -min(o['sum_fill_factors'] for o in observations), places=14)
                self.assertEqual(len(observations), 3)
                np.testing.assert_array_equal(x, before)

    def test_zero_and_strict_boundary_support_counts(self):
        evaluator = ESAEvaluator('beginner')
        zero = np.zeros(30)
        observations = observe_formation(evaluator, zero)
        self.assertTrue(all(o['support_counts'] == [1, 1, 1] for o in observations))
        zero[:5] = 1
        first = observe_formation(evaluator, zero)[0]
        self.assertEqual(first['survivor_count'], 0)
        self.assertEqual(first['support_counts'], [0, 0, 0])

    def test_beginner_counting_upper_bound(self):
        # At most N(N-1)+1 distinct ordered differences per projection.
        evaluator = ESAEvaluator('beginner')
        n, grid = evaluator.n_sat, evaluator.udp.grid_size
        upper = 3 * (n * (n - 1) + 1) / (2 * grid - 1) ** 2
        self.assertAlmostEqual(upper, 1 / 7)
        geometry = STMGeometry.from_evaluator(evaluator)
        rng = np.random.default_rng(1001)
        scores = [evaluator.evaluate(geometry.to_chromosome(geometry.sample(rng))) for _ in range(40)]
        self.assertTrue(all(value >= -upper - 1e-14 for value in scores))


if __name__ == '__main__':
    unittest.main()
