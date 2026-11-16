"""Geometry properties and real ESA survival checks, not surrogate optimization."""

import unittest
import numpy as np

from src.chromosome import validate_chromosome
from src.evaluator import ESAEvaluator
from src.stm_initialization import STMGeometry


class TestSTMGeometry(unittest.TestCase):
    def setUp(self):
        self.geometry = STMGeometry(np.array([np.eye(6)] * 3), 5, 1.23, grid_size=11)

    def test_whitening_and_inverse_residuals(self):
        g = self.geometry
        np.testing.assert_allclose(g.to_latent @ g.to_state, np.eye(6), atol=1e-13)
        np.testing.assert_allclose(g.whitened_constraints.T @ g.whitened_constraints,
                                   np.eye(6), atol=1e-13)

    def test_sample_bounds_survival_diversity_and_layout(self):
        g = self.geometry
        latent = g.sample(np.random.default_rng(7))
        x = g.to_chromosome(latent)
        validate_chromosome(x, 5)
        states = x.reshape(6, 5).T
        self.assertLessEqual(np.max(np.abs(states @ g.constraints.T)), g.margin + 1e-13)
        self.assertEqual(np.unique(states, axis=0).shape[0], 5)
        np.testing.assert_allclose(g.from_chromosome(x), latent, atol=1e-13)
        np.testing.assert_allclose(g.space.to_physical(g.unit_vector(latent)), x, atol=1e-14)

    def test_seed_reproducibility(self):
        np.testing.assert_array_equal(self.geometry.sample(np.random.default_rng(72)),
                                      self.geometry.sample(np.random.default_rng(72)))

    def test_repair_is_feasible_and_keeps_feasible_input(self):
        g = self.geometry
        original = g.sample(np.random.default_rng(4))
        np.testing.assert_array_equal(g.repair(original), original)
        repaired = g.repair(original * 100)
        self.assertLessEqual(np.max(np.abs(repaired @ g.whitened_constraints.T)), g.margin + 1e-12)
        np.testing.assert_array_equal(g.repair(np.zeros((5, 6))), np.zeros((5, 6)))

    def test_moves_preserve_constraints_on_boundary_and_do_not_mutate(self):
        g = self.geometry
        rng = np.random.default_rng(5)
        state = g.repair(g.sample(rng, 1) * 100)[0]
        before = state.copy()
        for cells in (.1, 2., 8., None):
            for _ in range(25):
                moved = g.move(state, rng, cells=cells)
                self.assertLessEqual(np.max(np.abs(g.whitened_constraints @ moved)), g.margin + 1e-12)
        np.testing.assert_array_equal(before, state)

    def test_invalid_stms_and_parameters(self):
        for stms in (np.zeros((2, 6, 6)), np.full((3, 6, 6), np.nan),
                     np.ones((3, 6, 6), dtype=complex), np.zeros((3, 6, 6))):
            with self.subTest(stms=str(stms.shape)), self.assertRaises(ValueError):
                STMGeometry(stms, 5, 1.23)
        for kwargs in (dict(n_satellites=0), dict(n_satellites=True), dict(inflation_factor=0),
                       dict(margin=1), dict(margin=float('nan')), dict(grid_size=True)):
            args = dict(n_satellites=5, inflation_factor=1.23)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                STMGeometry(np.array([np.eye(6)] * 3), **args)

    def test_ill_conditioned_geometry_is_rejected(self):
        stms = np.array([np.eye(6)] * 3)
        stms[2, 0, 0] = 1e14
        with self.assertRaisesRegex(ValueError, "ill-conditioned"):
            STMGeometry(stms, 5, 1.23)

    def test_invalid_coordinate_shapes_and_nonfinite_values(self):
        g = self.geometry
        for rows in (np.zeros(6), np.zeros((0, 6)), np.zeros((5, 5)),
                     np.full((5, 6), np.inf), np.zeros((5, 6), dtype=complex)):
            with self.subTest(shape=rows.shape), self.assertRaises(ValueError):
                g.repair(rows)
        for x in ([0] * 29, [float('nan')] * 30, [11] * 30):
            with self.assertRaises(ValueError):
                g.from_chromosome(x)
        with self.assertRaises(ValueError):
            g.to_chromosome(g.sample(np.random.default_rng(1), 6))


class TestRealSTMInitialization(unittest.TestCase):
    def test_all_instances_survive_and_beat_zero_with_reproducible_samples(self):
        for name in ('beginner', 'intermediate', 'hard'):
            evaluator = ESAEvaluator(name)
            g = STMGeometry.from_evaluator(evaluator)
            rng = np.random.default_rng(20261116)
            with self.subTest(instance=name):
                self.assertLess(g.condition_number, 1000)
                for _ in range(8):
                    x = g.to_chromosome(g.sample(rng))
                    states = x.reshape(6, evaluator.n_sat).T
                    # Same operation ordering as ESA, including scaling and inflation.
                    for epoch, stm in enumerate(evaluator.udp.stms):
                        pos = (stm @ (states * evaluator.udp.scaling_factor).T).T[:, :3]
                        pos /= evaluator.udp.scaling_factor
                        if epoch:
                            pos /= evaluator.udp.inflation_factor
                        self.assertTrue(np.all(np.abs(pos) < 1))
                    self.assertLess(evaluator.evaluate(x), 0)


if __name__ == '__main__':
    unittest.main()
