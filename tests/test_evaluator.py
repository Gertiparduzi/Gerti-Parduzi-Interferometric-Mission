"""Real ESA integration and an independent, set-based baseline-count oracle.

The oracle is test-only. Production always uses the unchanged ESA evaluator.
Synthetic STM tests isolate cropping, inflation and discretization behavior;
the other tests use the real CR3BP STMs initialized by ESA/heyoka.
"""

import copy
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pygmo as pg

from src.chromosome import chromosome_bounds, decode_chromosome, encode_chromosome
from src.evaluator import ESAEvaluator, SOURCE_FILES


def epoch_sums_by_pair_differences(problem, chromosome, *, compound_inflation=False):
    """Count exact distinct grid displacements without SciPy correlation."""
    states = np.asarray(chromosome, dtype=np.float64).reshape(6, problem.n_sat).T
    sums = []
    for epoch, stm in enumerate(problem.stms):
        positions = (stm @ (states * problem.scaling_factor).T).T[:, :3]
        positions = positions / problem.scaling_factor
        if epoch:
            divisor = problem.inflation_factor ** epoch if compound_inflation else problem.inflation_factor
            positions = positions / divisor
        cells = set()
        for position in positions:
            if all(-1.0 < coordinate < 1.0 for coordinate in position):
                cells.add(tuple(math.trunc(coordinate * problem.grid_size / 2
                                          + problem.grid_size // 2) for coordinate in position))
        covered = 0
        for first, second in ((0, 1), (0, 2), (1, 2)):
            projected = {(cell[first], cell[second]) for cell in cells}
            differences = {(a[0] - b[0], a[1] - b[1]) for a in projected for b in projected}
            covered += len(differences)
        sums.append(covered / (2 * problem.grid_size - 1) ** 2)
    return sums


class TestESAEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluators = {name: ESAEvaluator(name) for name in SOURCE_FILES}
        path = Path(__file__).resolve().parents[1] / "examples" / "beginner_candidate.json"
        cls.candidate = json.loads(path.read_text(encoding="utf-8"))

    def test_zero_vector_has_hand_calculated_objective(self):
        for name, evaluator in self.evaluators.items():
            with self.subTest(instance=name):
                expected = -3.0 / (2 * evaluator.udp.grid_size - 1) ** 2
                self.assertAlmostEqual(evaluator.evaluate([0.0] * (6 * evaluator.n_sat)), expected, places=14)

    def test_bounds_and_continuous_dimension_match_official_contract(self):
        for name, evaluator in self.evaluators.items():
            with self.subTest(instance=name):
                self.assertEqual(evaluator.get_bounds(), chromosome_bounds(evaluator.n_sat))
                self.assertEqual(evaluator.get_nix(), 0)
                self.assertEqual(len(evaluator.get_bounds()[0]), 6 * evaluator.n_sat)

    def test_real_stms_have_three_epochs_and_start_at_identity(self):
        for name, evaluator in self.evaluators.items():
            with self.subTest(instance=name):
                self.assertEqual(evaluator.udp.stms.shape, (3, 6, 6))
                np.testing.assert_allclose(evaluator.udp.stms[0], np.eye(6), rtol=0, atol=1e-15)
                self.assertEqual(evaluator.udp.T, 2 * evaluator.config["period"])

    def test_predefined_nonzero_candidate_matches_direct_official_call(self):
        evaluator = self.evaluators["beginner"]
        direct = evaluator.udp.fitness(self.candidate)
        self.assertEqual(evaluator.fitness(self.candidate), direct)
        self.assertLess(evaluator.evaluate(self.candidate), 0.0)

    def test_random_valid_vectors_match_independent_baseline_oracle(self):
        generator = np.random.default_rng(20261115)
        for name, evaluator in self.evaluators.items():
            for repetition in range(6):
                x = np.concatenate((generator.uniform(-0.8, 0.8, 3 * evaluator.n_sat),
                                    generator.uniform(-0.04, 0.04, 3 * evaluator.n_sat)))
                with self.subTest(instance=name, repetition=repetition):
                    sums = epoch_sums_by_pair_differences(evaluator.udp, x)
                    self.assertAlmostEqual(evaluator.evaluate(x), -min(sums), places=13)
                    self.assertEqual(evaluator.fitness(x), evaluator.udp.fitness(x))

    def test_objective_uses_worst_epoch_and_sums_the_three_planes(self):
        evaluator = self.evaluators["beginner"]
        sums = epoch_sums_by_pair_differences(evaluator.udp, self.candidate)
        self.assertGreater(max(sums), min(sums))
        self.assertAlmostEqual(evaluator.evaluate(self.candidate), -min(sums), places=14)
        self.assertNotAlmostEqual(evaluator.evaluate(self.candidate), -max(sums), places=10)

    def test_inclusive_decision_edge_can_be_cropped_by_esa(self):
        for name, evaluator in self.evaluators.items():
            x = [0.0] * (6 * evaluator.n_sat)
            x[:evaluator.n_sat] = [1.0] * evaluator.n_sat
            with self.subTest(instance=name):
                self.assertEqual(evaluator.evaluate(x), 0.0)

    def test_3d_cropping_happens_before_xy_projection(self):
        evaluator = self.evaluators["beginner"]
        rows = [(value, value / 2, 1.0, 0.0, 0.0, 0.0)
                for value in (-0.8, -0.4, 0.0, 0.4, 0.8)]
        x = encode_chromosome(rows, check_bounds=True)
        self.assertEqual(epoch_sums_by_pair_differences(evaluator.udp, x)[0], 0.0)
        self.assertEqual(evaluator.evaluate(x), 0.0)

    def test_later_inflation_is_not_compounded_synthetic_stms(self):
        problem = copy.copy(self.evaluators["beginner"].udp)
        problem.stms = np.array([np.eye(6), 1.25 * np.eye(6), 1.75 * np.eye(6)])
        x = encode_chromosome([(0.8, 0.0, 0.0, 0.0, 0.0, 0.0)] * 5, check_bounds=True)
        correct = epoch_sums_by_pair_differences(problem, x)
        compounded = epoch_sums_by_pair_differences(problem, x, compound_inflation=True)
        self.assertEqual(correct[2], 0.0)
        self.assertGreater(compounded[2], 0.0)
        self.assertEqual(problem.fitness(x), [-min(correct)])

    def test_grid_conversion_truncates_after_shift_synthetic_stms(self):
        problem = copy.copy(self.evaluators["beginner"].udp)
        problem.stms = np.array([np.eye(6)] * 3)
        problem.inflation_factor = 1.0
        x = encode_chromosome([(value, 0.0, 0.0, 0.0, 0.0, 0.0)
                               for value in (-0.99, -0.8, -0.6, -0.4, -0.2)], check_bounds=True)
        # x grid cells [0, 0, 1, 2, 3]: 7 XY + 7 XZ + 1 YZ displacements.
        self.assertAlmostEqual(problem.fitness(x)[0], -15.0 / 441.0, places=14)

    def test_round_trip_keeps_official_evaluation_for_all_instances(self):
        generator = np.random.default_rng(37)
        for name, evaluator in self.evaluators.items():
            x = generator.uniform(-0.5, 0.5, 6 * evaluator.n_sat)
            reconstructed = encode_chromosome(decode_chromosome(x, evaluator.n_sat), check_bounds=True)
            with self.subTest(instance=name):
                np.testing.assert_array_equal(reconstructed, x)
                self.assertEqual(evaluator.fitness(reconstructed), evaluator.udp.fitness(x))

    def test_invalid_vector_is_rejected_before_calling_esa(self):
        evaluator = self.evaluators["beginner"]
        invalid = [[0.0] * 29, [0.0] * 31, [float("nan")] * 30,
                   [float("inf")] * 30, [True] * 30, [2.0] * 30,
                   np.zeros((30, 1)), np.zeros(30, dtype=complex)]
        with patch.object(evaluator.udp, "fitness") as official:
            for vector in invalid:
                with self.subTest(kind=type(vector).__name__), self.assertRaises(ValueError):
                    evaluator.fitness(vector)
            official.assert_not_called()

    def test_incompatible_heyoka_version_has_actionable_error(self):
        modern = SimpleNamespace(__version__="7.13.2")
        with patch.dict("sys.modules", {"heyoka": modern}):
            with self.assertRaisesRegex(RuntimeError, "heyoka.py=3.2.0"):
                ESAEvaluator("beginner")

    def test_evaluation_keeps_readonly_numpy_input_unchanged(self):
        evaluator = self.evaluators["beginner"]
        x = np.array(self.candidate, dtype=np.float64)
        before = x.copy()
        x.flags.writeable = False
        evaluator.fitness(x)
        np.testing.assert_array_equal(x, before)

    def test_both_reference_and_wrapper_support_pygmo_problem(self):
        evaluator = self.evaluators["beginner"]
        for udp in (evaluator.udp, evaluator):
            with self.subTest(kind=type(udp).__name__):
                problem = pg.problem(udp)
                self.assertEqual(problem.get_nx(), 30)
                self.assertEqual(problem.get_nobj(), 1)
                self.assertEqual(problem.get_nix(), 0)
                np.testing.assert_allclose(problem.fitness(self.candidate), evaluator.fitness(self.candidate), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
