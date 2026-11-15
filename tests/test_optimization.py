"""Shared numerical and accounting contracts, with no ESA initialization."""

import unittest

import numpy as np

from src.optimization import EvaluationBudget, ObjectiveEvaluationError, SearchSpace, reflect_unit


class TestOptimizationContracts(unittest.TestCase):
    def test_invalid_bounds_are_rejected(self):
        invalid = [None, ([], []), ([0], [1, 2]), ([2], [1]), ([0], [np.inf]),
                   ([np.nan], [1]), ([True, 0], [1, 1]), (["0"], [1]),
                   ([0j], [1]), ([[0, 0]], [[1, 1]]), ([-1e308], [1e308])]
        for bounds in invalid:
            with self.subTest(bounds=repr(bounds)), self.assertRaises(ValueError):
                SearchSpace(bounds)

    def test_bound_inputs_are_copied_and_fixed_coordinates_work(self):
        lower = np.array([-2.0, 4.0])
        upper = np.array([6.0, 4.0])
        space = SearchSpace((lower, upper))
        lower[:] = 99
        upper[:] = 100
        np.testing.assert_array_equal(space.to_physical(np.array([0.25, 0.9])), [0.0, 4.0])

    def test_reflection_handles_both_edges_and_repeated_crossings(self):
        values = np.array([-4.2, -1.2, -0.2, 0.0, 0.3, 1.0, 1.2, 2.2, 5.2])
        np.testing.assert_allclose(reflect_unit(values), [0.2, 0.8, 0.2, 0.0, 0.3, 1.0, 0.8, 0.2, 0.8])
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                reflect_unit(np.array([bad]))

    def test_budget_rejects_invalid_limits_and_uncallable_objective(self):
        for maximum in (0, -1, True, 2.0, "2", None):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                EvaluationBudget(lambda x: 0.0, ([0], [1]), maximum)
        with self.assertRaises(ValueError):
            EvaluationBudget(None, ([0], [1]), 1)

    def test_extra_call_is_blocked_before_the_objective(self):
        calls = []
        budget = EvaluationBudget(lambda x: calls.append(x.copy()) or 2.0, ([0], [1]), 1)
        budget.evaluate(np.array([0.5]))
        with self.assertRaises(RuntimeError):
            budget.evaluate(np.array([0.7]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(budget.evaluations, 1)

    def test_failed_attempts_are_counted_and_reported(self):
        def objective(x):
            raise ArithmeticError("deliberate failure")
        budget = EvaluationBudget(objective, ([0], [1]), 3)
        with self.assertRaisesRegex(ObjectiveEvaluationError, "evaluation 1.*deliberate failure"):
            budget.evaluate(np.array([0.2]))
        self.assertEqual(budget.evaluations, 1)

    def test_invalid_objective_values_fail_instead_of_becoming_scores(self):
        for value in (np.nan, np.inf, -np.inf, True, np.bool_(False), [0.0], 0j, None):
            budget = EvaluationBudget(lambda x: value, ([0], [1]), 1)
            with self.subTest(value=repr(value)), self.assertRaises(ObjectiveEvaluationError):
                budget.evaluate(np.array([0.5]))

    def test_accidental_objective_mutation_is_detected(self):
        def objective(x):
            x[0] = 99
            return 0.0
        budget = EvaluationBudget(objective, ([0], [1]), 1)
        with self.assertRaises(ObjectiveEvaluationError):
            budget.evaluate(np.array([0.3]))

    def test_sparse_history_records_improvements_and_final_budget(self):
        values = iter([5.0, 5.0, 3.0, 4.0])
        budget = EvaluationBudget(lambda x: next(values), ([0], [1]), 4)
        for _ in range(4):
            budget.evaluate(np.array([0.5]))
        result = budget.finish("test", 7)
        self.assertEqual([(p.evaluations, p.best_fitness) for p in result.history],
                         [(1, 5.0), (3, 3.0), (4, 3.0)])
        self.assertEqual(result.best_found_at, 3)
        self.assertFalse(result.best_chromosome.flags.writeable)


if __name__ == "__main__":
    unittest.main()
