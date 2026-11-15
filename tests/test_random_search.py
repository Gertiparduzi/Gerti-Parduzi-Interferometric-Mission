"""Uniform sampling baseline contracts, independent of the ESA runtime."""

import unittest

import numpy as np

from src.random_search import random_search


class TestRandomSearch(unittest.TestCase):
    def test_exact_budget_bounds_and_best_observed_candidate(self):
        observations = []
        lower, upper = np.array([-2.0, 7.0, 0.0]), np.array([3.0, 7.0, 0.1])
        def objective(x):
            self.assertTrue(np.all(x >= lower) and np.all(x <= upper))
            score = float(np.dot(x, x))
            observations.append((x.copy(), score))
            return score
        result = random_search(objective, (lower, upper), 53, seed=19)
        expected = min(observations, key=lambda pair: pair[1])
        self.assertEqual(len(observations), 53)
        self.assertEqual(result.evaluations, 53)
        self.assertEqual(result.best_fitness, expected[1])
        np.testing.assert_array_equal(result.best_chromosome, expected[0])

    def test_same_seed_is_exactly_reproducible(self):
        objective = lambda x: float(x @ x)
        first = random_search(objective, ([-5] * 4, [5] * 4), 83, seed=123)
        second = random_search(objective, ([-5] * 4, [5] * 4), 83, seed=123)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_global_numpy_random_state_is_untouched(self):
        state = np.random.get_state()
        random_search(lambda x: float(x @ x), ([-1], [1]), 10, seed=2)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_one_evaluation_and_flat_ties(self):
        for maximum in (1, 17):
            result = random_search(lambda x: 0.0, ([0], [1]), maximum, seed=4)
            self.assertEqual(result.best_found_at, 1)
            self.assertEqual(result.zero_fitness_evaluations, maximum)
            self.assertEqual(result.history[-1].evaluations, maximum)
            self.assertEqual(result.history[-1].best_fitness, 0.0)

    def test_larger_budget_extends_the_same_search_prefix(self):
        small = random_search(lambda x: float(x @ x), ([-3] * 2, [3] * 2), 10, seed=9)
        large = random_search(lambda x: float(x @ x), ([-3] * 2, [3] * 2), 100, seed=9)
        self.assertLessEqual(large.best_fitness, small.best_fitness)

    def test_invalid_seeds_are_rejected(self):
        for seed in (-1, True, 3.5, None, "3"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                random_search(lambda x: 0.0, ([0], [1]), 2, seed=seed)


if __name__ == "__main__":
    unittest.main()
