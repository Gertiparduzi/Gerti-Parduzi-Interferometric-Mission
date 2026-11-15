"""DE donor, crossover, budget and minimization contracts without ESA."""

import unittest
from unittest.mock import patch

import numpy as np

from src.differential_evolution import binomial_crossover, differential_evolution, donor_indices
from src.optimization import reflect_unit


class TestDifferentialEvolution(unittest.TestCase):
    def test_donors_are_distinct_and_exclude_each_target_at_minimum_population(self):
        generator = np.random.default_rng(7)
        for population_size in (4, 9, 40):
            for target in range(population_size):
                chosen = donor_indices(target, population_size, generator)
                self.assertEqual(len(set(chosen)), 3)
                self.assertNotIn(target, chosen)
                self.assertTrue(np.all((0 <= chosen) & (chosen < population_size)))

    def test_zero_crossover_rate_still_takes_one_mutant_coordinate(self):
        child = binomial_crossover(np.zeros(12), np.ones(12), 0.0, np.random.default_rng(3))
        self.assertEqual(np.count_nonzero(child), 1)

    def test_full_crossover_rate_takes_the_entire_mutant(self):
        child = binomial_crossover(np.zeros(12), np.ones(12), 1.0, np.random.default_rng(3))
        np.testing.assert_array_equal(child, np.ones(12))

    def test_donors_use_a_frozen_generation_not_previous_trial_updates(self):
        initial = np.random.default_rng(9).random((4, 2))
        observed = []
        def objective(x):
            observed.append(x.copy())
            return 10.0 if len(observed) <= 4 else -float(len(observed))
        def donors(target, size, generator):
            return np.array([index for index in range(size) if index != target])
        with patch("src.differential_evolution.donor_indices", donors):
            differential_evolution(objective, ([0, 0], [1, 1]), 8,
                                   seed=9, population_size=4, F=0.5, CR=1.0)
        for target in range(4):
            a, b, c = donors(target, 4, None)
            expected = reflect_unit(initial[a] + 0.5 * (initial[b] - initial[c]))
            np.testing.assert_array_equal(observed[4 + target], expected)

    def test_equal_fitness_trials_allow_neutral_population_movement(self):
        targets = []
        def crossover(target, mutant, rate, generator):
            targets.append(target.copy())
            return np.full_like(target, 0.9)
        with patch("src.differential_evolution.binomial_crossover", crossover):
            differential_evolution(lambda x: 0.0, ([0, 0], [1, 1]), 12,
                                   seed=9, population_size=4)
        for target in targets[4:]:
            np.testing.assert_array_equal(target, [0.9, 0.9])

    def test_exact_budget_with_partial_initialization_and_generations(self):
        for maximum in (1, 3, 4, 5, 7, 8, 13):
            calls = []
            def sphere(x):
                self.assertTrue(np.all((-1 <= x) & (x <= 1)))
                calls.append(x.copy())
                return float(x @ x)
            result = differential_evolution(sphere, ([-1] * 3, [1] * 3), maximum,
                                            seed=2, population_size=4, F=2.0, CR=1.0)
            self.assertEqual(len(calls), maximum)
            self.assertEqual(result.evaluations, maximum)
            self.assertEqual(result.best_fitness, min(float(x @ x) for x in calls))

    def test_sphere_improves_beyond_initial_population(self):
        scores = []
        def sphere(x):
            scores.append(float(x @ x))
            return scores[-1]
        result = differential_evolution(sphere, ([-5] * 3, [5] * 3), 600,
                                        seed=0, population_size=12)
        self.assertLess(result.best_fitness, min(scores[:12]))
        self.assertGreater(result.best_found_at, 12)

    def test_seed_reproducibility_and_monotone_history(self):
        first = differential_evolution(lambda x: float(x @ x), ([-3] * 2, [3] * 2), 107,
                                       seed=19, population_size=8)
        second = differential_evolution(lambda x: float(x @ x), ([-3] * 2, [3] * 2), 107,
                                        seed=19, population_size=8)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(all(a.best_fitness >= b.best_fitness
                            for a, b in zip(first.history, first.history[1:])))

    def test_one_dimensional_and_fixed_search_spaces(self):
        result = differential_evolution(lambda x: float(x[0] ** 2), ([-1], [1]), 43,
                                        seed=5, population_size=4, CR=0.0)
        self.assertEqual(result.evaluations, 43)
        fixed = differential_evolution(lambda x: float(x[0]), ([2], [2]), 9, population_size=4)
        self.assertEqual(fixed.best_fitness, 2.0)
        np.testing.assert_array_equal(fixed.best_chromosome, [2.0])

    def test_invalid_parameters(self):
        for parameters in (dict(population_size=3), dict(population_size=4.0), dict(F=0),
                           dict(F=-0.1), dict(F=2.01), dict(F=np.nan), dict(F=True),
                           dict(CR=-0.01), dict(CR=1.01), dict(CR=np.inf)):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                differential_evolution(lambda x: 0.0, ([-1], [1]), 5, **parameters)


if __name__ == "__main__":
    unittest.main()
