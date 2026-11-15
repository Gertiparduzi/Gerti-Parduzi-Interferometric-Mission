"""GA operators and synthetic optimization behavior without ESA initialization."""

import unittest
from unittest.mock import patch

import numpy as np

from src.genetic_algorithm import blend_crossover, gaussian_mutation, genetic_algorithm, tournament_select


class TestGeneticAlgorithm(unittest.TestCase):
    def test_tournament_minimizes(self):
        scores = np.array([8.0, -3.0, 2.0, 5.0])
        self.assertEqual(tournament_select(scores, 4, np.random.default_rng(1)), 1)

    def test_satellite_blend_keeps_component_blocks_aligned(self):
        child = blend_crossover(np.zeros(18), np.ones(18), 0.0, np.random.default_rng(2), 3)
        for component in range(1, 6):
            np.testing.assert_array_equal(child[:3], child[3 * component:3 * (component + 1)])
        self.assertGreater(len(set(child[:3])), 1)

    def test_mutation_zero_is_identity_and_strong_mutation_stays_bounded(self):
        x = np.linspace(0.0, 1.0, 50)
        np.testing.assert_array_equal(gaussian_mutation(x, 0.0, 1.0, np.random.default_rng(7)), x)
        changed = gaussian_mutation(x, 1.0, 1.0, np.random.default_rng(7))
        self.assertTrue(np.all((0 <= changed) & (changed <= 1)))
        self.assertFalse(np.array_equal(changed, x))

    def test_budget_edges_and_all_candidates_in_bounds(self):
        for maximum in (1, 5, 12, 13, 23, 24, 37):
            calls = []
            def sphere(x):
                self.assertTrue(np.all((-5 <= x) & (x <= 5)))
                calls.append(x.copy())
                return float(x @ x)
            result = genetic_algorithm(sphere, ([-5] * 3, [5] * 3), maximum,
                                       seed=4, population_size=12)
            self.assertEqual(len(calls), maximum)
            self.assertEqual(result.evaluations, maximum)
            self.assertAlmostEqual(result.best_fitness, min(float(x @ x) for x in calls))

    def test_sphere_improves_beyond_initial_population(self):
        scores = []
        def sphere(x):
            scores.append(float(x @ x))
            return scores[-1]
        result = genetic_algorithm(sphere, ([-5] * 3, [5] * 3), 600,
                                   seed=0, population_size=12)
        self.assertLess(result.best_fitness, min(scores[:12]))
        self.assertGreater(result.best_found_at, 12)

    def test_seed_and_elitism_keep_a_monotone_best_history(self):
        settings = dict(seed=12, population_size=10, elite_count=3)
        first = genetic_algorithm(lambda x: float(x @ x), ([-2] * 4, [2] * 4), 105, **settings)
        second = genetic_algorithm(lambda x: float(x @ x), ([-2] * 4, [2] * 4), 105, **settings)
        self.assertEqual(first.to_dict(), second.to_dict())
        values = [point.best_fitness for point in first.history]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_no_variation_uses_cached_elites_and_never_loses_best(self):
        scores = []
        def sphere(x):
            scores.append(float(x @ x))
            return scores[-1]
        result = genetic_algorithm(sphere, ([-2] * 2, [2] * 2), 43, seed=5,
                                   population_size=8, crossover_rate=0.0, mutation_probability=0.0)
        self.assertEqual(len(scores), 43)
        self.assertEqual(result.best_fitness, min(scores[:8]))
        self.assertTrue(all(value in scores[:8] for value in scores))

    def test_elite_survives_generations_without_being_reevaluated(self):
        observed, selection_minima = [], []
        def objective(x):
            observed.append(x.copy())
            return float(x @ x)
        def select_worst(scores, size, generator):
            selection_minima.append(float(np.min(scores)))
            return int(np.argmax(scores))
        with patch("src.genetic_algorithm.tournament_select", select_worst):
            result = genetic_algorithm(objective, ([-1] * 2, [1] * 2), 27,
                                       seed=3, population_size=6,
                                       crossover_rate=0.0, mutation_probability=0.0)
        initial_best = min(float(x @ x) for x in observed[:6])
        self.assertTrue(all(value == initial_best for value in selection_minima))
        self.assertTrue(all(float(x @ x) != initial_best for x in observed[6:]))
        self.assertEqual(result.best_fitness, initial_best)

    def test_invalid_parameters(self):
        settings = [dict(population_size=1), dict(population_size=True), dict(tournament_size=41),
                    dict(elite_count=40), dict(elite_count=-1), dict(mutation_sigma=np.inf),
                    dict(mutation_probability=1.01), dict(crossover_rate=-0.1),
                    dict(crossover_alpha=np.nan), dict(n_satellites=2)]
        for parameters in settings:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                genetic_algorithm(lambda x: 0.0, ([-1] * 3, [1] * 3), 4, **parameters)


if __name__ == "__main__":
    unittest.main()
