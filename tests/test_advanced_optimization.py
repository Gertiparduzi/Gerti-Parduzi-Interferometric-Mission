"""Meaningful optimizer contracts on counted objectives and the real ESA UDP."""

import unittest
from unittest.mock import patch

import numpy as np

from src.advanced_optimization import (
    informed_differential_evolution, informed_genetic_algorithm, informed_random_search,
    satellite_binomial_crossover, satellite_crossover,
)
from src.advanced_experiments import run_variant
from src.chromosome import validate_chromosome
from src.evaluator import ESAEvaluator
from src.hybrid_optimization import sage_hybrid
from src.optimization import ObjectiveEvaluationError
from src.stm_initialization import STMGeometry


SOLVERS = (informed_random_search, informed_genetic_algorithm,
           informed_differential_evolution, sage_hybrid)


class TestAdvancedContracts(unittest.TestCase):
    def setUp(self):
        self.g = STMGeometry(np.array([np.eye(6)] * 3), 5, 1.23, grid_size=11)

    def test_exact_budget_including_partial_initialization_and_generations(self):
        for solver in SOLVERS:
            for maximum in (1, 3, 4, 7, 13, 41):
                calls = []
                def objective(x):
                    calls.append(x.copy())
                    self.assertFalse(x.flags.writeable)
                    validate_chromosome(x, 5)
                    return float(x @ x)
                with self.subTest(solver=solver.__name__, maximum=maximum):
                    result = solver(objective, self.g, maximum, seed=9)
                    self.assertEqual(len(calls), maximum)
                    self.assertEqual(result.evaluations, maximum)
                    self.assertAlmostEqual(result.best_fitness, min(float(x @ x) for x in calls))
                    np.testing.assert_array_equal(result.best_chromosome, calls[result.best_found_at - 1])
                    self.assertEqual(result.history[-1].evaluations, maximum)
                    self.assertTrue(all(a.best_fitness >= b.best_fitness
                                        for a, b in zip(result.history, result.history[1:])))
                    self.assertFalse(result.best_chromosome.flags.writeable)

    def test_seed_reproduces_vector_score_and_improvement_history(self):
        for solver in SOLVERS:
            with self.subTest(solver=solver.__name__):
                first = solver(lambda x: float(x @ x), self.g, 83, seed=73)
                second = solver(lambda x: float(x @ x), self.g, 83, seed=73)
                self.assertEqual(first.to_dict(), second.to_dict())

    def test_initialization_only_budgets_share_the_same_informed_samples(self):
        results = [solver(lambda x: float(x @ x), self.g, 3, seed=18)
                   for solver in (informed_random_search, informed_genetic_algorithm,
                                  informed_differential_evolution)]
        for result in results[1:]:
            np.testing.assert_array_equal(result.best_chromosome, results[0].best_chromosome)
            self.assertEqual(result.best_fitness, results[0].best_fitness)

    def test_restart_paths_remain_budgeted_and_keep_earliest_tied_incumbent(self):
        for solver in (informed_genetic_algorithm, informed_differential_evolution, sage_hybrid):
            options = dict(population_size=4)
            options.update(local_restart_after=2 if solver is sage_hybrid else 600)
            if solver is not sage_hybrid:
                del options['local_restart_after']
                options['restart_generations'] = 1
            seen = []
            def constant(x):
                seen.append(x.copy())
                return 0.
            with self.subTest(solver=solver.__name__):
                result = solver(constant, self.g, 45, seed=1, **options)
                self.assertEqual(len(seen), 45)
                self.assertEqual(result.zero_fitness_evaluations, 45)
                self.assertEqual(result.best_found_at, 1)
                np.testing.assert_array_equal(result.best_chromosome, seen[0])
                self.assertEqual([p.evaluations for p in result.history], [1, 45])

    def test_nonfinite_and_nonreal_objectives_fail_explicitly(self):
        for solver in SOLVERS:
            for value in (float('nan'), float('inf'), True, [0.]):
                with self.subTest(solver=solver.__name__, value=value), self.assertRaises(ObjectiveEvaluationError):
                    solver(lambda x: value, self.g, 1)

    def test_bad_common_arguments_fail_before_objective_calls(self):
        for solver in SOLVERS:
            with patch('builtins.print'):
                for bad in (0, -1, True, 2.5):
                    with self.assertRaises(ValueError):
                        solver(lambda x: self.fail('Must not evaluate'), self.g, bad)
                with self.assertRaises(ValueError):
                    solver(lambda x: self.fail('Must not evaluate'), self.g, 5, seed=True)

    def test_invalid_algorithm_settings(self):
        cases = [(informed_random_search, dict(informed_fraction=1.1)),
                 (informed_genetic_algorithm, dict(crossover='gene')),
                 (informed_genetic_algorithm, dict(population_size=4, elite_count=4)),
                 (informed_genetic_algorithm, dict(mutation_cells=np.nan)),
                 (informed_differential_evolution, dict(population_size=3)),
                 (informed_differential_evolution, dict(F=0)),
                 (informed_differential_evolution, dict(CR=1.1)),
                 (informed_differential_evolution, dict(adaptive='yes')),
                 (sage_hybrid, dict(genetic_fraction=1.1)),
                 (sage_hybrid, dict(local_restart_after=0))]
        for solver, options in cases:
            with self.subTest(solver=solver.__name__, options=options), self.assertRaises(ValueError):
                solver(lambda x: self.fail('Must not evaluate'), self.g, 4, **options)

    def test_satellite_crossover_never_separates_state_components(self):
        first = np.arange(30).reshape(5, 6).astype(float)
        second = first + 100
        child = satellite_crossover(first, second, np.random.default_rng(4))
        for index in range(5):
            self.assertTrue(np.array_equal(child[index], first[index]) or
                            np.array_equal(child[index], second[index]))
        np.testing.assert_array_equal(first, np.arange(30).reshape(5, 6))

    def test_de_forces_exactly_one_satellite_at_zero_cr(self):
        child = satellite_binomial_crossover(np.zeros((5, 6)), np.ones((5, 6)),
                                             0., np.random.default_rng(8))
        self.assertEqual(np.count_nonzero(np.sum(child, axis=1)), 1)
        self.assertEqual(np.sum(child), 6)

    def test_convex_blend_preserves_polytope_constraints(self):
        rng = np.random.default_rng(77)
        first, second = self.g.sample(rng), self.g.sample(rng)
        for _ in range(10):
            child = satellite_crossover(first, second, rng, kind='blend')
            self.assertLess(np.max(np.abs(child @ self.g.whitened_constraints.T)), 1)

    def test_hybrid_ablation_and_all_genetic_endpoint_use_exact_budget(self):
        for fraction in (0., 1.):
            result = sage_hybrid(lambda x: float(x @ x), self.g, 51,
                                 seed=4, genetic_fraction=fraction, population_size=4)
            self.assertEqual(result.evaluations, 51)

    def test_uniform_mixture_endpoint_is_full_official_box(self):
        seen = []
        def objective(x):
            seen.append(x.copy())
            return 0.
        informed_random_search(objective, self.g, 4, seed=4, informed_fraction=0)
        rng = np.random.default_rng(4)
        for vector in seen:
            np.testing.assert_array_equal(vector, self.g.space.to_physical(rng.random(30)))


class TestAdvancedESA(unittest.TestCase):
    def test_all_advanced_algorithms_on_each_official_instance_and_recheck_accounting(self):
        for name in ('beginner', 'intermediate', 'hard'):
            evaluator = ESAEvaluator(name)
            for variant in ('stm-random', 'stm-ga', 'stm-de', 'sage', 'stm-local'):
                with self.subTest(instance=name, variant=variant):
                    with patch.object(evaluator, 'evaluate', wraps=evaluator.evaluate) as objective:
                        record = run_variant(evaluator, variant, 17, seed=51, population_size=4)
                        self.assertEqual(objective.call_count, 17)
                    validate_chromosome(record['best_chromosome'], evaluator.n_sat)
                    self.assertEqual(record['post_search_fitness_calls'], 1)
                    self.assertTrue(record['official_verification_matches'])
                    self.assertEqual(record['best_fitness'], evaluator.udp.fitness(record['best_chromosome'])[0])
                    self.assertLess(record['best_fitness'], 0)

    def test_real_reproducibility(self):
        evaluator = ESAEvaluator('beginner')
        for variant in ('stm-ga', 'stm-de', 'sage'):
            first = run_variant(evaluator, variant, 53, seed=9, population_size=4)
            second = run_variant(evaluator, variant, 53, seed=9, population_size=4)
            for key in ('best_chromosome', 'best_fitness', 'history', 'best_found_at'):
                self.assertEqual(first[key], second[key])


if __name__ == '__main__':
    unittest.main()
