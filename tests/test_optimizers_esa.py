"""Cross-solver contracts and real ESA integration; no surrogate fitness."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.chromosome import validate_chromosome
from src.differential_evolution import differential_evolution
from src.evaluator import ESAEvaluator, SOURCE_FILES
from src.experiments import history_curve, run_optimization, summarize_runs, write_run
from src.genetic_algorithm import genetic_algorithm
from src.random_search import random_search


class TestOptimizerESAIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluators = {name: ESAEvaluator(name) for name in SOURCE_FILES}

    def test_all_solvers_run_on_all_three_real_instances_with_exact_budgets(self):
        for name, evaluator in self.evaluators.items():
            for solver in ("random-search", "ga", "de"):
                with self.subTest(instance=name, solver=solver):
                    with patch.object(evaluator, "evaluate", wraps=evaluator.evaluate) as objective:
                        record = run_optimization(evaluator, solver, 19, seed=71, population_size=8)
                        self.assertEqual(objective.call_count, 19)
                    self.assertEqual(record["evaluations"], 19)
                    validate_chromosome(record["best_chromosome"], evaluator.n_sat)
                    self.assertEqual(record["best_fitness"], evaluator.udp.fitness(record["best_chromosome"])[0])
                    self.assertEqual(record["history"][-1]["evaluations"], 19)
                    curve = history_curve(record)
                    self.assertTrue(np.all(curve[1:] <= curve[:-1]))

    def test_real_esa_runs_are_exactly_reproducible(self):
        evaluator = self.evaluators["beginner"]
        for solver in ("random-search", "ga", "de"):
            first = run_optimization(evaluator, solver, 23, seed=9, population_size=8)
            second = run_optimization(evaluator, solver, 23, seed=9, population_size=8)
            for key in ("best_chromosome", "best_fitness", "history", "evaluations"):
                with self.subTest(solver=solver, field=key):
                    self.assertEqual(first[key], second[key])

    def test_short_budgets_share_the_same_initial_random_sample_prefix(self):
        objective = lambda x: float(x @ x)
        functions = (random_search, genetic_algorithm, differential_evolution)
        results = [function(objective, ([-1] * 3, [1] * 3), 3, seed=11) for function in functions]
        for result in results[1:]:
            np.testing.assert_array_equal(result.best_chromosome, results[0].best_chromosome)
            self.assertEqual(result.best_fitness, results[0].best_fitness)

    def test_exported_best_is_a_flat_vector_accepted_by_foundation(self):
        evaluator = self.evaluators["beginner"]
        record = run_optimization(evaluator, "ga", 12, seed=7, population_size=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "with spaces" / "run.json"
            write_run(path, record)
            restored = json.loads(path.read_text())
            vector = json.loads(path.with_suffix(".best.json").read_text())
            self.assertEqual(restored, record)
            validate_chromosome(vector, 5)
            self.assertEqual(evaluator.evaluate(vector), record["best_fitness"])

    def test_summary_and_sparse_history_expansion_use_minimization(self):
        records = []
        for fitness in (-0.1, -0.3, -0.2):
            records.append(dict(instance="beginner", solver="ga", best_fitness=fitness,
                                evaluations=10, elapsed_seconds=2.0, zero_fitness_evaluations=5,
                                history=[dict(evaluations=1, best_fitness=0.0),
                                         dict(evaluations=5, best_fitness=fitness),
                                         dict(evaluations=10, best_fitness=fitness)]))
        summary = summarize_runs(records)[0]
        self.assertEqual(summary["best"], -0.3)
        self.assertEqual(summary["median"], -0.2)
        self.assertAlmostEqual(summary["mean"], -0.2)
        self.assertEqual(summary["mean_zero_fitness_fraction"], 0.5)
        np.testing.assert_array_equal(history_curve(records[0]), [0.0] * 4 + [-0.1] * 6)

    def test_unknown_solver_is_rejected_without_an_objective_call(self):
        evaluator = self.evaluators["beginner"]
        with patch.object(evaluator, "evaluate") as objective:
            with self.assertRaises(ValueError):
                run_optimization(evaluator, "unknown", 3)
            objective.assert_not_called()


if __name__ == "__main__":
    unittest.main()
