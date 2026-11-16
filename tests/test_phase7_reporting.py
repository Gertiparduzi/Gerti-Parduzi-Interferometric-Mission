"""Saved experiment evidence must be consistent before it is reported."""

import json
import tempfile
import unittest
from pathlib import Path

from src.advanced_experiments import compare, run_variant
from src.evaluator import ESAEvaluator
from src.experiments import write_run
from src.phase7_reporting import export_best, load_runs


class TestPhase7Reporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluators = {name: ESAEvaluator(name) for name in ('beginner', 'intermediate', 'hard')}

    def test_export_reload_and_direct_verification_for_all_winners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, evaluator in self.evaluators.items():
                record = run_variant(evaluator, 'stm-random', 2, seed=15)
                write_run(root / 'confirmation' / f'{name}-stm-random-seed15-budget2.json', record)
            records = load_runs(root / 'confirmation')
            self.assertEqual(len(records), 3)
            winners = export_best(root, records)
            for winner in winners:
                self.assertTrue(winner['official_match'])
                self.assertTrue(winner['all_satellites_survive'])
                detail = json.loads((root / 'best' / f"{winner['instance']}.json").read_text())
                self.assertEqual(detail['final_additional_objective_calls'], 2)
                self.assertGreater(detail['minimum_observation_boundary_margin'], 0)

    def test_mismatched_saved_vector_is_rejected(self):
        evaluator = self.evaluators['beginner']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = run_variant(evaluator, 'stm-random', 1, seed=15)
            path = root / 'confirmation' / 'beginner-stm-random-seed15-budget1.json'
            write_run(path, record)
            path.with_suffix('.best.json').write_text(json.dumps([0.] * 30))
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                export_best(root, load_runs(root / 'confirmation'))

    def test_experiment_does_not_overwrite_existing_study(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'metadata.json').write_text('{"keep":true}')
            with self.assertRaisesRegex(ValueError, 'already contains'):
                compare(['beginner'], [1], ['stm-random'], 1, path)
            self.assertEqual((path / 'metadata.json').read_text(), '{"keep":true}')

    def test_invalid_experiment_axes_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            for instances, seeds, variants in (([], [1], ['sage']), (['beginner'], [1, 1], ['sage']),
                                                (['beginner'], [1], ['missing'])):
                with self.assertRaises(ValueError):
                    compare(instances, seeds, variants, 1, Path(directory))


if __name__ == '__main__':
    unittest.main()
