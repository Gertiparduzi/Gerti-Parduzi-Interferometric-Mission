"""Validation, component ordering and lossless conversion beyond the first tests."""

import json
import math
import random
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np

from src.chromosome import (
    chromosome_bounds,
    decode_chromosome,
    encode_chromosome,
    validate_chromosome,
)


class TestChromosomeValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "config" / "instances.json"
        cls.instances = json.loads(path.read_text(encoding="utf-8"))

    def test_encode_known_component_order(self):
        rows = [(1, 3, 5, 7, 9, 11), (2, 4, 6, 8, 10, 12)]
        self.assertEqual(encode_chromosome(rows), list(range(1, 13)))

    def test_round_trips_for_all_instances(self):
        generator = random.Random(20261115)
        for name, config in self.instances.items():
            n_sat = config["n_sat"]
            lower, upper = chromosome_bounds(n_sat)
            for repetition in range(4):
                with self.subTest(instance=name, repetition=repetition):
                    x = [generator.uniform(a, b) for a, b in zip(lower, upper)]
                    rows = decode_chromosome(x, n_sat, check_bounds=True)
                    self.assertEqual(len(rows), n_sat)
                    self.assertEqual(encode_chromosome(rows, n_sat, check_bounds=True), x)
                    self.assertEqual(decode_chromosome(encode_chromosome(rows), n_sat), rows)

    def test_invalid_satellite_counts(self):
        for value in (0, -1, True, False, 2.0, 2.5, "2", None, 1 + 0j):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                decode_chromosome([], value)

    def test_invalid_numeric_values(self):
        values = (float("nan"), float("inf"), -float("inf"), True, False,
                  "0", None, 0 + 0j, Decimal("0"), np.bool_(True))
        for value in values:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                validate_chromosome([0.0] * 5 + [value], 1)

    def test_overflowing_integer_is_rejected_cleanly(self):
        with self.assertRaises(ValueError):
            validate_chromosome([0] * 5 + [10 ** 10000], 1)

    def test_invalid_vector_containers(self):
        for value in (None, "000000", b"000000", {i: 0 for i in range(6)},
                      iter([0] * 6), [[0] for _ in range(6)]):
            with self.subTest(container=type(value).__name__), self.assertRaises(ValueError):
                validate_chromosome(value, 1)

    def test_wrong_lengths_for_all_instances(self):
        for name, config in self.instances.items():
            n_sat = config["n_sat"]
            for length in (6 * n_sat - 1, 6 * n_sat + 1):
                with self.subTest(instance=name, length=length), self.assertRaises(ValueError):
                    validate_chromosome([0.0] * length, n_sat)

    def test_inclusive_decision_bounds_for_all_instances(self):
        for name, config in self.instances.items():
            n_sat = config["n_sat"]
            with self.subTest(instance=name):
                for vector in chromosome_bounds(n_sat):
                    self.assertIsNone(validate_chromosome(vector, n_sat))

    def test_each_component_rejects_values_just_outside_bounds(self):
        for component, limit in enumerate((1.0, 1.0, 1.0, 10.0, 10.0, 10.0)):
            for sign in (-1, 1):
                x = [0.0] * 12
                x[component * 2 + 1] = math.nextafter(sign * limit, sign * math.inf)
                with self.subTest(component=component, sign=sign), self.assertRaises(ValueError):
                    validate_chromosome(x, 2)

    def test_layout_examples_can_opt_in_to_decision_bounds(self):
        x = list(range(1, 13))
        rows = decode_chromosome(x, 2)
        self.assertEqual(encode_chromosome(rows), x)
        with self.assertRaises(ValueError):
            decode_chromosome(x, 2, check_bounds=True)
        with self.assertRaises(ValueError):
            encode_chromosome(rows, check_bounds=True)

    def test_bad_satellite_row_shapes_and_counts(self):
        cases = [([], None), ([[0] * 5], None), ([[0] * 7], None),
                 ([[0] * 6], 2), ([None], None), (["000000"], None)]
        for rows, count in cases:
            with self.subTest(rows=rows, count=count), self.assertRaises(ValueError):
                encode_chromosome(rows, count)

    def test_numpy_integer_count_is_supported(self):
        self.assertEqual(decode_chromosome([0.0] * 6, np.int64(1)), [(0.0,) * 6])
        with self.assertRaises(ValueError):
            decode_chromosome([], np.bool_(True))

    def test_read_only_noncontiguous_float64_view_preserves_values(self):
        original = np.linspace(-0.9, 0.9, 60, dtype=np.float64)
        before = original.copy()
        view = original[::2]
        view.flags.writeable = False
        rows = decode_chromosome(view, 5, check_bounds=True)
        result = encode_chromosome(rows, check_bounds=True)
        np.testing.assert_array_equal(result, view)
        np.testing.assert_array_equal(original, before)
        self.assertIsInstance(result[0], np.float64)

    def test_float32_scalars_are_not_silently_retyped(self):
        x = np.linspace(-0.5, 0.5, 30, dtype=np.float32)
        result = encode_chromosome(decode_chromosome(x, 5), check_bounds=True)
        np.testing.assert_array_equal(result, x)
        self.assertIsInstance(result[0], np.float32)

    def test_float64_precision_and_negative_zero_are_retained(self):
        x = [math.nextafter(0.5, 1.0), -0.0, 0.0, 0.0, 0.0, 0.0]
        result = encode_chromosome(decode_chromosome(x, 1), check_bounds=True)
        self.assertEqual(result, x)
        self.assertEqual(math.copysign(1.0, result[1]), -1.0)

    def test_bounds_flag_must_be_boolean(self):
        with self.assertRaises(ValueError):
            validate_chromosome([0.0] * 6, 1, check_bounds="yes")


if __name__ == "__main__":
    unittest.main()
