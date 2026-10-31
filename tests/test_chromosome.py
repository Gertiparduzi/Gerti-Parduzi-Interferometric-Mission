import unittest

from src.chromosome import decode_chromosome


class TestChromosome(unittest.TestCase):

    def test_valid_chromosome(self):
        chromosome = list(range(1, 13))

        result = decode_chromosome(chromosome, 2)

        expected = [
            (1, 3, 5, 7, 9, 11),
            (2, 4, 6, 8, 10, 12)
        ]

        self.assertEqual(result, expected)

    def test_invalid_length(self):
        chromosome = [1, 2, 3]

        with self.assertRaises(ValueError):
            decode_chromosome(chromosome, 2)

    def test_invalid_satellite_count(self):
        chromosome = []

        with self.assertRaises(ValueError):
            decode_chromosome(chromosome, 0)

    def test_invalid_numeric_value(self):
        chromosome = [1, 2, 3, 4, 5, float("nan")]

        with self.assertRaises(ValueError):
            decode_chromosome(chromosome, 1)


if __name__ == "__main__":
    unittest.main()