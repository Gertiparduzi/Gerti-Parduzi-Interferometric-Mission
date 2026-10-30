import json
import unittest
from pathlib import Path


CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "instances.json"


class TestConfiguration(unittest.TestCase):

    def test_instances(self):
        with open(CONFIG_FILE, encoding="utf-8") as file:
            instances = json.load(file)

        for name in ("beginner", "intermediate", "hard"):
            with self.subTest(instance=name):
                self.assertIn(name, instances)

                config = instances[name]

                self.assertGreater(config["n_sat"], 0)
                self.assertEqual(len(config["ic"]), 6)
                self.assertGreater(config["grid_size"], 0)
                self.assertEqual(config["n_meas"], 3)


if __name__ == "__main__":
    unittest.main()