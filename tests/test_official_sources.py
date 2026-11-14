"""Verify provenance and numeric configuration without executing ESA imports."""

import ast
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluator import ESA_REVISION, ROOT, SOURCE_FILES, load_instance, verify_official_sources


class TestOfficialSources(unittest.TestCase):
    def test_all_three_source_hashes(self):
        self.assertEqual(set(verify_official_sources()), set(SOURCE_FILES))

    def test_config_matches_literal_constants_in_esa_sources(self):
        fields = {"n_sat": "N", "ic": "ic", "period": "period", "n_meas": "M",
                  "grid_size": "grid_size", "mu": "mu", "scaling_factor": "scaling_factor",
                  "inflation_factor": "inflation_factor"}
        for name, filename in SOURCE_FILES.items():
            path = ROOT / "third_party" / "esa_spoc3" / filename
            constants = {}
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    try:
                        constants[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
            config = load_instance(name)
            for local, official in fields.items():
                with self.subTest(instance=name, field=local):
                    self.assertEqual(config[local], constants[official])

    def test_sources_are_pinned_to_one_revision(self):
        for name in SOURCE_FILES:
            config = load_instance(name)
            self.assertEqual(config["source_commit"], ESA_REVISION)
            self.assertIn("/blob/" + ESA_REVISION + "/", config["source_file"])

    def test_modified_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "third_party" / "esa_spoc3"
            destination.mkdir(parents=True)
            for filename in SOURCE_FILES.values():
                shutil.copyfile(ROOT / "third_party" / "esa_spoc3" / filename,
                                destination / filename)
            file = destination / SOURCE_FILES["beginner"]
            file.write_bytes(file.read_bytes() + b"\n# modified\n")
            with patch("src.evaluator.ROOT", root), self.assertRaisesRegex(ValueError, "checksum"):
                verify_official_sources()

    def test_unknown_instance_is_rejected(self):
        with self.assertRaises(ValueError):
            load_instance("unknown")


if __name__ == "__main__":
    unittest.main()
