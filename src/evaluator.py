"""Checked integration of unchanged, revision-pinned ESA reference evaluators.

No dynamics or objective formula is reimplemented here. Scientific dependencies
are loaded only when requested, keeping configuration and layout tests usable
before installation.
"""

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from time import perf_counter

from src.chromosome import validate_chromosome


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "instances.json"
ESA_REVISION = "b45d23bb1950212510fde3d7b6159e9f9f6fbad4"
SOURCE_FILES = {
    "beginner": "golomb_simple.py",
    "intermediate": "golomb_medium.py",
    "hard": "golomb_hard.py",
}
DEPENDENCIES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "PIL": "pillow",
    "heyoka": "heyoka",
    "pygmo": "pygmo",
}


def load_instance(name):
    """Read a fresh configuration without importing scientific libraries."""
    instances = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if name not in SOURCE_FILES or name not in instances:
        raise ValueError(f"Unknown instance {name!r}; choose {tuple(SOURCE_FILES)}.")
    config = instances[name]
    if config.get("source_commit") != ESA_REVISION:
        raise ValueError("The ESA source revision does not match this integration.")
    return config


def verify_official_sources():
    """Verify vendored Python files against the pinned SHA-256 checksums."""
    verified = {}
    for name, filename in SOURCE_FILES.items():
        config = load_instance(name)
        path = ROOT / "third_party" / "esa_spoc3" / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config["source_sha256"]:
            raise ValueError(f"ESA source checksum mismatch: {filename}.")
        verified[name] = digest
    return verified


def environment_details():
    """Check actual imports, including binary dependencies, and record versions."""
    packages = {}
    for module_name, distribution in DEPENDENCIES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            if version is None:
                version = importlib.metadata.version(distribution)
            packages[distribution] = {"status": "ok", "version": str(version)}
        except (ImportError, OSError, importlib.metadata.PackageNotFoundError) as error:
            packages[distribution] = {"status": "unavailable", "error": str(error)}
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "all_imports_ok": all(item["status"] == "ok" for item in packages.values()),
        "reference_api_compatible": packages.get("heyoka", {}).get("version") == "3.2.0",
    }


class ESAEvaluator:
    """Delegate validated candidates to ESA's original UDP.

    fitness(x) preserves the official one-objective list contract [f].
    evaluate(x) returns the same objective as a scalar. Lower values are better.
    Python caches imported modules and their precomputed STMs. The measured
    initialization time therefore reflects actual cold or warm construction.
    """

    def __init__(self, instance="beginner"):
        started = perf_counter()
        self.instance = instance
        self.config = load_instance(instance)
        verify_official_sources()
        module_name = "third_party.esa_spoc3." + Path(SOURCE_FILES[instance]).stem
        try:
            heyoka = importlib.import_module("heyoka")
            if heyoka.__version__ != "3.2.0":
                raise RuntimeError(
                    "This unchanged ESA revision requires heyoka.py=3.2.0. "
                    "Run: conda env update -n interferometric-mission -f environment.yml"
                )
            module = importlib.import_module(module_name)
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Cannot load the official ESA evaluator. In the existing Conda "
                "environment run: conda env update -n interferometric-mission "
                "-f environment.yml. Original error: " + str(error)
            ) from error
        self.udp = module.udp
        self.n_sat = self.config["n_sat"]
        self._check_configuration()
        self.initialization_seconds = perf_counter() - started

    def _check_configuration(self):
        expected = {
            "n_sat": self.config["n_sat"],
            "ic": tuple(self.config["ic"]),
            "T": self.config["period"] * (self.config["n_meas"] - 1),
            "n_meas": self.config["n_meas"],
            "grid_size": self.config["grid_size"],
            "mu": self.config["mu"],
            "scaling_factor": self.config["scaling_factor"],
            "inflation_factor": self.config["inflation_factor"],
        }
        for attribute, value in expected.items():
            actual = getattr(self.udp, attribute)
            if attribute == "ic":
                actual = tuple(actual)
            if actual != value:
                raise ValueError(f"Configuration disagrees with ESA: {attribute}.")

    def get_bounds(self):
        return self.udp.get_bounds()

    def get_nix(self):
        return self.udp.get_nix()

    def fitness(self, chromosome):
        """Return ESA's [f] without repairing or reordering the decision vector."""
        validate_chromosome(chromosome, self.n_sat, check_bounds=True)
        result = self.udp.fitness(chromosome)
        if len(result) != 1 or not math.isfinite(float(result[0])):
            raise RuntimeError("ESA evaluator did not return one finite objective.")
        return [float(result[0])]

    def evaluate(self, chromosome):
        """Return the scalar objective to minimize."""
        return self.fitness(chromosome)[0]
