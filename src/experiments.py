"""Reproducible ESA runs and equal-search-budget comparisons.

Run: python -m src.experiments --instances beginner --budget 1000 --plot
No warm-up fitness calls or post-search reevaluations are hidden in the budget.
Every reported best value was observed during the counted optimization run.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.differential_evolution import differential_evolution
from src.evaluator import ESAEvaluator, ROOT, SOURCE_FILES, environment_details, verify_official_sources
from src.genetic_algorithm import genetic_algorithm
from src.optimization import integer_parameter
from src.random_search import random_search


SOLVERS = ("random-search", "ga", "de")


def run_optimization(
    evaluator: ESAEvaluator,
    solver: str,
    max_evaluations: int,
    *,
    seed: int = 0,
    population_size: int = 40,
    ga_options: dict[str, Any] | None = None,
    de_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reuse an existing evaluator and record a single genuine search run."""
    if solver not in SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}; choose {SOLVERS}.")
    # Importing signal loads lazy SciPy machinery without spending a fitness call.
    import scipy.signal  # noqa: F401

    common = dict(seed=seed)
    if solver == "random-search":
        function, parameters = random_search, {}
    elif solver == "ga":
        function = genetic_algorithm
        parameters = dict(population_size=population_size, tournament_size=2,
                          crossover_rate=0.9, crossover_alpha=0.25,
                          mutation_probability=1.0 / (6 * evaluator.n_sat),
                          mutation_sigma=0.05, elite_count=1, n_satellites=evaluator.n_sat)
        parameters.update(ga_options or {})
    else:
        function = differential_evolution
        parameters = dict(population_size=population_size, F=0.7, CR=0.9)
        parameters.update(de_options or {})

    started = perf_counter()
    result = function(evaluator.evaluate, evaluator.get_bounds(), max_evaluations,
                      **common, **parameters)
    seconds = perf_counter() - started
    record = result.to_dict()
    record.update({
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instance": evaluator.instance,
        "solver": solver,
        "problem_id": evaluator.config["problem_id"],
        "source_commit": evaluator.config["source_commit"],
        "source_sha256": evaluator.config["source_sha256"],
        "objective_direction": "minimize",
        "max_evaluations": int(max_evaluations),
        "parameters": parameters,
        "initialization": "uniform-full-official-box",
        "boundary_repair": "reflection" if solver != "random-search" else "not-needed",
        "history_kind": "strict-improvements-plus-final-evaluation",
        "elapsed_seconds": seconds,
        "post_search_fitness_calls": 0,
    })
    return record


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON atomically; nonfinite values are an error."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_run(path: Path, record: dict[str, Any]) -> None:
    """Also export a flat vector that the foundation's --chromosome accepts."""
    write_json(path, record)
    write_json(path.with_suffix(".best.json"), record["best_chromosome"])


def summarize_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["instance"], record["solver"])].append(record)
    summary = []
    for (instance, solver), group in groups.items():
        scores = [record["best_fitness"] for record in group]
        summary.append({
            "instance": instance, "solver": solver, "runs": len(group),
            "evaluations_per_run": group[0]["evaluations"],
            "best": min(scores), "median": statistics.median(scores),
            "mean": statistics.mean(scores), "worst": max(scores),
            "sample_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "mean_seconds": statistics.mean(record["elapsed_seconds"] for record in group),
            "mean_zero_fitness_fraction": statistics.mean(
                record["zero_fitness_evaluations"] / record["evaluations"] for record in group
            ),
        })
    return summary


def history_curve(record: dict[str, Any]) -> np.ndarray:
    """Expand sparse best-so-far changes for plotting, without new evaluations."""
    indexes = np.array([point["evaluations"] for point in record["history"]])
    values = np.array([point["best_fitness"] for point in record["history"]])
    evaluations = np.arange(1, record["evaluations"] + 1)
    return values[np.searchsorted(indexes, evaluations, side="right") - 1]


def plot_progress(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for instance in dict.fromkeys(record["instance"] for record in records):
        figure, axes = plt.subplots(figsize=(8, 4.5))
        for solver in SOLVERS:
            selected = [record for record in records
                        if record["instance"] == instance and record["solver"] == solver]
            if selected:
                curves = np.array([history_curve(record) for record in selected])
                axes.step(np.arange(1, curves.shape[1] + 1), np.median(curves, axis=0),
                          where="post", label=solver)
        axes.set(xlabel="Objective evaluations", ylabel="Median best objective (minimize)",
                 title=f"ESA {instance}: equal budgets and matched seeds")
        axes.grid(alpha=0.25)
        axes.legend()
        figure.tight_layout()
        figure.savefig(output / f"progress-{instance}.png", dpi=160)
        plt.close(figure)


def compare(
    instances: list[str], seeds: list[int], algorithms: list[str],
    max_evaluations: int, output: Path, *, population_size: int = 40,
    ga_options: dict[str, Any] | None = None, de_options: dict[str, Any] | None = None,
    plot: bool = False,
) -> list[dict[str, Any]]:
    """Create one evaluator per instance and compare independent seeded runs."""
    max_evaluations = integer_parameter(max_evaluations, "max_evaluations")
    seeds = [integer_parameter(seed, "seed", minimum=0) for seed in seeds]
    for label, values, allowed in (("instances", instances, SOURCE_FILES),
                                   ("algorithms", algorithms, SOLVERS)):
        if not values or len(set(values)) != len(values) or any(x not in allowed for x in values):
            raise ValueError(f"{label} must be a nonempty list of distinct known values.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct nonnegative integers.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instances": instances, "seeds": seeds, "algorithms": algorithms,
        "search_evaluations_per_run": max_evaluations,
        "population_size": population_size,
        "environment": environment_details(),
        "verified_sources": verify_official_sources(),
        "counting": "Initialization included; no uncounted objective calls.",
        "initialization": "Uniform full official box for every solver; no injected solution.",
        "evaluator_initialization_seconds": {},
    }
    try:
        metadata["project_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        metadata["project_tracked_files_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        metadata["project_commit"] = None
    write_json(output / "metadata.json", metadata)

    records = []
    for instance in instances:
        evaluator = ESAEvaluator(instance)
        metadata["evaluator_initialization_seconds"][instance] = evaluator.initialization_seconds
        for seed in seeds:
            for solver in algorithms:
                record = run_optimization(evaluator, solver, max_evaluations, seed=seed,
                                          population_size=population_size,
                                          ga_options=ga_options, de_options=de_options)
                name = f"{instance}-{solver}-seed{seed}-budget{max_evaluations}.json"
                write_run(output / name, record)
                records.append(record)
                print(f"{instance:12} {solver:13} seed={seed:5} "
                      f"evaluations={record['evaluations']} best={record['best_fitness']:.12g}", flush=True)
    summary = summarize_runs(records)
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(output / "metadata.json", metadata)
    if plot:
        plot_progress(records, output)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Equal-budget ESA optimizer comparison")
    parser.add_argument("--instances", nargs="+", choices=tuple(SOURCE_FILES), default=["beginner"])
    parser.add_argument("--algorithms", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--tournament-size", type=int, default=2)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--crossover-alpha", type=float, default=0.25)
    parser.add_argument("--mutation-probability", type=float)
    parser.add_argument("--mutation-sigma", type=float, default=0.05)
    parser.add_argument("--elite-count", type=int, default=1)
    parser.add_argument("--de-f", type=float, default=0.7)
    parser.add_argument("--de-cr", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    ga_options = dict(tournament_size=args.tournament_size, crossover_rate=args.crossover_rate,
                      crossover_alpha=args.crossover_alpha, mutation_sigma=args.mutation_sigma,
                      elite_count=args.elite_count)
    if args.mutation_probability is not None:
        ga_options["mutation_probability"] = args.mutation_probability
    try:
        compare(args.instances, args.seeds, args.algorithms, args.budget, args.output,
                population_size=args.population_size, ga_options=ga_options,
                de_options=dict(F=args.de_f, CR=args.de_cr), plot=args.plot)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Experiment failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
