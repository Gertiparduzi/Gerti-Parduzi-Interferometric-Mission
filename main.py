"""Inspect, evaluate or optimize ESA instances while preserving the foundation CLI."""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter

from src.evaluator import ESAEvaluator, environment_details, verify_official_sources


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "instances.json"


def main() -> int:
    instances = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(
        description="ESA SpOC Interferometric Mission"
    )
    parser.add_argument(
        "--instance",
        choices=tuple(instances),
        default="beginner",
        help="Problem instance to inspect (default: beginner).",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check-environment", action="store_true",
        help="Check real dependency imports, versions and ESA API compatibility.",
    )
    action.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate one candidate; defaults to a zero-vector smoke test.",
    )
    action.add_argument(
        "--optimize", choices=("random-search", "ga", "de"),
        help="Run one independently implemented optimizer.",
    )
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--chromosome", type=Path, help="JSON file containing a flat ESA vector.")
    parser.add_argument("--report", type=Path, help="Write evaluation or optimization evidence as JSON.")

    args = parser.parse_args()
    if args.chromosome and not args.evaluate:
        parser.error("--chromosome requires --evaluate.")
    if args.report and not (args.evaluate or args.optimize):
        parser.error("--report requires --evaluate or --optimize.")
    if args.check_environment:
        details = environment_details()
        print(json.dumps(details, indent=2))
        return 0 if details["all_imports_ok"] and details["reference_api_compatible"] else 1

    config = instances[args.instance]

    period = config["period"]
    times = [index * period for index in range(config["n_meas"])]

    print("Interferometric Mission - ESA reference evaluation")
    print(f"Instance: {args.instance}")
    print(f"ESA problem ID: {config['problem_id']}")
    print(f"Mothership orbit: {config['orbit']}")
    print(f"Satellites: {config['n_sat']}")
    print(f"Decision variables: {6 * config['n_sat']}")
    print(f"Grid: {config['grid_size']} x {config['grid_size']}")
    print(f"Observation times: {times}")
    print("Observation planes: XY, XZ, YZ")
    if not (args.evaluate or args.optimize):
        print("Status: configuration loaded; use --check-environment, --evaluate or --optimize.")
        return 0

    if args.optimize:
        from src.experiments import run_optimization, write_run
        try:
            evaluator = ESAEvaluator(args.instance)
            record = run_optimization(evaluator, args.optimize, args.budget, seed=args.seed,
                                      population_size=args.population_size)
            record["environment"] = environment_details()
            record["evaluator_initialization_seconds"] = evaluator.initialization_seconds
            report = args.report or Path(
                f"results/{args.instance}-{args.optimize}-seed{args.seed}-budget{args.budget}.json"
            )
            write_run(report, record)
            print(f"Best objective (minimize): {record['best_fitness']:.17g}")
            print(f"Objective evaluations: {record['evaluations']}")
            print(f"Search time: {record['elapsed_seconds']:.6f} s")
            print(f"Report: {report}")
            print(f"Best vector: {report.with_suffix('.best.json')}")
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(2, f"Optimization failed: {error}\n")
        return 0

    try:
        if args.chromosome:
            chromosome = json.loads(args.chromosome.read_text(encoding="utf-8"))
            candidate_kind = "provided-json-vector"
        else:
            chromosome = [0.0] * (6 * config["n_sat"])
            candidate_kind = "zero-vector-smoke-test"

        evaluator = ESAEvaluator(args.instance)
        started = perf_counter()
        fitness = evaluator.fitness(chromosome)
        evaluation_seconds = perf_counter() - started
        direct = [float(value) for value in evaluator.udp.fitness(chromosome)]
        if direct != fitness:
            raise RuntimeError("Wrapper disagrees with the direct ESA reference call.")

        print(f"Candidate: {candidate_kind}")
        print(f"Objective (minimize): {fitness[0]:.17g}")
        print("Direct ESA agreement: True")
        print(f"Initialization: {evaluator.initialization_seconds:.6f} s")
        print(f"One evaluation: {evaluation_seconds:.6f} s")

        if args.report:
            report = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "instance": args.instance,
                "problem_id": config["problem_id"],
                "source_commit": config["source_commit"],
                "verified_source_sha256": verify_official_sources()[args.instance],
                "n_sat": config["n_sat"],
                "decision_variables": len(chromosome),
                "observation_times": times,
                "candidate_kind": candidate_kind,
                "chromosome": chromosome,
                "fitness": fitness,
                "objective_direction": "minimize",
                "objective_definition": "-min(sum_xy_xz_yz_at_each_epoch)",
                "direct_esa_fitness": direct,
                "direct_reference_match": fitness == direct,
                "initialization_seconds": evaluator.initialization_seconds,
                "evaluation_seconds": evaluation_seconds,
                "environment": environment_details(),
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Report: {args.report}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
