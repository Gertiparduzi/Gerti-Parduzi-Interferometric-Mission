"""Display the problem configuration before adding the solvers."""

import argparse
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "instances.json"


def main() -> None:
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

    args = parser.parse_args()
    config = instances[args.instance]

    period = config["period"]
    times = [index * period for index in range(config["n_meas"])]

    print("Interferometric Mission - initial project setup")
    print(f"Instance: {args.instance}")
    print(f"ESA problem ID: {config['problem_id']}")
    print(f"Mothership orbit: {config['orbit']}")
    print(f"Satellites: {config['n_sat']}")
    print(f"Decision variables: {6 * config['n_sat']}")
    print(f"Grid: {config['grid_size']} x {config['grid_size']}")
    print(f"Observation times: {times}")
    print("Observation planes: XY, XZ, YZ")
    print("Status: configuration ready; evaluator and solvers are the next step.")


if __name__ == "__main__":
    main()