"""Phase 7 experiments: real official objectives, auditable costs and artifacts.

Example: python -m src.advanced_experiments --budget 2000 --seeds 11 22
Each saved winner is rechecked by the direct ESA UDP; these verification calls
are explicitly separate from the equal *search* budgets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from src.advanced_optimization import (
    informed_differential_evolution, informed_genetic_algorithm, informed_random_search,
)
from src.evaluator import ESAEvaluator, ROOT, SOURCE_FILES, environment_details, verify_official_sources
from src.experiments import run_optimization, summarize_runs, write_json, write_run
from src.hybrid_optimization import sage_hybrid
from src.optimization import integer_parameter
from src.stm_initialization import STMGeometry


BASE_COMMIT = '355adee126cff1ab9c4d2ece368e25042b542b7f'
VARIANTS = ('random-search', 'ga', 'de', 'stm-random', 'stm-mixed', 'stm-ga',
            'stm-ga-blend', 'stm-de', 'stm-de-fixed', 'sage', 'stm-local')


def source_fingerprint():
    """Include untracked new implementation files during a pilot as well."""
    paths = [ROOT / 'main.py', *sorted((ROOT / 'src').glob('*.py'))]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def process_peak_rss_mib():
    """Process lifetime high-water mark, NOT incremental/per-run algorithm memory."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 ** 2 if platform.system() == 'Darwin' else 1024)
    except ImportError:
        return None


def run_variant(evaluator, variant, budget, *, seed=0, population_size=40):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}.')
    import scipy.signal  # Load lazy machinery without an objective call.
    if variant in ('random-search', 'ga', 'de'):
        record = run_optimization(evaluator, variant, budget, seed=seed, population_size=population_size)
        record['geometry_setup_seconds'] = 0.0
        record['geometry'] = None
    else:
        started = perf_counter()
        geometry = STMGeometry.from_evaluator(evaluator)
        setup_seconds = perf_counter() - started
        parameters = {}
        if variant in ('stm-random', 'stm-mixed'):
            function = informed_random_search
            parameters = dict(informed_fraction=.8 if variant == 'stm-mixed' else 1.)
        elif variant in ('stm-ga', 'stm-ga-blend'):
            function = informed_genetic_algorithm
            parameters = dict(population_size=population_size, tournament_size=2, elite_count=2,
                              crossover_rate=.9, crossover='blend' if variant == 'stm-ga-blend' else 'satellite',
                              mutation_cells=2., replacement_probability=.15, restart_generations=20)
        elif variant in ('stm-de', 'stm-de-fixed'):
            function = informed_differential_evolution
            parameters = dict(population_size=population_size, F=.5, CR=.6,
                              adaptive=variant == 'stm-de', restart_generations=20)
        else:
            function = sage_hybrid
            parameters = dict(population_size=population_size,
                              genetic_fraction=0. if variant == 'stm-local' else .2,
                              local_restart_after=600)
        search_started = perf_counter()
        result = function(evaluator.evaluate, geometry, budget, seed=seed, **parameters)
        search_seconds = perf_counter() - search_started
        record = result.to_dict()
        record.update(schema_version=2, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      instance=evaluator.instance, solver=variant, parameters=parameters,
                      problem_id=evaluator.config['problem_id'], source_commit=evaluator.config['source_commit'],
                      source_sha256=evaluator.config['source_sha256'], objective_direction='minimize',
                      max_evaluations=budget, initialization='STM-whitened radial sampling',
                      boundary_repair='satellite radial contraction or feasible chord',
                      history_kind='strict-improvements-plus-final-evaluation',
                      elapsed_seconds=setup_seconds + search_seconds,
                      geometry_setup_seconds=setup_seconds, geometry=geometry.diagnostics())
    # This independent call is counted separately, never charged to search.
    started = perf_counter()
    verified = float(evaluator.udp.fitness(record['best_chromosome'])[0])
    record.update(post_search_fitness_calls=1, official_verification_fitness=verified,
                  official_verification_matches=verified == record['best_fitness'],
                  verification_seconds=perf_counter() - started,
                  process_peak_rss_mib=process_peak_rss_mib())
    if not record['official_verification_matches']:
        raise RuntimeError('Saved incumbent disagrees with the direct ESA evaluator.')
    return record


def compare(instances, seeds, variants, budget, output, *, population_size=40, stage='comparison'):
    budget = integer_parameter(budget, 'budget')
    population_size = integer_parameter(population_size, 'population_size', minimum=4)
    for label, values, allowed in (('instances', instances, SOURCE_FILES), ('variants', variants, VARIANTS)):
        if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
            raise ValueError(f'{label} must contain distinct known names.')
    seeds = [integer_parameter(s, 'seed', minimum=0) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be nonempty and distinct.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metadata.json').exists():
        raise ValueError('Output already contains an experiment. Choose a new directory.')
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                    base_commit=BASE_COMMIT, instances=instances, seeds=seeds, variants=variants,
                    search_budget_per_run=budget, population_size=population_size,
                    source_fingerprint=source_fingerprint(), environment=environment_details(),
                    official_sources=verify_official_sources(),
                    counting='All initialization/proposals/restarts counted. One separate direct ESA recheck per run.',
                    timing='elapsed_seconds includes STM geometry setup and search, excludes ESA import and final recheck.',
                    memory='process_peak_rss_mib is cumulative process lifetime RSS high-water, not per-run memory.',
                    prior_candidates_injected=False, evaluator_initialization_seconds={}, completed=False)
    try:
        metadata['implementation_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        metadata['git_status'] = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        metadata['implementation_commit'] = None
    write_json(output / 'metadata.json', metadata)
    records = []
    for instance in instances:
        evaluator = ESAEvaluator(instance)
        metadata['evaluator_initialization_seconds'][instance] = evaluator.initialization_seconds
        for seed in seeds:
            for variant in variants:
                record = run_variant(evaluator, variant, budget, seed=seed, population_size=population_size)
                record['stage'] = stage
                write_run(output / f'{instance}-{variant}-seed{seed}-budget{budget}.json', record)
                records.append(record)
                print(f"{instance:12} {variant:13} seed={seed:4} budget={budget:6} "
                      f"best={record['best_fitness']:.12f} seconds={record['elapsed_seconds']:.2f}", flush=True)
    summary = summarize_runs(records)
    write_json(output / 'summary.json', summary)
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata.update(completed=True, runs=len(records), search_evaluations=sum(r['evaluations'] for r in records),
                    verification_evaluations=len(records),
                    all_official_rechecks_match=all(r['official_verification_matches'] for r in records))
    write_json(output / 'metadata.json', metadata)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', nargs='+', choices=tuple(SOURCE_FILES), default=list(SOURCE_FILES))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS,
                        default=['random-search', 'ga', 'de', 'stm-random', 'stm-ga', 'stm-de', 'sage'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--budget', type=int, default=2000)
    parser.add_argument('--population-size', type=int, default=40)
    parser.add_argument('--stage', default='comparison')
    parser.add_argument('--output', type=Path, default=Path('results/phase7/comparison'))
    args = parser.parse_args()
    try:
        compare(args.instances, args.seeds, args.variants, args.budget, args.output,
                population_size=args.population_size, stage=args.stage)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f'Experiment failed: {error}\n')


if __name__ == '__main__':
    main()
