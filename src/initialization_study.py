"""Measure negative-fitness probability and survival without optimizing."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.evaluator import ESAEvaluator
from src.experiments import write_json
from src.formation_diagnostics import observe_formation
from src.stm_initialization import STMGeometry
from src.optimization import integer_parameter


def study(output, samples=100, seeds=(901, 902, 903)):
    samples = integer_parameter(samples, 'samples')
    seeds = [integer_parameter(s, 'seed', minimum=0) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be nonempty and distinct.')
    output = Path(output)
    if (output / 'summary.json').exists():
        raise ValueError('Study already exists; choose a new directory.')
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for instance in ('beginner', 'intermediate', 'hard'):
        evaluator = ESAEvaluator(instance)
        geometry = STMGeometry.from_evaluator(evaluator)
        for method in ('uniform', 'stm'):
            selected = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                for index in range(samples):
                    x = (geometry.space.to_physical(rng.random(6 * evaluator.n_sat)) if method == 'uniform'
                         else geometry.to_chromosome(geometry.sample(rng)))
                    score = evaluator.evaluate(x)
                    observed = observe_formation(evaluator, x)
                    if not np.isclose(score, -min(o['sum_fill_factors'] for o in observed), atol=1e-14, rtol=0):
                        raise RuntimeError('Diagnostic pair counts disagree with ESA.')
                    row = dict(instance=instance, method=method, seed=seed, sample=index, fitness=score,
                               survivors_t0=observed[0]['survivor_count'], survivors_t1=observed[1]['survivor_count'],
                               survivors_t2=observed[2]['survivor_count'])
                    rows.append(row)
                    selected.append(row)
            record = dict(instance=instance, method=method, samples=len(selected),
                          negative_fraction=sum(r['fitness'] < 0 for r in selected) / len(selected),
                          median_fitness=float(np.median([r['fitness'] for r in selected])),
                          best_fitness=min(r['fitness'] for r in selected),
                          mean_survivors=[float(np.mean([r[f'survivors_t{k}'] for r in selected])) for k in range(3)],
                          all_satellites_survive_fraction=sum(all(r[f'survivors_t{k}'] == evaluator.n_sat
                              for k in range(3)) for r in selected) / len(selected))
            summary.append(record)
            print(json.dumps(record), flush=True)
    write_json(output / 'summary.json', summary)
    write_json(output / 'metadata.json', dict(seeds=seeds, samples_per_seed_method_instance=samples,
              objective_evaluations=len(rows), diagnostic_only=True,
              warning='Empirical frequencies on finite samples, not universal probabilities. No search or winner selection.'))
    with (output / 'samples.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/phase7/initialization'))
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seeds', nargs='+', type=int, default=[901, 902, 903])
    args = parser.parse_args()
    study(args.output, args.samples, args.seeds)
