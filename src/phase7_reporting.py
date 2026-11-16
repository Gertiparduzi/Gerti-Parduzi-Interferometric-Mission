"""Generate publication-style plots and independently verified best artifacts.

All plotted coordinates and scores come from saved, executed runs. Nothing in
this module is imported by an optimizer. PDF/PNG plots can be regenerated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.evaluator import ESAEvaluator
from src.experiments import history_curve, write_json, write_run
from src.formation_diagnostics import PLANES, observe_formation


LABELS = {'random-search': 'Uniform random', 'ga': 'Baseline GA', 'de': 'Baseline DE',
          'stm-random': 'STM random', 'stm-ga': 'STM GA', 'stm-de': 'STM DE',
          'sage': 'SAGE hybrid', 'stm-local': 'STM local'}
COLORS = dict(zip(LABELS, ['#909090', '#333333', '#b5a390', '#0072b2',
                         '#56b4e9', '#cc79a7', '#d55e00', '#009e73']))


def load_runs(directory):
    records = []
    for path in sorted(Path(directory).glob('*budget*.json')):
        if path.name.endswith('.best.json'):
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        if 'best_chromosome' in record and 'evaluations' in record:
            record['_source_file'] = str(path)
            records.append(record)
    return records


def _save_plot(figure, output, name, plt):
    figure.savefig(output / (name + '.png'), dpi=180, bbox_inches='tight')
    figure.savefig(output / (name + '.pdf'), bbox_inches='tight')
    plt.close(figure)


def export_best(root, records):
    """Two explicitly logged fresh objective calls per final winner."""
    root = Path(root)
    output = root / 'best'
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for instance in ('beginner', 'intermediate', 'hard'):
        candidates = [r for r in records if r['instance'] == instance]
        if not candidates:
            raise ValueError(f'No executed run available for {instance}.')
        winner = min(candidates, key=lambda r: (r['best_fitness'], r['best_found_at']))
        source_file = Path(winner['_source_file'])
        flat = json.loads(source_file.with_suffix('.best.json').read_text(encoding='utf-8'))
        if flat != winner['best_chromosome']:
            raise ValueError(f'Flat saved vector disagrees with run record: {source_file}')
        evaluator = ESAEvaluator(instance)
        wrapper = evaluator.evaluate(flat)
        direct = float(evaluator.udp.fitness(flat)[0])
        if direct != wrapper or direct != winner['best_fitness']:
            raise RuntimeError('Fresh official best-vector verification failed.')
        observations = observe_formation(evaluator, flat)
        if not np.isclose(direct, -min(o['sum_fill_factors'] for o in observations), atol=1e-14, rtol=0):
            raise RuntimeError('Independent diagnostic support counts disagree.')
        record = {k: v for k, v in winner.items() if not k.startswith('_')}
        coordinates = np.asarray([o['positions'] for o in observations])
        shifted_cells = coordinates * evaluator.udp.grid_size / 2 + evaluator.udp.grid_size // 2
        record.update(selected_from=str(source_file.relative_to(root)),
                      final_wrapper_fitness=wrapper, final_direct_esa_fitness=direct,
                      final_verification_matches=True, final_additional_objective_calls=2,
                      observations=observations,
                      minimum_observation_boundary_margin=float(1 - np.max(np.abs(coordinates))),
                      minimum_grid_boundary_distance=float(np.min(np.abs(shifted_cells - np.rint(shifted_cells)))))
        write_run(output / f'{instance}.json', record)
        index.append(dict(instance=instance, fitness=direct, solver=winner['solver'], seed=winner['seed'],
                          stage=winner.get('stage'), evaluations=winner['evaluations'],
                          best_found_at=winner['best_found_at'],
                          chromosome_file=f'best/{instance}.best.json',
                          chromosome_sha256=hashlib.sha256((output / f'{instance}.best.json').read_bytes()).hexdigest(),
                          all_satellites_survive=all(o['survivor_count'] == evaluator.n_sat for o in observations),
                          official_match=True))
    write_json(root / 'best-index.json', index)
    with (root / 'best-index.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    return index


def make_plots(root):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from itertools import product, combinations

    root = Path(root)
    output = root / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    records = load_runs(root / 'confirmation')
    if not records:
        raise ValueError('No confirmation records to plot.')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 10.5), sharex=True)
    for ax, instance in zip(axes, ('beginner', 'intermediate', 'hard')):
        for variant in LABELS:
            selected = [r for r in records if r['instance'] == instance and r['solver'] == variant]
            if not selected:
                continue
            curves = np.array([history_curve(r) for r in selected])
            evaluations = np.arange(1, curves.shape[1] + 1)
            ax.step(evaluations, np.median(curves, axis=0), where='post',
                    color=COLORS[variant], label=LABELS[variant], lw=1.7,
                    linestyle='--' if variant in ('random-search', 'ga', 'de') else '-')
            if variant in ('sage', 'stm-local'):
                ax.fill_between(evaluations, curves.min(axis=0), curves.max(axis=0),
                                color=COLORS[variant], alpha=.09, step='post')
        ax.set_title(instance.capitalize(), loc='left', fontweight='bold')
        ax.set_ylabel('Best ESA fitness (lower is better)')
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Counted objective evaluations (initialization included)')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .005))
    figure.suptitle('Phase 7 confirmation: median over seeds 101, 202, 303\nShading: seed range for hybrid and local search', fontsize=13)
    figure.tight_layout(rect=(0, .07, 1, .95))
    _save_plot(figure, output, 'convergence', plt)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for ax, instance in zip(axes, ('beginner', 'intermediate', 'hard')):
        variants = [v for v in LABELS if any(r['instance'] == instance and r['solver'] == v for r in records)]
        for index, variant in enumerate(variants):
            values = np.array([r['best_fitness'] for r in records if r['instance'] == instance and r['solver'] == variant])
            ax.scatter(index + np.linspace(-.12, .12, len(values)), values, s=35, color=COLORS[variant], zorder=3)
            ax.plot([index - .25, index + .25], [np.median(values)] * 2, color=COLORS[variant], lw=2.5)
        ax.set_xticks(range(len(variants)), [LABELS[v] for v in variants], rotation=55, ha='right')
        ax.set_title(instance.capitalize(), fontweight='bold')
        ax.set_ylabel('Final fitness at 10,000 evaluations')
        ax.grid(axis='y', alpha=.2)
    figure.suptitle('Equal-budget comparison: each dot is one real seed; bars show medians', fontsize=13)
    figure.tight_layout()
    _save_plot(figure, output, 'algorithm-comparison', plt)

    for instance in ('beginner', 'intermediate', 'hard'):
        record = json.loads((root / 'best' / f'{instance}.json').read_text())
        observations = record['observations']
        n = len(record['best_chromosome']) // 6
        grid = 11 if instance == 'beginner' else 21
        figure, axes = plt.subplots(3, 3, figsize=(10.5, 10))
        for row, observation in enumerate(observations):
            cells = np.asarray(observation['cells'], dtype=int).reshape(-1, 3)
            for column, ((first, second), label) in enumerate(zip(PLANES, ('XY', 'XZ', 'YZ'))):
                ax = axes[row, column]
                ax.scatter(cells[:, first], cells[:, second], s=35, marker='s', color='#0072b2')
                ax.set(xlim=(-.5, grid - .5), ylim=(-.5, grid - .5), aspect='equal',
                       xlabel=label[0] + ' grid cell', ylabel=label[1] + ' grid cell')
                ax.set_xticks(np.arange(grid), minor=True)
                ax.set_yticks(np.arange(grid), minor=True)
                ax.grid(which='minor', alpha=.12)
                ax.set_title(f"t={row}T, {label}: coverage {observation['plane_fill_factors'][column]:.4f}", fontsize=10)
        figure.suptitle(f"{instance.capitalize()}: saved best, ESA fitness {record['best_fitness']:.9f}\n"
                       f"Exact occupied grid cells; all {n} satellites survive each observation", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, .94))
        _save_plot(figure, output, f'projections-{instance}', plt)

        figure = plt.figure(figsize=(14, 4.8))
        for column, observation in enumerate(observations):
            ax = figure.add_subplot(1, 3, column + 1, projection='3d')
            positions = np.asarray(observation['positions'])
            for start, end in combinations(np.array(list(product((-1, 1), repeat=3))), 2):
                if np.sum(np.abs(start - end)) == 2:
                    ax.plot3D(*zip(start, end), color='#9aa4ad', alpha=.45, lw=.7)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=np.arange(n), cmap='viridis', s=30)
            ax.scatter([0], [0], [0], marker='x', color='black', s=35)
            ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel='X', ylabel='Y', zlabel='Z',
                   title=f"t={column}T: {observation['survivor_count']}/{n} inside")
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=38)
        figure.suptitle(f'{instance.capitalize()}: observation coordinates normalized exactly as ESA\n'
                       'Later epochs divided by the single inflation factor 1.23', fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, .9))
        _save_plot(figure, output, f'formation-3d-{instance}', plt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/phase7'))
    args = parser.parse_args()
    records = []
    for directory in sorted(args.root.iterdir()):
        if directory.is_dir() and directory.name not in ('best', 'previous-baseline'):
            records.extend(load_runs(directory))
    print(json.dumps(export_best(args.root, records), indent=2))
    make_plots(args.root)


if __name__ == '__main__':
    main()
