"""Render the frozen structured-sector results, including the model bound."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from run_study import DARK, MODELS, OMEGA, effective, sha


def main(run, output):
    if output.exists():
        raise FileExistsError('Choose a new figure directory')
    report = json.loads((run/'summary.json').read_text())
    rows = list(csv.DictReader((run/'trials.csv').open()))
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9})
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.5), constrained_layout=True)
    a, b, c, d = axes.ravel()
    for name, bath, scale in MODELS:
        value = effective(OMEGA, bath, scale)
        a.plot(OMEGA, value.real, label=name.replace('_', ' '))
        b.plot(OMEGA, value.imag/(2*OMEGA))
    a.axhline(0, c='.6', lw=.6)
    a.set(xlabel='Frequency', ylabel='Re $Q$', title='(a) Compressed hidden kernel')
    a.legend(fontsize=6.8, loc='upper left')
    b.set(xlabel='Frequency', ylabel='Im $Q/(2\\omega)$', title='(b) Apparent damping')
    positions = np.arange(5); bottom = np.zeros(5)
    handles = []
    for status, color in [('inconsistent', '#4c78a8'), ('compatible', '#f2b447')]:
        values = [sum(g[status] for g in report['groups'] if g['model']==m[0]) for m in MODELS]
        c.bar(positions, values, bottom=bottom, color=color); bottom += values
        handles.append(Patch(facecolor=color, label=status))
    c.set(xticks=positions, xticklabels=['isolated', 'near weak', 'near strong', 'far weak', 'far strong'],
          ylim=(0, 44), ylabel='Retained cases', title='(c) Isolated-mode test')
    c.tick_params(axis='x', labelrotation=35)
    c.legend(handles=handles, fontsize=7, loc='upper center', ncol=2)
    for model_index, model in enumerate(('far_weak', 'far_strong')):
        row = next(r for r in rows if r['model']==model and float(r['field'])==.1
                   and float(r['epsilon'])==.001 and int(r['replicate'])==0)
        for bound_index, prefix in enumerate(('', 'remainder_')):
            low, high = float(row[prefix+'omega_lower']), float(row[prefix+'omega_upper'])
            midpoint = (low+high)/2; y = 3-2*model_index-bound_index
            d.errorbar(midpoint, y, xerr=[[midpoint-low], [high-midpoint]],
                       fmt='o', markersize=3, capsize=3, color=('#4c78a8' if prefix else '#c44e52'))
    d.axvline(DARK, color='.2', ls='--', lw=.9, label='Bare frequency')
    d.set(xlabel='Natural-frequency interval', yticks=[3, 2, 1, 0],
          yticklabels=['weak: isolated', 'weak: bounded', 'strong: isolated', 'strong: bounded'],
          ylim=(-.6, 3.8), title='(d) Independent bath bound')
    d.tick_params(axis='y', labelsize=7)
    d.legend(fontsize=7, loc='upper right')
    output.mkdir(parents=True)
    for ext in ('pdf', 'png'):
        fig.savefig(output/f'structured_sector.{ext}', dpi=200)
    plt.close(fig)
    inputs = [run/'summary.json', run/'trials.csv', Path(__file__), Path(__file__).with_name('run_study.py')]
    manifest = dict(inputs={str(p): sha(p) for p in inputs},
                    outputs={p.name: sha(p) for p in output.iterdir()},
                    interval_selection='B=.1, epsilon=.001, replicate=0, both far-bath strengths')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); main(args.run.resolve(), args.output.resolve())
