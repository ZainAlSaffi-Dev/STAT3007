"""Write the sweep as two CSV files that use the definitions of the golden report.

runs.csv uses older definitions. Its kernel columns come from the trace kernel on a mixed probe.
Its events are measured from memorisation, with memorisation at training accuracy 1. This script
uses report section 3.5 instead. The memorisation step is the first checkpoint at which training
accuracy reaches 0.99. The grokking step at level l is the first checkpoint at which test accuracy
reaches l, for l in {0.80, 0.90, 0.95}, counted from step 0. A run that does not reach a level is
censored at its budget. Each event also records the checkpoint before it, so the event lies in
(step_prev, step].

The kernel statistics S_t, R_t, D_t and A_t of report Equation 6 come from the checkpoint columns
S_sum, R_sum, D_sum and A_sum. report_runs.csv gives them at each event step and at the last step.
report_kernel.csv gives them at the 48 steps whose weights were saved, from report_kernel.parquet.
Each seed has its own split of the pairs, from data seed 42 + seed, as report section 3.1 asks.

Run from model_fitting/:  ../.venv/bin/python report_runs.py
It writes data/report_runs.csv and data/report_kernel.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / 'data'
EVENTS = {'mem': ('train_acc', 0.99), 'grok80': ('test_acc', 0.80),
          'grok90': ('test_acc', 0.90), 'grok95': ('test_acc', 0.95)}


KERNEL = ['S_sum', 'R_sum', 'D_sum', 'A_sum']


def events(run):
    """First checkpoint at which each accuracy reaches its level, the checkpoint before it, the censoring flag,
    and the kernel statistics at that checkpoint. They are empty for a censored event."""
    steps, row = run['step'].to_numpy(), {}
    for name, (column, level) in EVENTS.items():
        hit = np.flatnonzero(run[column].to_numpy() >= level)
        if len(hit):
            row[f't_{name}'] = int(steps[hit[0]])
            row[f't_{name}_prev'] = int(steps[hit[0] - 1]) if hit[0] > 0 else 0
            row[f'censored_{name}'] = False
            row.update({f'{k[0]}_t_at_{name}': run[k].iat[hit[0]] for k in KERNEL})
        else:
            row[f't_{name}'], row[f't_{name}_prev'] = int(steps[-1]), int(steps[-1])
            row[f'censored_{name}'] = True
            row.update({f'{k[0]}_t_at_{name}': np.nan for k in KERNEL})
    return row


def main():
    fixed = json.loads((DATA / 'manifest.json').read_text())['fixed']
    runs = pd.read_parquet(DATA / 'runs.parquet', columns=['run_id', 'alpha', 'width', 'eta_kappa', 'seed',
                                                           'data_seed', 'lr', 'steps', 'status'])
    curves = pd.read_parquet(DATA / 'checkpoints.parquet', columns=['run_id', 'step', 'train_acc', 'test_acc',
                                                                    'train_loss', 'test_loss', *KERNEL])
    rows = []
    for run_id, run in curves.sort_values(['run_id', 'step']).groupby('run_id', sort=False):
        last = run.iloc[-1]
        rows.append({'run_id': run_id, **events(run), 'train_acc_final': last.train_acc,
                     'test_acc_final': last.test_acc, 'train_loss_final': last.train_loss,
                     'test_loss_final': last.test_loss, **{f'{k[0]}_t_final': last[k] for k in KERNEL}})
    out = runs.rename(columns={'steps': 'budget'}).merge(pd.DataFrame(rows), on='run_id', validate='1:1')
    out.insert(5, 'eta_0', fixed['eta_0'])
    out.insert(6, 'p', fixed['p'])
    out.insert(7, 'train_fraction', fixed['train_fraction'])
    assert len(out) == 720 and out['S_t_final'].notna().all(), 'a run is missing or has no kernel statistics'

    kernel = pd.read_parquet(DATA / 'report_kernel.parquet')

    out.to_csv(DATA / 'report_runs.csv', index=False)
    kernel.merge(runs[['run_id', 'alpha', 'width', 'eta_kappa', 'seed']], on='run_id', validate='m:1') \
          .to_csv(DATA / 'report_kernel.csv', index=False)
    print(out.shape, 'rows in data/report_runs.csv;', len(kernel), 'rows in data/report_kernel.csv')


if __name__ == '__main__':
    main()
