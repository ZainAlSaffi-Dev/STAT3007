"""N1, N3 and N4: numbers quoted in Table 2 and Section 3.4 that no figure script owns.

N1: the compute of the sweep. total_seconds in manifest.json is the sum over runs of each run's own
wall time (repoduced-code/dataset_sweep.py, line 824), so it counts the parallel workers separately.
The elapsed wall time comes from the first and last lines of the run command in logs/sweep.log.

N3: the gap between an event and the checkpoint before it, as a share of the event step, for
memorisation and for grokking at 0.95, over the runs that reach the event. It bounds how far the true
event can lie before the recorded one.

N4: in the baseline cell (alpha = 1, N = 100, eta lambda = 0), the range of the test loss over the
checkpoints at and after the first one with perfect test accuracy, over all seeds.
"""
import json

import numpy as np

import common as C

SEEDS = {}


def main():
    m = C.load_manifest()
    timing = json.loads((C.SLIM / 'sweep_timing.json').read_text())
    n1 = {'summed_run_seconds': m['total_seconds'], 'summed_run_hours': m['total_seconds'] / 3600,
          'elapsed_seconds': timing['elapsed_seconds'], 'elapsed_hours': timing['elapsed_seconds'] / 3600,
          'workers': timing['wide_workers'] + timing['narrow_workers'], **timing,
          'platform': m['versions']['platform'], 'machine': m['versions']['machine'],
          'python': m['versions']['python'], 'torch': m['versions']['torch']}

    runs = C.load_runs()
    n3 = {}
    for ev in ('mem', 'grok95'):
        r = runs[~runs[f'censored_{ev}']]
        gap = (r[f't_{ev}'] - r[f't_{ev}_prev']) / r[f't_{ev}']
        n3[ev] = {'max': float(gap.max()), 'median': float(gap.median()),
                  'max_by_alpha': {f'{a:g}': float(g.max()) for a, g in gap.groupby(r.alpha)},
                  'max_above_1000': float(gap[r[f't_{ev}'] >= 1000].max())}

    base = runs[(runs.alpha == 1.0) & (runs.width == 100) & (runs.eta_kappa == 0)]
    ck = C.load_checkpoints(['run_id', 'step', 'test_acc', 'test_loss'])
    ck = ck[ck.run_id.isin(base.run_id)]
    after = []
    for rid, g in ck.groupby('run_id'):
        hit = np.flatnonzero(g.test_acc.to_numpy() >= 1.0)
        if len(hit):
            after.append(g.test_loss.to_numpy()[hit[0]:])
    loss = np.concatenate(after)
    n4 = {'min': float(loss.min()), 'max': float(loss.max()), 'seeds_with_perfect_test': len(after),
          'final_test_loss': base.test_loss_final.tolist()}
    C.write_json('N', {'N1': n1, 'N3': n3, 'N4': n4}, SEEDS)
    print('N1', {k: n1[k] for k in ('summed_run_hours', 'elapsed_hours', 'workers', 'platform')})
    print('N3', n3)
    print('N4', n4)


if __name__ == '__main__':
    main()
