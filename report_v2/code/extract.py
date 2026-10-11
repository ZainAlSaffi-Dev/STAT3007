"""Write the slim data extract that every item script reads.

The full sweep of 3 October 2026 is about 11 GB. It lives in the repository's data/ folder, which is
not in git; set STAT3007_DATA to read it from elsewhere. The report needs only a small part of it:

- report_runs.csv: one row per run, with the settings and the events (copied as it is);
- report_kernel.parquet: S, R, D and A at the 48 saved steps (copied as it is);
- manifest.json: the design, the fixed settings and the versions (copied as it is);
- sweep_timing.json: the elapsed wall time and the workers of the sweep, read from logs/sweep.log;
- checkpoints_slim.parquet: nine columns of checkpoints.parquet, for all 307 checkpoints of each run.

The script checks the extract before it writes the hash list: 720 runs, all with status ok, 307
checkpoints each on one shared grid, and events in report_runs.csv that match the first checkpoint
at which each accuracy reaches its level. make_all.py checks the hash list, so a reader who has only
the extract can confirm that it is the one used for the report.

Run:  python extract.py
"""
import hashlib
import json
import re
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import BUDGET, FULL_DATA, LEVELS, SLIM

COLUMNS = ['run_id', 'step', 'train_acc', 'test_acc', 'train_loss', 'test_loss', 'S_sum', 'R_sum', 'A_sum']
COPIED = ['report_runs.csv', 'report_kernel.parquet', 'manifest.json']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check(runs, ckpt):
    assert len(runs) == 720 and runs.run_id.is_unique, 'expected 720 runs'
    assert (runs.status == 'ok').all(), 'a run did not finish'
    counts = ckpt.groupby('run_id').size()
    assert (counts == 307).all() and len(counts) == 720, 'expected 307 checkpoints per run'
    grids = ckpt.groupby('run_id')['step'].apply(tuple)
    assert grids.nunique() == 1, 'the runs do not share one step grid'
    steps = np.array(grids.iloc[0])
    assert steps[0] == 0 and steps[-1] == BUDGET

    events = {'mem': ('train_acc', 0.99), **{k: ('test_acc', v) for k, v in LEVELS.items()}}
    wide = {c: ckpt.pivot(index='run_id', columns='step', values=c).loc[runs.run_id] for c in ('train_acc', 'test_acc')}
    for name, (col, level) in events.items():
        hit = wide[col].to_numpy() >= level
        reached = hit.any(axis=1)
        first = np.where(reached, steps[hit.argmax(axis=1)], BUDGET)
        assert (runs[f'censored_{name}'].to_numpy() == ~reached).all(), f'censoring of {name} differs'
        assert (runs[f't_{name}'].to_numpy() == first).all(), f'event steps of {name} differ'


def sweep_timing(log):
    """The elapsed wall time and the workers of the last run command in logs/sweep.log."""
    start = finish = workers = None
    for line in log.read_text().splitlines():
        m = re.match(r'(\S+ \S+) INFO\s+MainProcess\s+run: (\d+) runs, (\d+) wide and (\d+) narrow workers', line)
        if m:
            start, workers, finish = m.group(1), (int(m.group(3)), int(m.group(4))), None
        elif 'MainProcess        run: finished' in line and start:
            finish = line[:19]
    fmt = '%Y-%m-%d %H:%M:%S'
    elapsed = (datetime.strptime(finish, fmt) - datetime.strptime(start, fmt)).total_seconds()
    return {'start': start, 'finish': finish, 'elapsed_seconds': elapsed, 'wide_workers': workers[0],
            'narrow_workers': workers[1]}


def main():
    SLIM.mkdir(parents=True, exist_ok=True)
    ckpt = pq.read_table(FULL_DATA / 'checkpoints.parquet', columns=COLUMNS).to_pandas()
    ckpt = ckpt.sort_values(['run_id', 'step']).reset_index(drop=True)
    runs = pd.read_csv(FULL_DATA / 'report_runs.csv')
    check(runs, ckpt)

    ckpt.to_parquet(SLIM / 'checkpoints_slim.parquet', compression='zstd', index=False)
    for name in COPIED:
        shutil.copyfile(FULL_DATA / name, SLIM / name)
    (SLIM / 'sweep_timing.json').write_text(json.dumps(sweep_timing(FULL_DATA / 'logs' / 'sweep.log'), indent=1) + '\n')

    files = sorted(p.name for p in SLIM.iterdir() if p.name != 'SHA256SUMS.json')
    sums = {name: sha256(SLIM / name) for name in files}
    (SLIM / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=1) + '\n')
    size = sum((SLIM / n).stat().st_size for n in files) / 1e6
    print(f'extract: {len(runs)} runs, {len(ckpt)} checkpoint rows, {size:.1f} MB in {SLIM}')
    print('extract: events in report_runs.csv match the checkpoints')


if __name__ == '__main__':
    main()
