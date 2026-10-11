"""Rebuild every figure, table and number of the report from the slim data extract.

Steps:
1. Check the extract in data/ against data/SHA256SUMS.json.
2. Run every item script in an order that respects what each one reads from another.
3. Write overleaf/numbers.tex from the JSON outputs.
4. Copy out/figures/*.pdf to overleaf/figures/ and out/tables/*.tex to overleaf/tables/.
5. Write environment.txt and the run time of each script to out/logs/.

On macOS the script starts caffeinate, which keeps the computer awake until the script ends.
The whole run takes about 20 minutes on an Apple M-series laptop, most of it in E06's bootstrap.

Run:  python make_all.py            (from this folder, in an environment with the packages of README.md)
"""
import json
import os
import shutil
import subprocess
import sys
import time

import common as C
from extract import sha256

ORDER = ['E01_baseline', 'E03_placebo', 'E02_thresholds', 'E04_map', 'E06_aft', 'E05_scaling', 'E07_mechanism',
         'E08_ladder', 'E09_surface', 'E10_subsets', 'E12_kumar', 'E13_grid', 'E15_settings', 'E16_levels',
         'N_numbers']


def keep_awake():
    if sys.platform == 'darwin' and shutil.which('caffeinate'):
        return subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    return None


def check_extract():
    sums = json.loads((C.SLIM / 'SHA256SUMS.json').read_text())
    for name, digest in sums.items():
        if sha256(C.SLIM / name) != digest:
            raise SystemExit(f'make_all: {name} does not match SHA256SUMS.json; rerun extract.py')
    print(f'make_all: the extract matches its hash list ({len(sums)} files)')


def main():
    awake = keep_awake()
    check_extract()
    logs = C.OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    times = {}
    for script in ORDER + ['make_numbers']:
        t0 = time.time()
        with open(logs / f'{script}.log', 'w') as log:
            done = subprocess.run([sys.executable, '-u', f'{script}.py'], cwd=C.CODE, stdout=log,
                                  stderr=subprocess.STDOUT)
        times[script] = round(time.time() - t0, 1)
        print(f'make_all: {script:16s} {times[script]:7.1f} s', flush=True)
        if done.returncode:
            raise SystemExit(f'make_all: {script} failed; see out/logs/{script}.log')

    for src, dst, pattern in ((C.OUT / 'figures', C.OVERLEAF / 'figures', '*.pdf'),
                              (C.OUT / 'tables', C.OVERLEAF / 'tables', '*.tex')):
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob(pattern)):
            shutil.copyfile(f, dst / f.name)
    (logs / 'times.json').write_text(json.dumps(times, indent=1) + '\n')
    env = C.versions()
    (C.CODE / 'environment.txt').write_text(''.join(f'{k}=={v}\n' if k not in ('python', 'platform') else
                                                    f'# {k}: {v}\n' for k, v in env.items()))
    print(f'make_all: done in {sum(times.values()) / 60:.1f} min')
    if awake:
        awake.terminate()


if __name__ == '__main__':
    main()
