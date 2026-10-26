"""Linear regression on the same feature sets and folds as the Tier 3 kernel ridge, for slide 19.

The report's E10 (report_v2/code/E10_subsets.py) scores RBF kernel ridge on each of the seven
non-empty subsets of {S_t, log R_t, A_t}, with each of the 74 cells held out in turn. It records a
straight line only for R_t and log R_t alone. This script fits ordinary least squares on all seven
subsets, through E10's own held_out function, so the runs, the statistics at t* = 2,430 and the
folds are the same. It also fits two richer straight lines on all three statistics: one with a
squared term for S_t, and one that adds the three pairwise products, so the slide can say how much
each adds. The kernel ridge scores are read from E10.json and are not refitted.

Run it from the repository root with the environment of pyproject.toml:

    .venv/bin/python presentation/fit_tier3_linear.py

It writes presentation/assets/tier3_linear.json. make_figures.py draws slide 19's chart from it.
"""
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'report_v2' / 'code'
sys.path.insert(0, str(CODE))
import E10_subsets as E10  # noqa: E402
import models as M  # noqa: E402

OUT = HERE / 'assets' / 'tier3_linear.json'


def main():
    data = M.load_data()
    grok = data[data.grokked == 1].reset_index(drop=True)
    grok['logT'] = np.log(grok['T'])
    assert len(grok) == 355
    subsets = [list(c) for k in (1, 2, 3) for c in itertools.combinations(['S_t', 'logR_t', 'A_t'], k)]
    linear = {}
    for cols in subsets:
        name = ' + '.join(E10.NAMES[c] for c in cols)
        linear[name] = E10.held_out(grok, cols, 'linear')
        print(f'linear {name:16s} R2 {linear[name]["R2"]:.3f}', flush=True)
    # The richer lines take z-scores, as in model_fitting/4_nonlinear_exploration.ipynb. Ordinary
    # least squares gives the same predictions with or without them, since each term is a linear
    # function of the others' rescaled forms.
    z = (grok[['S_t', 'logR_t', 'A_t']] - grok[['S_t', 'logR_t', 'A_t']].mean()) / grok[['S_t', 'logR_t', 'A_t']].std()
    grok = grok.assign(zS=z.S_t, zR=z.logR_t, zA=z.A_t)
    grok = grok.assign(zSS=grok.zS ** 2, zSR=grok.zS * grok.zR, zSA=grok.zS * grok.zA, zRA=grok.zR * grok.zA)
    richer = {
        'line + S squared': E10.held_out(grok, ['zS', 'zR', 'zA', 'zSS'], 'linear'),
        'line + products': E10.held_out(grok, ['zS', 'zR', 'zA', 'zSR', 'zSA', 'zRA'], 'linear'),
        'line + S squared + products': E10.held_out(grok, ['zS', 'zR', 'zA', 'zSS', 'zSR', 'zSA', 'zRA'], 'linear'),
    }
    for name, r in richer.items():
        print(f'{name:28s} R2 {r["R2"]:.3f}', flush=True)
    kernel = json.loads((CODE / 'out' / 'numbers' / 'E10.json').read_text())['results']['subsets']
    OUT.write_text(json.dumps({
        'linear': linear, 'richer_lines': richer, 'kernel_ridge_from_E10': kernel,
        'n_runs': len(grok), 'n_cells': int(grok.cell.nunique()),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scikit-learn': sklearn.__version__},
    }, indent=2) + '\n')
    print('wrote', OUT.relative_to(HERE.parent))


if __name__ == '__main__':
    main()
