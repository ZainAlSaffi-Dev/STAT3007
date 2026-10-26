"""E10. Kernel ridge on subsets of the statistics, and the variance split (Table 5, Section 4.4).

The runs are the 355 that grok at level 0.95 among those with eta lambda at most 1e-3, with the
statistics at t* = 2,430. For each of the seven non-empty subsets of {S_t, log R_t, A_t}, RBF kernel
ridge predicts log T. Each cell is held out in turn (74 cells). Inside each fold the penalty and the
width are chosen by grouped 5-fold cross-validation over the training cells (models.kernel_ridge).
The scores are R^2 of log T and the concordance index on the held-out predictions.

The variance split is the share of the spread of log T, S_t, log R_t and A_t that lies between cells,
1 - SS_within / SS_total (model_fitting/4_nonlinear_exploration.ipynb, Part 2).

The raw R_t check refits with R_t on its own scale, to show why Tier 3 uses log R_t (the backup slide
on log R_t, after model_fitting/4_nonlinear_exploration.ipynb, Part 3). A straight line in R_t alone
and kernel ridge on all three statistics are each fitted once with raw R_t, plus the straight line in
log R_t alone. It also records how R_t at t* is spread.
"""
import itertools
import warnings

import numpy as np
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import common as C
import models as M

SEEDS = {}
NAMES = {'S_t': 'S', 'logR_t': 'log R', 'A_t': 'A'}


def held_out(grok, cols, model):
    """R^2 and concordance of log T, with each cell predicted by a model fitted on the other cells."""
    p = np.zeros(len(grok))
    for cell in grok.cell.unique():
        test = (grok.cell == cell).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if model == 'linear':
                m = LinearRegression().fit(grok.loc[~test, cols], grok.logT[~test])
            else:
                m = M.kernel_ridge().fit(grok.loc[~test, cols], grok.logT[~test], groups=grok.cell[~test])
        p[test] = m.predict(grok.loc[test, cols])
    return {'R2': float(r2_score(grok.logT, p)), 'concordance': float(concordance_index(grok.logT, p))}


def main():
    data = M.load_data()
    grok = data[data.grokked == 1].reset_index(drop=True)
    grok['logT'] = np.log(grok['T'])
    assert len(grok) == 355
    subsets = [list(c) for k in (1, 2, 3) for c in itertools.combinations(['S_t', 'logR_t', 'A_t'], k)]
    res = {}
    for cols in subsets:
        name = ' + '.join(NAMES[c] for c in cols)
        res[name] = held_out(grok, cols, 'kernel ridge')
        print(f'{name:16s} R2 {res[name]["R2"]:.3f}  C {res[name]["concordance"]:.3f}', flush=True)
    raw = {'linear R': held_out(grok, ['R_t'], 'linear'), 'linear log R': held_out(grok, ['logR_t'], 'linear'),
           'kernel ridge R': held_out(grok, ['R_t'], 'kernel ridge'),
           'kernel ridge S + R + A': held_out(grok, ['S_t', 'R_t', 'A_t'], 'kernel ridge')}
    print('raw R_t', {k: round(v['R2'], 3) for k, v in raw.items()})
    raw['R_t'] = {'min': float(grok.R_t.min()), 'median': float(grok.R_t.median()), 'max': float(grok.R_t.max()),
                  'share_below_0.01': float((grok.R_t < 0.01).mean())}
    split = {q: C.between_share(grok, q, 'cell') for q in ('logT', 'S_t', 'logR_t', 'A_t')}
    print('between cells', {k: round(v, 3) for k, v in split.items()})
    C.write_json('E10', {'subsets': res, 'raw_R': raw, 'between_cells': split, 'n_runs': len(grok),
                         'n_cells': int(grok.cell.nunique())}, SEEDS)


if __name__ == '__main__':
    main()
