"""E10. Kernel ridge on subsets of the statistics, and the variance split (Table 5, Section 4.4).

The runs are the 355 that grok at level 0.95 among those with eta lambda at most 1e-3, with the
statistics at t* = 2,430. For each of the seven non-empty subsets of {S_t, log R_t, A_t}, RBF kernel
ridge predicts log T. Each cell is held out in turn (74 cells). Inside each fold the penalty and the
width are chosen by grouped 5-fold cross-validation over the training cells (models.kernel_ridge).
The scores are R^2 of log T and the concordance index on the held-out predictions.

The variance split is the share of the spread of log T, S_t, log R_t and A_t that lies between cells,
1 - SS_within / SS_total (model_fitting/4_nonlinear_exploration.ipynb, Part 2).
"""
import itertools
import warnings

import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score

import common as C
import models as M

SEEDS = {}
NAMES = {'S_t': 'S', 'logR_t': 'log R', 'A_t': 'A'}


def main():
    data = M.load_data()
    grok = data[data.grokked == 1].reset_index(drop=True)
    grok['logT'] = np.log(grok['T'])
    assert len(grok) == 355
    subsets = [list(c) for k in (1, 2, 3) for c in itertools.combinations(['S_t', 'logR_t', 'A_t'], k)]
    res = {}
    for cols in subsets:
        p = np.zeros(len(grok))
        for cell in grok.cell.unique():
            test = (grok.cell == cell).to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = M.kernel_ridge().fit(grok.loc[~test, cols], grok.logT[~test], groups=grok.cell[~test])
            p[test] = m.predict(grok.loc[test, cols])
        name = ' + '.join(NAMES[c] for c in cols)
        res[name] = {'R2': float(r2_score(grok.logT, p)), 'concordance': float(concordance_index(grok.logT, p))}
        print(f'{name:16s} R2 {res[name]["R2"]:.3f}  C {res[name]["concordance"]:.3f}', flush=True)
    split = {q: C.between_share(grok, q, 'cell') for q in ('logT', 'S_t', 'logR_t', 'A_t')}
    print('between cells', {k: round(v, 3) for k, v in split.items()})
    C.write_json('E10', {'subsets': res, 'between_cells': split, 'n_runs': len(grok),
                         'n_cells': int(grok.cell.nunique())}, SEEDS)


if __name__ == '__main__':
    main()
