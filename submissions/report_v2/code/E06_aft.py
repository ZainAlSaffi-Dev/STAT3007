"""E6. Scaling fits (Table 4, Section 4.3). Also feeds E5 and E16.

At alpha = 1 the script fits the log-normal accelerated failure time model of report Eq. 5,

    log T = c + a x + b y + gamma x y + sigma eps,   x = log N - x0,   y = log(eta lambda) - y0,

by maximum likelihood with lifelines' LogNormalAFTFitter. A censored run enters through the
probability that its step exceeds the budget. Both logarithms are natural and centred at the middle
of the main grid: x0 is the mean of log N over the six widths and y0 the mean of log(eta lambda)
over 1e-5, 3e-5 and 1e-4. The same centre is used for every row, so a and b are always the slopes
at that point.

Rows:
- main: eta lambda in {1e-5, 3e-5, 1e-4} at all six widths, level 0.95 (90 runs);
- with 3e-4: the main cells and eta lambda = 3e-4;
- level 0.80 and level 0.90: the main cells at those levels;
- memorisation: the main cells, with the memorisation step as T.

Intervals are 95% percentile intervals from 1,000 bootstrap draws that resample whole cells with
replacement and then seeds within each drawn cell. This lets the misfit of the cells about the plane
enter the interval. Draws whose design cannot identify the four coefficients (fewer than two widths
or two decays among the drawn cells) are redrawn. A second bootstrap resamples seeds within cells and
keeps the cells fixed, as the planning values did; it is written for comparison.

Beside sigma the script gives the spread between seeds within cells (median within-cell SD of log T
over uncensored cells) and the misfit of the cells (SD of the cell means of the residuals).

Check: on the main row the fit must reproduce a = +0.237 and b = -0.633 of
repoduced-code/results/tier2_consistency.json, part C, which used the same runs and a hand-written
fit with the product term centred in the same way.
"""
import warnings

import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter

import common as C

N_BOOT = 1_000
SEED = 0
SEEDS = {'bootstrap': SEED, 'n_boot': N_BOOT}
MAIN_DECAYS = [1e-5, 3e-5, 1e-4]
X0 = float(np.mean(np.log(C.WIDTHS)))
Y0 = float(np.mean(np.log(MAIN_DECAYS)))
ROWS = {
    'main': (MAIN_DECAYS, 'grok95'),
    'with3e-4': (MAIN_DECAYS + [3e-4], 'grok95'),
    'level80': (MAIN_DECAYS, 'grok80'),
    'level90': (MAIN_DECAYS, 'grok90'),
    'mem': (MAIN_DECAYS, 'mem'),
}
NAMES = {'x': 'a', 'y': 'b', 'xy': 'gamma', 'Intercept': 'c'}


def design(df, event):
    d = pd.DataFrame({'x': np.log(df.width) - X0, 'y': np.log(df.eta_kappa) - Y0})
    d['xy'] = d.x * d.y
    d['T'] = df[f't_{event}'].to_numpy()
    d['E'] = (~df[f'censored_{event}']).astype(int).to_numpy()
    d['cell'] = list(zip(df.width, df.eta_kappa))
    return d.reset_index(drop=True)


def fit(d):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = LogNormalAFTFitter()
        m.fit(d[['x', 'y', 'xy', 'T', 'E']], duration_col='T', event_col='E')
    mu = m.params_.loc['mu_']
    out = {NAMES[k]: float(v) for k, v in mu.items()}
    out['sigma'] = float(np.exp(m.params_.loc['sigma_'].iloc[0]))
    return out, m


def bootstrap(d, rng, kind):
    cells = sorted(set(d.cell))
    by_cell = {c: d.index[d.cell == c].to_numpy() for c in cells}
    draws, skipped = [], 0
    while len(draws) < N_BOOT:
        if kind == 'cells':
            pick = [cells[i] for i in rng.integers(len(cells), size=len(cells))]
            if len({c[0] for c in pick}) < 2 or len({c[1] for c in pick}) < 2:
                skipped += 1
                continue
        else:
            pick = cells
        idx = np.concatenate([rng.choice(by_cell[c], size=len(by_cell[c]), replace=True) for c in pick])
        b = d.loc[idx].reset_index(drop=True)
        if b.E.sum() < 5:
            skipped += 1
            continue
        try:
            draws.append(fit(b)[0])
        except Exception:
            skipped += 1
    df = pd.DataFrame(draws)
    return {k: [float(df[k].quantile(0.025)), float(df[k].quantile(0.975))] for k in df.columns}, skipped


def spreads(d, m):
    """Within-cell SD of log T over uncensored cells, and SD of the cell means of the residuals."""
    pred = m.predict_median(d[['x', 'y', 'xy']]).to_numpy().ravel()
    d = d.assign(logT=np.log(d['T']), res=np.log(d['T']) - np.log(pred))
    unc = d.groupby('cell').filter(lambda g: g.E.all())
    within = unc.groupby('cell')['logT'].std(ddof=1)
    misfit = unc.groupby('cell')['res'].mean()
    return {'within_cell_sd_median': float(within.median()), 'within_cell_sd_max': float(within.max()),
            'cell_misfit_sd': float(misfit.std(ddof=1)), 'n_cells_uncensored': int(len(within))}


def main():
    runs = C.load_runs()
    a1 = runs[runs.alpha == 1.0]
    rng = np.random.default_rng(SEED)
    out = {'centre': {'x0_logN': X0, 'y0_log_decay': Y0, 'N0': float(np.exp(X0)), 'decay0': float(np.exp(Y0))},
           'rows': {}}
    for name, (decays, event) in ROWS.items():
        d = design(a1[a1.eta_kappa.isin(decays)], event)
        est, m = fit(d)
        ci_cells, skip_c = bootstrap(d, rng, 'cells')
        row = {'decays': decays, 'event': event, 'n_runs': int(len(d)), 'n_censored': int((d.E == 0).sum()),
               'estimate': est, 'ci_cells': ci_cells, 'boot_redrawn_cells': skip_c, **spreads(d, m)}
        if name == 'main':
            row['ci_seeds'], row['boot_redrawn_seeds'] = bootstrap(d, rng, 'seeds')
        out['rows'][name] = row
        ci = row['ci_cells']
        print(f"{name:9s} n={row['n_runs']} cens={row['n_censored']}  " + '  '.join(
            f"{k} {est[k]:+.3f} [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]" for k in ('a', 'b', 'gamma', 'sigma')))
        print(f"          within-cell sd {row['within_cell_sd_median']:.3f}, cell misfit sd {row['cell_misfit_sd']:.3f}")
    main = out['rows']['main']
    assert abs(main['estimate']['a'] - 0.237) < 0.005 and abs(main['estimate']['b'] + 0.633) < 0.005, main['estimate']
    print('seed-only intervals on main:', {k: [round(v, 3) for v in main['ci_seeds'][k]] for k in ('a', 'b', 'gamma')})
    C.write_json('E06', out, SEEDS)


if __name__ == '__main__':
    main()
