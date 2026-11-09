"""E9. The S x log R surface (Fig. 7, Section 4.4), with the interaction coefficient gamma_SR.

gamma_SR is the coefficient of the product of the z-scores of S_t and log R_t in the log-normal AFT
with all three statistics and their pairwise products, fitted with censoring on the 630 runs with
eta lambda at most 1e-3 (penalizer 1e-4). The z-scores use the mean and spread of all 630 runs. Its
interval comes from 200 bootstrap draws of whole cells (model_fitting/3_kernel_only.ipynb, Part 2).
A negative gamma_SR means that a larger scale shortens the grokking time more when the kernel has
changed shape more.

Figure: the kernel ridge prediction of log10 T over S_t and R_t at t*, with A_t at its median, fitted
on the 355 runs that grok (model_fitting/4_nonlinear_exploration.ipynb, cells 28 and 33). The runs
are drawn as points. Grid points farther than 0.25 standard deviations (in S and log R) from every
run are shaded, since the surface there is an extrapolation.
"""
import numpy as np
import pandas as pd

import common as C
import models as M

N_BOOT = 200
SEED = 0
SEEDS = {'cell_bootstrap': SEED, 'n_boot': N_BOOT}
EMPTY_RADIUS = 0.25
TERM = 'S_t*logR_t'


def main():
    data = M.load_data()
    est = M.fit_aft(data, M.design(data, True, ref=data), M.AFT_PENALIZER_FULL).params_.loc['mu_']
    rng = np.random.default_rng(SEED)
    cells = data.cell.unique()
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(cells, size=len(cells), replace=True)
        s = data.iloc[np.concatenate([np.flatnonzero(data.cell == c) for c in pick])].reset_index(drop=True)
        boot.append(M.fit_aft(s, M.design(s, True, ref=data), M.AFT_PENALIZER_FULL).params_.loc['mu_'])
    boot = pd.DataFrame(boot)
    terms = [c for c in boot.columns if c != 'Intercept']
    coef = {t: {'estimate': float(est[t]), 'ci': [float(boot[t].quantile(0.025)), float(boot[t].quantile(0.975))]}
            for t in terms}

    plt = C.setup_matplotlib()
    import matplotlib
    grok = data[data.grokked == 1].reset_index(drop=True)
    krr = M.kernel_ridge().fit(grok[M.COVS], np.log(grok['T']), groups=grok['cell'])
    s_grid = np.linspace(grok.S_t.min(), grok.S_t.max(), 120)
    r_grid = np.linspace(grok.logR_t.min(), grok.logR_t.max(), 120)
    S, LR = np.meshgrid(s_grid, r_grid)
    probe = pd.DataFrame({'A_t': grok.A_t.median(), 'S_t': S.ravel(), 'logR_t': LR.ravel()})
    surface = krr.predict(probe).reshape(S.shape) / np.log(10)
    zs, zr = (S - grok.S_t.mean()) / grok.S_t.std(), (LR - grok.logR_t.mean()) / grok.logR_t.std()
    ps = ((grok.S_t - grok.S_t.mean()) / grok.S_t.std()).to_numpy()
    pr = ((grok.logR_t - grok.logR_t.mean()) / grok.logR_t.std()).to_numpy()
    dist = np.sqrt((zs.ravel()[:, None] - ps[None]) ** 2 + (zr.ravel()[:, None] - pr[None]) ** 2).min(axis=1)
    empty = (dist > EMPTY_RADIUS).reshape(S.shape)

    fig, ax = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
    im = ax.contourf(S, np.exp(LR), surface, levels=12, cmap=C.VIOLET_CMAP)
    ax.contourf(S, np.exp(LR), empty.astype(float), levels=[0.5, 1.5], colors=['#f0efec'], alpha=0.85)
    ax.scatter(grok.S_t, grok.R_t, s=5, color=C.TEXT, lw=0, alpha=0.8, label='runs that grok')
    ax.set_yscale('log')
    ax.set_xlabel(r'scale $S_t$ at $t^\ast$')
    ax.set_ylabel(r'shape $R_t$ at $t^\ast$')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'predicted $\log_{10} T$')
    cb.outline.set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[ax.collections[-1], Patch(color='#f0efec', label='no runs nearby')], loc='lower right',
              fontsize=6.5, frameon=True, framealpha=0.9, edgecolor='none')
    C.save_figure(fig, 'E9_surface')
    out = {'coefficients': coef, 'gamma_SR': coef[TERM], 'n_runs': len(data),
           'surface_runs': len(grok), 'krr_params': krr.regressor_.best_params_}
    C.write_json('E09', out, SEEDS)
    print('gamma_SR', coef[TERM])
    print({t: round(v['estimate'], 3) for t, v in coef.items()})
    print('krr', krr.regressor_.best_params_)


if __name__ == '__main__':
    main()
