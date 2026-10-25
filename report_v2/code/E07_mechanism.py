"""E7. Width sets the shape change of the kernel and weight decay sets its scale (Fig. 5, Section 4.3),
with number N9.

Figure: two maps over N and eta lambda at alpha = 1, read at t* = 2,430, a saved step before any run
groks: the median log10 R_t and the median S_t over the 5 seeds of each cell.

N9: the share of the spread of log R_t and of S_t at t* that each setting explains on its own,
1 - SS_within / SS_total with the runs grouped by alpha, by N or by eta lambda. It is computed on
the runs that grok at 0.95 (355, as in model_fitting/4_nonlinear_exploration.ipynb Part 2), and
again on all 630 runs with eta lambda at most 1e-3 as a check.
"""
import numpy as np

import common as C

SEEDS = {}


def shares(df):
    return {col: {by: C.between_share(df, col, by) for by in ('alpha', 'width', 'eta_kappa')}
            for col in ('logR_t', 'S_t', 'A_t')}


def main():
    plt = C.setup_matplotlib()
    import matplotlib
    runs = C.at_tstar(C.load_runs())
    a1 = runs[runs.alpha == 1.0]
    med = a1.groupby(['width', 'eta_kappa'])[['R_t', 'S_t']].median()

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3), constrained_layout=True)
    for ax, (col, stat, transform, label) in zip(axes, [
            ('R_t', 'R', np.log10, r'median $\log_{10} R_t$ at $t^\ast$'),
            ('S_t', 'S', lambda v: v, r'median $S_t$ at $t^\ast$')]):
        grid = np.array([[transform(med.loc[(w, d), col]) for d in C.DECAYS] for w in C.WIDTHS])
        im = ax.imshow(grid, cmap=C.STAT_CMAP[stat], aspect='auto', origin='lower')
        lo, hi = np.nanmin(grid), np.nanmax(grid)
        for i in range(len(C.WIDTHS)):
            for j in range(len(C.DECAYS)):
                v = grid[i, j]
                txt = f'{v:.1f}' if stat == 'R' else f'{v:.2f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=6,
                        color='white' if (v - lo) / (hi - lo) > 0.6 else C.TEXT)
        ax.set_xticks(range(len(C.DECAYS)), [C.decay_tick(d) for d in C.DECAYS], fontsize=6)
        ax.set_yticks(range(len(C.WIDTHS)), [str(w) for w in C.WIDTHS])
        ax.set_xlabel(r'weight decay $\eta\lambda$')
        ax.set_ylabel('width $N$')
        ax.set_title(f'({"ab"[stat == "S"]}) {C.STAT_NAME[stat]}', loc='left')
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cb.set_label(label, fontsize=7)
        cb.outline.set_visible(False)
    C.save_figure(fig, 'E7_mechanism')

    grok = runs[(~runs.censored_grok95) & (runs.eta_kappa <= 1e-3)]
    ref = runs[runs.eta_kappa <= 1e-3]
    out = {'N9_grokked': shares(grok), 'N9_all': shares(ref), 'n_grokked': len(grok), 'n_all': len(ref),
           'alpha1_medians': {f'{w}_{d:g}': {'R': float(med.loc[(w, d), 'R_t']), 'S': float(med.loc[(w, d), 'S_t'])}
                              for w in C.WIDTHS for d in C.DECAYS}}
    C.write_json('E07', out, SEEDS)
    for k in ('N9_grokked', 'N9_all'):
        print(k, {c: {b: round(v, 3) for b, v in d.items()} for c, d in out[k].items()})


if __name__ == '__main__':
    main()
