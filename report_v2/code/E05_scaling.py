"""E5. Scaling of the grokking time with weight decay at each width (Fig. 4, Section 4.3).

At alpha = 1 and eta lambda > 0, the figure plots the grokking step at level 0.95 against eta lambda
on log axes, one colour per width. Each seed is a point. A censored run is an open marker at the
budget of 200,000. The line joins the median over seeds of each cell in which at least 3 seeds grok.
The dashed lines are the fitted plane of E6 (main row) over the main range 1e-5 to 1e-4. The grey
lines are reference slopes of 0 (rotation clock) and -1 (weight-decay clock).
"""
import numpy as np

import common as C

SEED = 0
SEEDS = {'jitter': SEED}


def main():
    plt = C.setup_matplotlib()
    runs = C.load_runs()
    a1 = runs[(runs.alpha == 1.0) & (runs.eta_kappa > 0)]
    fit = C.read_json('E06')
    est, cen = fit['rows']['main']['estimate'], fit['centre']
    colours = C.width_colours()
    rng = np.random.default_rng(SEED)
    decays = sorted(a1.eta_kappa.unique())

    fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
    for w in C.WIDTHS:
        g = a1[a1.width == w]
        jit = np.exp(rng.uniform(-0.06, 0.06, len(g)))
        ok = ~g.censored_grok95
        ax.scatter(g.eta_kappa[ok] * jit[ok], g.t_grok95[ok], s=9, color=colours[w], lw=0, alpha=0.8, zorder=3)
        ax.scatter(g.eta_kappa[~ok] * jit[~ok], np.full((~ok).sum(), C.BUDGET), s=11, facecolor='white',
                   edgecolor=colours[w], lw=0.8, zorder=3)
        med = g.groupby('eta_kappa').apply(lambda c: np.median(c.t_grok95[~c.censored_grok95])
                                           if (~c.censored_grok95).sum() >= 3 else np.nan, include_groups=False)
        ax.plot(med.index, med.values, color=colours[w], lw=1.2, label=f'$N={w}$', zorder=2)
        x = np.log(w) - cen['x0_logN']
        d = np.geomspace(1e-5, 1e-4, 20)
        y = np.log(d) - cen['y0_log_decay']
        ax.plot(d, np.exp(est['c'] + est['a'] * x + est['b'] * y + est['gamma'] * x * y), color=colours[w], ls='--',
                lw=0.9, zorder=4)
    # A key of the two reference slopes, drawn from one point in an empty corner.
    x0, t0, x1 = 5e-4, 1.6e4, 2.2e-3
    ax.plot([x0, x1], [t0, t0], color=C.MUTED, lw=0.9)
    ax.plot([x0, x1], [t0, t0 * (x1 / x0) ** -1], color=C.MUTED, lw=0.9)
    ax.text(x1 * 1.08, t0, 'slope 0', fontsize=6.5, color=C.MUTED, va='center')
    ax.text(x1 * 1.08, t0 * (x1 / x0) ** -1, 'slope $-1$', fontsize=6.5, color=C.MUTED, va='center')
    ax.axhline(C.BUDGET, color=C.MUTED, lw=0.6)
    ax.text(3.2e-6, C.BUDGET * 1.08, 'budget (open: censored)', fontsize=6.5, color=C.MUTED, va='bottom')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(decays, [C.decay_tick(d) for d in decays])
    ax.set_ylim(3e3, 3.2e5)
    ax.set_xlim(2.2e-6, 6e-3)
    ax.set_xlabel(r'weight decay $\eta\lambda$')
    ax.set_ylabel(r'grokking step $T$ (level 0.95)')
    ax.legend(loc='lower left', ncol=2, handlelength=1.5, fontsize=6.5)
    C.save_figure(fig, 'E5_scaling')
    C.write_json('E05', {'n_runs': len(a1), 'n_censored': int(a1.censored_grok95.sum())}, SEEDS)


if __name__ == '__main__':
    main()
