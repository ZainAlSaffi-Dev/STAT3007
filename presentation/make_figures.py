"""Slide versions of the report figures E2, E4, E5, E7, E8 and E9.

The report figures come from the scripts in report_v2/code. They are drawn for an A4 page, so their
type is too small on a projector. This script draws each one again for its place in slides.qmd.
Each figure is 16 by 9, its type matches the deck clips, and it keeps to the message of its slide.

It reads the committed extract and the saved numbers that the report scripts read, in
report_v2/code/data and report_v2/code/out/numbers, and it takes its colours from
report_v2/code/common.py. A change to the report numbers therefore reaches the slides when this
script runs again. It trains nothing. The only fit is the kernel ridge surface of E9, and the script
checks that the fit picks the same settings as the report.

Run it from the repository root with the root environment:

    .venv/bin/python presentation/make_figures.py

It writes presentation/assets/figures/E<k>_<name>.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'report_v2' / 'code'))
import common as C  # noqa: E402

OUT = HERE / 'assets' / 'figures'

# A figure is drawn at 100 pixels per inch of the 1920 by 1080 slide, so type of s points shows at
# 1.39 s pixels. A figure in a 55% column is 1056 pixels wide. E2 fills the slide below its title.
# The PNGs are saved at twice that resolution.
COLUMN = (10.56, 5.94)
WIDE = (14.58, 8.2)

NEUTRAL = '#8a8986'   # a bar that stands for no single statistic, as in the report figure E8
NO_GROK = '#f0efec'   # a cell or region with no data, as in the report figures E4 and E9

# Tick labels of a log axis in the TeX font, as the other formulas. Matplotlib's own labels set the
# digits in the text font.
POW10 = FuncFormatter(lambda v, _: f'$10^{{{round(np.log10(v))}}}$')

plt = C.setup_matplotlib()
plt.rcParams.update({
    # The deck sets Source Sans 3 with Helvetica Neue as its fallback. Matplotlib finds only the
    # fallback here, as Manim does for the clips. Formulas use the TeX fonts, as MathJax does.
    'font.family': 'Helvetica Neue', 'mathtext.fontset': 'cm', 'axes.unicode_minus': False,
    'font.size': 20, 'axes.titlesize': 22, 'axes.labelsize': 22, 'xtick.labelsize': 19, 'ytick.labelsize': 19,
    'legend.fontsize': 19, 'lines.linewidth': 2.4, 'lines.markersize': 9, 'axes.linewidth': 1.2,
    'grid.linewidth': 1.0, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2, 'xtick.major.size': 6,
    'ytick.major.size': 6, 'xtick.minor.size': 3.5, 'ytick.minor.size': 3.5,
    'figure.dpi': 100, 'savefig.dpi': 200, 'savefig.bbox': 'standard',
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', metadata={'Software': None})
    plt.close(fig)


def light_cmap(name, start):
    """A colour map without its lightest part, so that every cell stands out from the white gaps."""
    import matplotlib
    from matplotlib.colors import ListedColormap
    return ListedColormap(matplotlib.colormaps[name](np.linspace(start, 1, 256)))


def grid_ticks(ax):
    ax.set_xticks(np.arange(len(C.DECAYS)) + 0.5, [C.decay_tick(d) if d else '$0$' for d in C.DECAYS])
    ax.set_yticks(np.arange(len(C.WIDTHS)) + 0.5, [str(w) for w in C.WIDTHS])
    ax.set_ylabel('width $N$')
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def e2():
    """Slide 20: the statistics at the grokking step of every run that groks, after E02_thresholds.py.

    Each dot is a run and each bar a median. The ratio on each panel is the tightening ratio of the
    placebo test from E03. The report shades the dots by weight decay. The slide uses one hue per
    statistic, since its message is the spread.
    """
    runs = C.load_runs()
    grok = runs[~runs.censored_grok95]
    e3 = C.read_json('E03')['groups']
    rng = np.random.default_rng(0)
    rows = [('N100', grok[grok.width == 100], 'alpha', C.ALPHAS, r'width 100, by $\alpha$'),
            ('alpha1', grok[grok.alpha == 1.0], 'width', C.WIDTHS, r'$\alpha = 1$, by width')]
    fig, axes = plt.subplots(2, 3, figsize=WIDE, layout='constrained')
    for r, (group, g, by, levels, label) in enumerate(rows):
        for c, stat in enumerate(C.STATS):
            ax = axes[r, c]
            for i, level in enumerate(levels):
                v = g.loc[g[by] == level, f'{stat}_t_at_grok95']
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=46, color=C.STAT_COLOUR[stat], lw=0.6,
                           edgecolors='white', zorder=3)
                ax.hlines(v.median(), i - 0.34, i + 0.34, color=C.TEXT, lw=3, zorder=4)
            ax.set_xticks(range(len(levels)), [f'{v:g}' for v in levels])
            ax.set_xlim(-0.6, len(levels) - 0.4)
            ax.grid(axis='x', visible=False)
            ax.set_title(f"ratio {e3[group][stat]['sd']['tightening']:.2f}", loc='right', color=C.MUTED,
                         fontsize=19)
            if r == 0:
                ax.set_title(C.STAT_NAME[stat], loc='left', color=C.STAT_RAMP[stat][4], fontweight='bold')
        axes[r, 0].set_ylabel(label)
    save(fig, 'E2_thresholds')


def e4():
    """Slide 24: grokking time and outcome over width and weight decay at alpha = 1, after E04_map.py.

    The cells come from E04.json. At alpha = 1 every cell has 0 or 5 seeds that grok, so the slide
    drops the counts and the phase letters of the report figure. A cell with 1 to 4 would get its count.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch
    cells = pd.DataFrame(C.read_json('E04')['alpha1'])
    grid = cells.pivot(index='width', columns='eta_kappa', values='median_T').loc[C.WIDTHS, C.DECAYS].to_numpy(float)
    cmap = light_cmap(C.VIOLET_CMAP, 0.3)
    cmap.set_bad(NO_GROK)
    fig, ax = plt.subplots(figsize=COLUMN, layout='constrained')
    im = ax.pcolormesh(np.ma.masked_invalid(grid), cmap=cmap, norm=LogNorm(np.nanmin(grid), np.nanmax(grid)),
                       edgecolors='white', linewidth=2)
    for row in cells.itertuples():
        if 0 < row.grokked < row.seeds:
            ax.text(C.DECAYS.index(row.eta_kappa) + 0.5, C.WIDTHS.index(row.width) + 0.5,
                    f'{row.grokked}/{row.seeds}', ha='center', va='center')
    grid_ticks(ax)
    ax.set_xlabel(r'weight decay $\eta\lambda$')
    cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=16)
    cb.set_ticks([1e4, 3e4, 1e5], labels=['$10^4$', '$3{\\cdot}10^4$', '$10^5$'])
    cb.ax.minorticks_off()
    cb.set_label('median grokking step $T$')
    cb.outline.set_visible(False)
    ax.legend(handles=[Patch(facecolor=NO_GROK, label='no seed groks')], loc='lower left',
              bbox_to_anchor=(0, 1), borderaxespad=0.3, handlelength=1.2)
    save(fig, 'E4_map')


def e5():
    """Slide 22: the grokking time against weight decay at each width, alpha = 1, after E05_scaling.py.

    Dots are seeds, open circles at the budget are runs that did not grok, solid lines join the cell
    medians where at least 3 seeds grok, and dashed lines are the main fit of E06 over its range.
    """
    runs = C.load_runs()
    a1 = runs[(runs.alpha == 1.0) & (runs.eta_kappa > 0)]
    fit = C.read_json('E06')
    est, cen = fit['rows']['main']['estimate'], fit['centre']
    colours = C.width_colours()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=COLUMN, layout='constrained')
    for w in C.WIDTHS:
        g = a1[a1.width == w]
        jit = np.exp(rng.uniform(-0.06, 0.06, len(g)))
        ok = ~g.censored_grok95
        ax.scatter(g.eta_kappa[ok] * jit[ok], g.t_grok95[ok], s=24, color=colours[w], alpha=0.45, lw=0, zorder=3)
        ax.scatter(g.eta_kappa[~ok] * jit[~ok], np.full((~ok).sum(), C.BUDGET), s=34, facecolor='white',
                   edgecolor=colours[w], lw=1.4, zorder=3)
        hit = g[ok]
        med = hit.groupby('eta_kappa').t_grok95.median().where(hit.groupby('eta_kappa').size() >= 3)
        ax.plot(med.index, med.values, color=colours[w], marker='o', ms=7, label=f'$N = {w}$', zorder=4)
        d = np.geomspace(1e-5, 1e-4, 20)
        x, y = np.log(w) - cen['x0_logN'], np.log(d) - cen['y0_log_decay']
        ax.plot(d, np.exp(est['c'] + est['a'] * x + est['b'] * y + est['gamma'] * x * y), color=colours[w], ls='--',
                lw=1.8, zorder=5)
    # The two reference slopes, drawn from one point in the empty lower right corner.
    x0, t0, x1 = 6e-4, 1.4e4, 2.4e-3
    ax.plot([x0, x1], [t0, t0], color=C.MUTED, lw=2)
    ax.plot([x0, x1], [t0, t0 * x0 / x1], color=C.MUTED, lw=2)
    ax.text(x1 * 1.12, t0, 'slope 0', color=C.MUTED, va='center')
    ax.text(x1 * 1.12, t0 * x0 / x1, 'slope $-1$', color=C.MUTED, va='center')
    ax.axhline(C.BUDGET, color=C.MUTED, lw=1.2)
    ax.text(2.6e-6, C.BUDGET * 1.07, 'Open circles did not grok within 200,000 steps.', color=C.MUTED, va='bottom',
            fontsize=17)
    ax.set_xscale('log')
    ax.set_yscale('log')
    decays = sorted(a1.eta_kappa.unique())
    ax.set_xticks(decays, [C.decay_tick(d) for d in decays])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(POW10)
    ax.set_ylim(2e3, 3.4e5)
    ax.set_xlim(2.2e-6, 9e-3)
    ax.set_xlabel(r'weight decay $\eta\lambda$')
    ax.set_ylabel('grokking step $T$')
    ax.legend(loc='lower left', ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0.2, fontsize=18)
    save(fig, 'E5_scaling')


def e7():
    """Slide 25: median rotation and scale at t* over width and weight decay, alpha = 1, after E07_mechanism.py.

    The medians come from E07.json. The slide drops the number in each cell, since its message is
    the pattern: rotation changes along the widths and scale along the weight decays.
    """
    from matplotlib.colors import LogNorm, Normalize
    med = C.read_json('E07')['alpha1_medians']
    fig, axes = plt.subplots(2, 1, figsize=COLUMN, sharex=True, layout='constrained')
    for ax, stat in zip(axes, ('R', 'S')):
        grid = np.array([[med[f'{w}_{d:g}'][stat] for d in C.DECAYS] for w in C.WIDTHS])
        norm = LogNorm(grid.min(), grid.max()) if stat == 'R' else Normalize(0, grid.max())
        im = ax.pcolormesh(grid, cmap=light_cmap(C.STAT_CMAP[stat], 0.12), norm=norm, edgecolors='white',
                           linewidth=2)
        grid_ticks(ax)
        ax.set_title(C.STAT_NAME[stat], loc='left', color=C.STAT_RAMP[stat][4], fontweight='bold')
        cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=8)
        cb.ax.minorticks_off()
        cb.outline.set_visible(False)
        if stat == 'R':
            cb.ax.yaxis.set_major_formatter(POW10)
    axes[0].set_title(r'median over seeds at $t^{*}$', loc='right', color=C.MUTED, fontsize=19)
    axes[1].set_xlabel(r'weight decay $\eta\lambda$')
    save(fig, 'E7_mechanism')


def e8():
    """Slide 28: the prediction ladder on a held-out alpha, after E08_ladder.py.

    Bars are the mean concordance over the three held-out values of alpha, from E08.json. The
    dashed line is the reference given the settings. The slide drops the dots of each held-out value
    and the seed ranges, and names the models in plain words as the backup slide does.
    """
    ladder = C.read_json('E08')['ladder']
    models = [('A only', '$A_t$ alone', 'A'), ('S only', '$S_t$ alone', 'S'), ('log R only', r'$\log R_t$ alone', 'R'),
              ('linear AFT', 'linear', None), ('AFT with interactions', 'linear with interactions', None),
              ('kernel ridge', 'kernel ridge', None), ('XGBoost', 'boosted trees', None),
              ('GRU on history', 'recurrent network', None)]
    fig, ax = plt.subplots(figsize=COLUMN, layout='constrained')
    for i, (key, _, stat) in enumerate(models):
        mean = ladder[key]['mean']
        ax.barh(i, mean - 0.5, left=0.5, height=0.62, color=C.STAT_COLOUR[stat] if stat else NEUTRAL, zorder=2)
        ax.text(0.875, i, f'{mean:.2f}', ha='right', va='center')
    ref = ladder['settings reference']['mean']
    # The two lines stop below their labels.
    ax.vlines(ref, -0.55, len(models) - 0.45, color=C.TEXT, ls='--', lw=2, zorder=3)
    ax.text(ref, -0.7, f'given the settings, {ref:.2f}', ha='center', va='bottom')
    ax.vlines(0.5, -0.55, len(models) - 0.45, color=C.MUTED, lw=1.4, ls=':', zorder=1)
    ax.text(0.5, -0.7, 'chance', ha='center', va='bottom', color=C.MUTED)
    ax.set_yticks(range(len(models)), [label for _, label, _ in models])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0.45, 0.88)
    ax.set_ylim(len(models) - 0.45, -1.35)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel(r'concordance on an unseen $\alpha$')
    ax.grid(axis='y', visible=False)
    ax.spines['left'].set_visible(False)
    save(fig, 'E8_ladder')


def e9():
    """Slide 27: the kernel ridge surface of the grokking step over S and R at t*, after E09_surface.py.

    The fit is that of the report figure, on the runs that grok, with A at its median. Grid points
    farther than 0.25 standard deviations from every run are shaded, since the surface there is an
    extrapolation.
    """
    import models as M
    from matplotlib.patches import Patch
    data = M.load_data()
    grok = data[data.grokked == 1].reset_index(drop=True)
    krr = M.kernel_ridge().fit(grok[M.COVS], np.log(grok['T']), groups=grok['cell'])
    assert krr.regressor_.best_params_ == C.read_json('E09')['krr_params'], 'the fit differs from the report'
    S, LR = np.meshgrid(np.linspace(grok.S_t.min(), grok.S_t.max(), 120),
                        np.linspace(grok.logR_t.min(), grok.logR_t.max(), 120))
    probe = pd.DataFrame({'A_t': grok.A_t.median(), 'S_t': S.ravel(), 'logR_t': LR.ravel()})
    surface = krr.predict(probe).reshape(S.shape) / np.log(10)
    z = lambda v, col: (v - grok[col].mean()) / grok[col].std()
    dist = np.sqrt((z(S.ravel(), 'S_t')[:, None] - z(grok.S_t, 'S_t').to_numpy()) ** 2
                   + (z(LR.ravel(), 'logR_t')[:, None] - z(grok.logR_t, 'logR_t').to_numpy()) ** 2).min(axis=1)
    empty = (dist > 0.25).reshape(S.shape)

    fig, ax = plt.subplots(figsize=COLUMN, layout='constrained')
    im = ax.contourf(S, np.exp(LR), surface, levels=12, cmap=C.VIOLET_CMAP)
    ax.contourf(S, np.exp(LR), empty.astype(float), levels=[0.5, 1.5], colors=[NO_GROK], alpha=0.85)
    dots = ax.scatter(grok.S_t, grok.R_t, s=20, color=C.TEXT, lw=0, alpha=0.8, label='runs that grok')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(POW10)
    ax.set_xlabel(r'scale $S_t$ at $t^{*}$')
    ax.set_ylabel(r'rotation $R_t$ at $t^{*}$')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=16)
    cb.set_ticks(np.log10([1e4, 3e4, 1e5, 3e5]), labels=['$10^4$', '$3{\\cdot}10^4$', '$10^5$', '$3{\\cdot}10^5$'])
    cb.set_label('predicted grokking step $T$')
    cb.outline.set_visible(False)
    ax.legend(handles=[dots, Patch(color=NO_GROK, label='no runs nearby')], loc='lower right', frameon=True,
              framealpha=0.9, edgecolor='none', markerscale=1.6)
    save(fig, 'E9_surface')


if __name__ == '__main__':
    for draw in (e2, e4, e5, e7, e8, e9):
        draw()
        print('drew', draw.__name__)
