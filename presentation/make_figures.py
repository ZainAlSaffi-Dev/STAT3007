"""Slide versions of the report figures E2, E4, E5, E7, E8 and E19, and the placebo boxes E3.

The report figures come from the scripts in report_v2/code. They are drawn for an A4 page, so their
type is too small on a projector. This script draws each one again for its place in slides.qmd.
Each figure is 16 by 9, its type matches the deck clips, and it keeps to the message of its slide.

It reads the committed extract and the saved numbers that the report scripts read, in
report_v2/code/data and report_v2/code/out/numbers, and it takes its colours from
report_v2/code/common.py. A change to the report numbers therefore reaches the slides when this
script runs again. It trains nothing and fits nothing.

Run it from the repository root with Python 3.11 or later and the packages of pyproject.toml:

    python presentation/make_figures.py

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
# 1.39 s pixels. Tick labels of 21 points show at 29 pixels, a little above the deck's small print. A figure in a 55% column is 1056 pixels wide. E2 fills the slide below its title.
# The PNGs are saved at twice that resolution.
COLUMN = (10.56, 5.94)
WIDE = (14.58, 8.2)

NEUTRAL = '#8a8986'   # a bar that stands for no single statistic, as in the report figure E8
BAND = '#f0efec'      # the shaded fit range of E5

# The deck calls R_t "shape". The names are set here, so the slides do not depend on the report's labels.
STAT_NAME = {'S': 'scale $S_t$', 'R': 'shape $R_t$', 'A': 'alignment $A_t$'}
NO_GROK = '#f0efec'   # a cell with no data, as in the report figure E4

# Tick labels of a log axis in the TeX font, as the other formulas. Matplotlib's own labels set the
# digits in the text font.
POW10 = FuncFormatter(lambda v, _: f'$10^{{{round(np.log10(v))}}}$')

plt = C.setup_matplotlib()
plt.rcParams.update({
    # The deck sets Source Sans 3 with Helvetica Neue as its fallback. Matplotlib finds only the
    # fallback here, as Manim does for the clips. Nimbus Sans, a free copy of Helvetica, stands in
    # where Helvetica Neue is not installed. Formulas use the TeX fonts, as MathJax does.
    'font.family': ['Helvetica Neue', 'Nimbus Sans'], 'mathtext.fontset': 'cm', 'axes.unicode_minus': False,
    'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 24, 'xtick.labelsize': 21, 'ytick.labelsize': 21,
    'legend.fontsize': 21, 'lines.linewidth': 2.4, 'lines.markersize': 9, 'axes.linewidth': 1.2,
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
    """Slide 17: the statistics at the grokking step of every run that groks, after E02_thresholds.py.

    Each dot is a run and each bar a median. The ratio on each panel is the SD ratio of the placebo
    test from E03, which the report called the tightening ratio. The report shades the dots by weight
    decay. The slide uses one hue per statistic, since its message is the spread. The figure sits in
    a column of 62%, so it is drawn smaller than WIDE to keep its type large.
    """
    runs = C.load_runs()
    grok = runs[~runs.censored_grok95]
    e3 = C.read_json('E03')['groups']
    rng = np.random.default_rng(0)
    rows = [('N100', grok[grok.width == 100], 'alpha', C.ALPHAS, r'width 100, by $\alpha$'),
            ('alpha1', grok[grok.alpha == 1.0], 'width', C.WIDTHS, r'$\alpha = 1$, by width')]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6), layout='constrained')
    for r, (group, g, by, levels, label) in enumerate(rows):
        for c, stat in enumerate(C.STATS):
            ax = axes[r, c]
            for i, level in enumerate(levels):
                v = g.loc[g[by] == level, f'{stat}_t_at_grok95']
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=46, color=C.STAT_COLOUR[stat], lw=0.6,
                           edgecolors='white', zorder=3)
                ax.hlines(v.median(), i - 0.34, i + 0.34, color=C.TEXT, lw=3, zorder=4)
            ax.set_xticks(range(len(levels)), [f'{v:g}' for v in levels], fontsize=21 if r == 0 else 18)
            ax.set_xlim(-0.6, len(levels) - 0.4)
            ax.grid(axis='x', visible=False)
            # The ratio is a text above the panel, since the left and right titles of an axes share one pad.
            ax.text(1.0, 1.02, f"SD ratio {e3[group][stat]['sd']['tightening']:.2f}", transform=ax.transAxes,
                    ha='right', va='bottom', color=C.MUTED, fontsize=19)
            if r == 0:
                ax.set_title(STAT_NAME[stat], loc='left', color=C.STAT_RAMP[stat][4], fontweight='bold', pad=40)
            else:
                ax.set_title(' ', pad=18)
        axes[r, 0].set_ylabel(label)
    save(fig, 'E2_thresholds')


def p_label(p, n_perm):
    """The permutation p as the slides print it: below 1 / n_perm when no permutation reached it."""
    return f'$p < {1 / n_perm:g}$' if p == 0 else f'$p = {p:.3f}$'


def e3():
    """Slide 16: the placebo test at width 100 as box plots, after E03_placebo.py.

    For each statistic, the left box holds its values at each run's own grokking step, and the right
    box its values when each run is read at the grokking step of another run. The right box pools the
    2,000 derangements of E03, drawn with the same seed in the same order, so the SD ratio printed on
    each panel is recomputed here and checked against E03.json. Whiskers reach the furthest value
    within 1.5 box heights, and values beyond them are not drawn.
    """
    import importlib
    e03 = importlib.import_module('E03_placebo')
    runs = C.load_runs()
    grok = runs[~runs.censored_grok95]
    group = grok[grok.width == 100]
    ckpt = C.load_checkpoints(['run_id', 'step', 'S_sum', 'R_sum', 'A_sum'])
    saved = C.read_json('E03')['groups']['N100']
    # E03 draws the derangements of the width-100 group first, from a fresh generator.
    rng = np.random.default_rng(e03.SEED)
    ids = group.run_id.to_numpy()
    t_grok = group.t_grok95.to_numpy()
    perms = e03.derangements(len(ids), e03.N_PERM, rng)

    fig, axes = plt.subplots(1, 3, figsize=(11.9, 6.7), layout='constrained')
    for ax, stat in zip(axes, C.STATS):
        wide = C.pivot(ckpt, C.CHECKPOINT_STATS[stat])
        own = C.value_at_step(wide, ids, t_grok)
        sub = wide.loc[ids]
        col = np.searchsorted(sub.columns.to_numpy(), t_grok[perms], side='right') - 1
        other = sub.to_numpy()[np.arange(len(ids))[None, :], col]
        ratio = np.median(C.spread(other)) / C.spread(own)
        assert np.isclose(ratio, saved[stat]['sd']['tightening'], rtol=1e-9), stat
        ramp = C.STAT_RAMP[stat]
        box = ax.boxplot([own, other.ravel()], positions=[0, 1], widths=0.55, whis=1.5, showfliers=False,
                         patch_artist=True, medianprops={'color': C.TEXT, 'lw': 3},
                         whiskerprops={'color': C.MUTED, 'lw': 1.6}, capprops={'color': C.MUTED, 'lw': 1.6})
        for patch, face in zip(box['boxes'], (C.STAT_COLOUR[stat], ramp[0])):
            patch.set(facecolor=face, edgecolor=C.STAT_COLOUR[stat], lw=1.6)
        ax.set_xticks([0, 1], ['own\nstep', "another\nrun's step"])
        ax.set_xlim(-0.55, 1.55)
        ax.grid(axis='x', visible=False)
        ax.tick_params(axis='x', length=0)
        ax.set_title(STAT_NAME[stat], loc='left', color=ramp[4], fontweight='bold', pad=40)
        ax.text(0, 1.025, f"SD ratio {ratio:.2f}, {p_label(saved[stat]['sd']['p'], e03.N_PERM)}",
                transform=ax.transAxes, color=C.MUTED, fontsize=19, va='bottom')
    save(fig, 'E3_placebo')


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
    """Slide 18: the grokking time against weight decay at each width, alpha = 1, after E05_scaling.py.

    Dots are seeds, open circles at the budget are runs that did not grok, and solid lines join the
    cell medians where at least 3 seeds grok. The grey band is the range of the main fit of E06, where
    every seed groks. The fitted lines are left out, since the slide gives the fit in a table. Two
    grey lines show the slope that each clock predicts.
    """
    runs = C.load_runs()
    a1 = runs[(runs.alpha == 1.0) & (runs.eta_kappa > 0)]
    colours = C.width_colours()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=COLUMN, layout='constrained')
    ax.axvspan(10 ** -5.15, 10 ** -3.85, color=BAND, lw=0, zorder=0)
    ax.text(10 ** -4.5, 1.55e5, 'fit range', color=C.MUTED, ha='center', va='center', fontsize=19)
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
    # The slope each clock predicts, drawn from one point in the empty lower right corner. Each label
    # names the clock, so the lines say what they are for.
    x0, t0, x1 = 4.5e-4, 1.6e4, 1.8e-3
    ax.plot([x0, x1], [t0, t0], color=C.MUTED, lw=2)
    ax.plot([x0, x1], [t0, t0 * x0 / x1], color=C.MUTED, lw=2)
    ax.text(x1 * 1.12, t0, 'shape clock\nslope 0', color=C.MUTED, va='center', fontsize=18, linespacing=1.1)
    ax.text(x1 * 1.12, t0 * x0 / x1, 'weight-decay\nclock\nslope $-1$', color=C.MUTED, va='center', fontsize=18,
            linespacing=1.1)
    ax.axhline(C.BUDGET, color=C.MUTED, lw=1.2)
    ax.text(2.6e-6, C.BUDGET * 1.07, 'Open circles did not grok within 200,000 steps.', color=C.MUTED, va='bottom',
            fontsize=19)
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
    ax.legend(loc='lower left', ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0.2, fontsize=20)
    save(fig, 'E5_scaling')


def e7():
    """Backup: median shape and scale at t* over width and weight decay, alpha = 1, after E07_mechanism.py.

    The medians come from E07.json. The slide drops the number in each cell, since its message is
    the pattern: the shape statistic changes along the widths and scale along the weight decays.
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
        ax.set_title(STAT_NAME[stat], loc='left', color=C.STAT_RAMP[stat][4], fontweight='bold')
        cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=8)
        cb.ax.minorticks_off()
        cb.outline.set_visible(False)
        if stat == 'R':
            cb.ax.yaxis.set_major_formatter(POW10)
    axes[0].set_title(r'median over seeds at $t^{*}$', loc='right', color=C.MUTED, fontsize=21)
    axes[1].set_xlabel(r'weight decay $\eta\lambda$')
    save(fig, 'E7_mechanism')


def e8():
    """Slide 27: the prediction ladder on a held-out alpha, after E08_ladder.py.

    Bars are the mean concordance over the three held-out values of alpha, from E08.json. The
    dashed line is the reference given the settings. As in the report figure since 17 October 2026,
    only each statistic alone, the trees on all three and the recurrent network on their history are
    drawn. The slide drops the dots of each held-out value and the seed ranges.
    """
    ladder = C.read_json('E08')['ladder']
    models = [('A only', '$A_t$ alone', 'A'), ('S only', '$S_t$ alone', 'S'), ('log R only', r'$\log R_t$ alone', 'R'),
              ('XGBoost', 'all three (trees)', None), ('GRU on history', 'history (recurrent)', None)]
    fig, ax = plt.subplots(figsize=COLUMN, layout='constrained')
    for i, (name, _, stat) in enumerate(models):
        mean = ladder[name]['mean']
        ax.barh(i, mean - 0.5, left=0.5, height=0.62, color=C.STAT_COLOUR[stat] if stat else NEUTRAL, zorder=2)
        ax.text(0.875, i, f'{mean:.2f}', ha='right', va='center')
    ref = ladder['settings reference']['mean']
    # The two lines stop below their labels.
    ax.vlines(ref, -0.55, len(models) - 0.45, color=C.TEXT, ls='--', lw=2, zorder=3)
    ax.text(ref + 0.004, -0.7, f'given the settings, {ref:.2f}', ha='right', va='bottom')
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




def e19():
    """Slide 20: MNIST, after E19_mnist.py.

    Two panels of the report figure, without its map of the grid. (a) The grokking step against
    weight decay, medians over 5 seeds, one colour per initial scale and one marker per width, with
    the law T = ln(s / n_c) / (eta lambda) drawn through them. (b) The weight norm at grokking in
    units of the norm at s = 1, one point per run, with its mean. Together they say that the time
    follows weight decay alone and that what is fixed at grokking is the norm. The key sits in one row
    below both panels and every label sits where no data lie, so nothing covers a point.
    """
    import matplotlib
    scales = [3.0, 5.0, 7.0, 9.0]
    widths = [200, 400, 800]
    budget = 500_000
    runs = pd.read_parquet(C.REPO / 'mnist' / 'sweep' / 'runs.parquet')
    # The event of E19: test accuracy 0.80, censored at the budget.
    hit = ~runs['censored_acc80'].to_numpy(dtype=bool)
    runs = runs.assign(T=np.where(hit, runs['t_gen_acc80'].to_numpy(dtype=float), budget), hit=hit)
    # n_c is the weight norm at grokking in units of the default initialisation norm, over the runs
    # of E19's law: width 200 and above, decay above zero, and the level reached.
    law = runs[(runs.width >= 200) & (runs.eta_kappa > 0) & runs.hit].copy()
    law['n_c'] = law.weight_norm_at_gen / law.init_weight_norm * law.init_scale
    n_c = law.n_c.mean()

    cmap = matplotlib.colormaps[C.VIOLET_CMAP]
    colours = {s: cmap(v) for s, v in zip(scales, np.linspace(0.45, 0.95, len(scales)))}
    markers = {200: 'o', 400: 's', 800: '^'}
    fig, axes = plt.subplots(1, 2, figsize=COLUMN, layout='constrained', gridspec_kw={'width_ratios': [1.18, 0.82]})

    ax = axes[0]
    grok = runs[(runs.width >= 200) & (runs.eta_kappa > 0)]
    xs = np.array([1e-6, 4e-5])
    for s in scales:
        for w in widths:
            cell = grok[(grok.init_scale == s) & (grok.width == w)].groupby('eta_kappa')
            med = cell.apply(lambda g: g['T'].median() if g.hit.all() else np.nan, include_groups=False)
            ax.plot(med.index, med.values, marker=markers[w], color=colours[s], linestyle='none', markersize=11,
                    markerfacecolor=colours[s] if w == 400 else 'white', markeredgewidth=1.8, zorder=3)
        ax.plot(xs, np.log(s / n_c) / xs, color=colours[s], linewidth=2.0, alpha=0.85, zorder=2)
    ax.axhline(budget, color=C.MUTED, linewidth=1.4, linestyle=':')
    # The label sits at the right end of the line, above the runs at large decay, which grok far sooner.
    ax.text(3.8e-5, budget * 1.1, 'budget', color=C.MUTED, va='bottom', ha='right', fontsize=19)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(POW10)
    ax.set_xlabel(r'weight decay $\eta\lambda$')
    ax.set_ylabel('grokking step $T$')
    ax.set_title('(a) the time follows the decay', loc='left')
    h = [matplotlib.lines.Line2D([], [], color=colours[s], linewidth=3, label=f'$s={s:g}$') for s in scales]
    h += [matplotlib.lines.Line2D([], [], color=C.MUTED, marker=markers[w], linestyle='none', markersize=10,
                                  markerfacecolor=C.MUTED if w == 400 else 'white', markeredgewidth=1.8,
                                  label=f'$N={w}$') for w in widths]
    fig.legend(handles=h, ncol=len(scales), loc='outside lower center', handlelength=1.4, columnspacing=1.2,
               frameon=False, fontsize=19)

    ax = axes[1]
    rng = np.random.default_rng(1)
    for s in scales:
        g = law[law.init_scale == s]
        jitter = g.width.map({200: -0.3, 400: 0.0, 800: 0.3}) + rng.normal(0, 0.06, len(g))
        ax.plot(s + jitter, g.n_c, linestyle='none', marker='.', markersize=9, color=colours[s], alpha=0.8)
    ax.hlines(n_c, 2.0, 9.55, color=C.TEXT, linewidth=2)
    # The mean is named to the right of the last group of points, where the panel is empty.
    ax.text(9.75, n_c, f'mean\n{n_c:.2f}', color=C.TEXT, va='center', ha='left', fontsize=19, linespacing=1.1)
    ax.set_xticks(scales, [f'{s:g}' for s in scales])
    ax.set_xlim(2.0, 11.2)
    ax.set_ylim(1.75, 2.45)
    ax.set_yticks([1.8, 2.0, 2.2, 2.4])
    ax.set_xlabel('initial scale $s$')
    ax.set_ylabel('norm at $T$')
    ax.set_title('(b) the norm is fixed', loc='left')
    save(fig, 'E19_mnist')


if __name__ == '__main__':
    for draw in (e2, e3, e4, e5, e7, e8, e19):
        draw()
        print('drew', draw.__name__)
