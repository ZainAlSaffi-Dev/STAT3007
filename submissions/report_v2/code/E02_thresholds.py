"""E2. The statistics at the grokking step (Fig. 2, Section 4.2).

Every run that groks at level 0.95 is a point. The top row holds the runs at N = 100, grouped by
alpha. The bottom row holds the runs at alpha = 1, grouped by width. Points are coloured by eta
lambda, light to dark in the hue of the statistic. Each panel is labelled with the spread (standard
deviation) at grokking and the SD ratio of the placebo test, both from E03. The SD ratio is the
median placebo SD over the SD at grokking (the key 'tightening' in E03.json).
"""
import numpy as np

import common as C

SEED = 0
SEEDS = {'jitter': SEED}


def main():
    plt = C.setup_matplotlib()
    runs = C.load_runs()
    grok = runs[~runs.censored_grok95]
    e3 = C.read_json('E03')['groups']
    rng = np.random.default_rng(SEED)
    decays = sorted(grok.eta_kappa.unique())

    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.6), constrained_layout=True)
    rows = [('N100', grok[grok.width == 100], 'alpha', C.ALPHAS, r'$\alpha$', '$N=100$, by $\\alpha$'),
            ('alpha1', grok[grok.alpha == 1.0], 'width', C.WIDTHS, '$N$', '$\\alpha=1$, by width')]
    for r, (gname, g, by, levels, xlabel, title) in enumerate(rows):
        for c, stat in enumerate(C.STATS):
            ax = axes[r, c]
            cols = C.ramp(stat, len(decays))
            for i, lev in enumerate(levels):
                sub = g[g[by] == lev]
                x = i + rng.uniform(-0.18, 0.18, len(sub))
                colours = [cols[decays.index(d)] for d in sub.eta_kappa]
                ax.scatter(x, sub[f'{stat}_t_at_grok95'], c=colours, s=11, lw=0.4, edgecolors='white', zorder=3)
                if len(sub):
                    ax.hlines(np.median(sub[f'{stat}_t_at_grok95']), i - 0.3, i + 0.3, color=C.TEXT, lw=1.0, zorder=4)
            ax.set_xticks(range(len(levels)), [f'{v:g}' for v in levels])
            ax.set_xlabel(xlabel)
            ax.grid(axis='x', visible=False)
            res = e3[gname][stat]['sd']
            ax.set_title(f'{C.STAT_NAME[stat]}: SD {res["at_grok"]:.2g}, SD ratio {res["tightening"]:.2f}',
                         loc='left', fontsize=7.5)
            if stat == 'S':
                ax.set_ylabel(title)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls='', marker='o', color=C.ramp('S', len(decays))[i], label=C.decay_tick(d))
               for i, d in enumerate(decays)]
    handles.append(Line2D([], [], color=C.TEXT, lw=1, label='median'))
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), title=r'$\eta\lambda$ (darker is larger; '
               'each panel uses the hue of its statistic)', title_fontsize=6.5, handletextpad=0.2, columnspacing=0.9)
    C.save_figure(fig, 'E2_thresholds')
    C.write_json('E02', {'n_runs': {k: int(len(g)) for k, g, *_ in rows}}, SEEDS)


if __name__ == '__main__':
    main()
