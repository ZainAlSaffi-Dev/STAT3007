"""E1. Baseline dynamics (Fig. 1, Section 4.1), with numbers N2 and N5.

The cells are N = 100 and eta lambda = 0, with alpha in {0.5, 1, 2} and 5 seeds each. This is the
baseline of Kumar et al. (2024), App. 8.3, arXiv v3, without the alpha = 1.5 they also sweep.

Figure: four panels against log step. (a) training and test accuracy, (b) S_t, (c) R_t, (d) A_t.
Each line is the median over seeds, with the range over seeds shaded. A hollow marker sits at the
median memorisation step and a filled marker at the median grokking step (level 0.95).

N2: the share of D_t^2 carried by the change of scale. By report Eq. 4,
D^2 = S^2 - 2S(1 - R) + 1 = (S - 1)^2 + 2SR, so the scale part is (S - 1)^2 and the rest is 2SR.
The share is taken at memorisation, at grokking and at the last step, per run, and its median over
the seeds of a cell is reported.

N5: the median and the range of the memorisation and grokking steps by alpha, and the number of
seeds that grok within the budget.
"""
import numpy as np

import common as C

SEEDS = {}  # nothing random


def scale_share(S, R):
    D2 = S ** 2 - 2 * S * (1 - R) + 1
    return (S - 1) ** 2 / D2


def main():
    plt = C.setup_matplotlib()
    runs = C.load_runs()
    base = runs[(runs.width == 100) & (runs.eta_kappa == 0)]
    ckpt = C.load_checkpoints(['run_id', 'step', 'train_acc', 'test_acc', 'S_sum', 'R_sum', 'A_sum'])
    ckpt = ckpt[ckpt.run_id.isin(base.run_id) & (ckpt.step >= 1)]

    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.25), constrained_layout=True)
    acc_grey = {0.5: '#b5b4b0', 1.0: '#6e6d69', 2.0: '#0b0b0b'}
    panels = [('acc', 'accuracy'), ('S', C.STAT_NAME['S']), ('R', C.STAT_NAME['R']), ('A', C.STAT_NAME['A'])]
    numbers = {'N5': {}, 'N2': {}}

    for alpha in C.ALPHAS:
        cell = base[base.alpha == alpha]
        c = ckpt[ckpt.run_id.isin(cell.run_id)]
        t_mem, t_grok = np.median(cell.t_mem), np.median(cell.t_grok95)
        grokked = int((~cell.censored_grok95).sum())
        numbers['N5'][f'{alpha:g}'] = {
            'mem_median': t_mem, 'mem_min': cell.t_mem.min(), 'mem_max': cell.t_mem.max(),
            'grok95_median': np.median(cell.t_grok95[~cell.censored_grok95]) if grokked else None,
            'grok95_min': cell.t_grok95[~cell.censored_grok95].min() if grokked else None,
            'grok95_max': cell.t_grok95[~cell.censored_grok95].max() if grokked else None,
            'grok95_median_all': t_grok, 'grokked': grokked, 'seeds': len(cell),
            'censored_seeds': cell.seed[cell.censored_grok95].tolist()}
        share = {}
        for at, (S, R) in {'mem': (cell.S_t_at_mem, cell.R_t_at_mem), 'grok95': (cell.S_t_at_grok95, cell.R_t_at_grok95),
                           'final': (cell.S_t_final, cell.R_t_final)}.items():
            v = scale_share(S.to_numpy(), R.to_numpy())
            share[at] = np.nanmedian(v) if np.isfinite(v).any() else None
        share['S_final_median'] = np.median(cell.S_t_final)
        share['R_final_median'] = np.median(cell.R_t_final)
        numbers['N2'][f'{alpha:g}'] = share

        g = c.groupby('step')
        for ax, (key, _) in zip(axes, panels):
            if key == 'acc':
                for col, ls, lab in (('train_acc', '--', 'train'), ('test_acc', '-', 'test')):
                    q = g[col].quantile([0, 0.5, 1]).unstack()
                    ax.fill_between(q.index, q[0], q[1], color=acc_grey[alpha], alpha=0.18, lw=0)
                    ax.plot(q.index, q[0.5], ls=ls, color=acc_grey[alpha])
                med_test = g['test_acc'].median()
                med_train = g['train_acc'].median()
                ax.plot(t_mem, np.interp(t_mem, med_train.index, med_train), 'o', mfc='white',
                        mec=acc_grey[alpha], mew=1.2, ms=5, zorder=5)
                if grokked >= 3:
                    ax.plot(t_grok, np.interp(t_grok, med_test.index, med_test), 'o', color=acc_grey[alpha], ms=5,
                            zorder=5)
            else:
                colour = C.ramp(key, 3)[C.ALPHAS.index(alpha)]
                col = C.CHECKPOINT_STATS[key]
                q = g[col].quantile([0, 0.5, 1]).unstack()
                ax.fill_between(q.index, q[0], q[1], color=colour, alpha=0.12, lw=0)
                ax.plot(q.index, q[0.5], color=colour, label=f'$\\alpha={alpha:g}$')
                ax.plot(t_mem, np.interp(t_mem, q.index, q[0.5]), 'o', mfc='white', mec=colour, mew=1.2, ms=5, zorder=5)
                if grokked >= 3:
                    ax.plot(t_grok, np.interp(t_grok, q.index, q[0.5]), 'o', color=colour, ms=5, zorder=5)

    for ax, (key, name), letter in zip(axes, panels, 'abcd'):
        ax.set_xscale('log')
        ax.set_xlim(1, C.BUDGET)
        ax.set_xticks([1e1, 1e3, 1e5])
        ax.set_xlabel('step')
        ax.set_title(f'({letter}) {name}', loc='left')
        if key == 'S':
            ax.axhline(1, color=C.MUTED, lw=0.6, ls=':')
        if key == 'R':
            ax.set_yscale('log')
            ax.set_ylim(1e-5, None)
        if key == 'S':
            ax.legend(loc='upper left', handlelength=1.2, title='darker is lazier', title_fontsize=6.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=acc_grey[a], label=f'$\\alpha={a:g}$') for a in C.ALPHAS]
    handles += [Line2D([], [], color=C.MUTED, ls='--', label='train'), Line2D([], [], color=C.MUTED, label='test'),
                Line2D([], [], ls='', marker='o', mfc='white', mec=C.MUTED, label='memorised'),
                Line2D([], [], ls='', marker='o', color=C.MUTED, label='grokked')]
    fig.legend(handles=handles, loc='outside lower center', ncol=7, handlelength=1.6, columnspacing=1.2)
    axes[0].set_ylim(0, 1.02)
    C.save_figure(fig, 'E1_baseline')
    C.write_json('E01', numbers, SEEDS)
    for a, v in numbers['N5'].items():
        print(f"alpha {a}: mem {v['mem_median']:.0f} [{v['mem_min']}, {v['mem_max']}], grok95 {v['grok95_median']} "
              f"[{v['grok95_min']}, {v['grok95_max']}], {v['grokked']}/5")
    for a, v in numbers['N2'].items():
        print(f'alpha {a}: scale share of D^2', {k: (round(x, 3) if x is not None else None) for k, x in v.items()})


if __name__ == '__main__':
    main()
