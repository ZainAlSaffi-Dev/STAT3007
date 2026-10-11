"""E13. Outcomes over the whole grid (Appendix C).

For each alpha, a table over N and eta lambda gives the seeds that grok at level 0.95 out of 5 and
the median grokking step in thousands of steps, shown when at least 3 seeds grok. A cell in which no
seed groks shows the most common phase of its seeds, as in E4 (common.cell_outcomes).
"""
import numpy as np

import common as C

SEEDS = {}


def entry(row):
    if row.grokked == 0:
        return f'0 {row.letter}'
    if np.isfinite(row.median_T):
        return f'{row.grokked}: {row.median_T / 1000:.0f}'
    return f'{row.grokked}'


def main():
    runs = C.load_runs()
    ckpt = C.load_checkpoints(['run_id', 'step', 'train_acc'])
    cells = C.cell_outcomes(runs, ckpt)
    head = ' & '.join(C.fmt_decay(d) for d in C.DECAYS)
    lines = ['\\begin{tabular}{ll' + 'r' * len(C.DECAYS) + '}', '\\toprule',
             f'$\\alpha$ & $N$ & {head} \\\\']
    for a in C.ALPHAS:
        lines.append('\\midrule')
        for i, w in enumerate(C.WIDTHS):
            sub = cells[(cells.alpha == a) & (cells.width == w)].set_index('eta_kappa')
            vals = ' & '.join(entry(sub.loc[d]) for d in C.DECAYS)
            lines.append(f"{f'{a:g}' if i == 0 else ''} & {w} & {vals} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    C.write_table('E13_grid', '\n'.join(lines))
    summary = cells.groupby('alpha').apply(lambda g: {'cells_all_grok': int((g.grokked == 5).sum()),
                                                      'cells_none_grok': int((g.grokked == 0).sum()),
                                                      'runs_grok': int(g.grokked.sum())}, include_groups=False)
    C.write_json('E13', {'cells': cells.to_dict('records'), 'by_alpha': {f'{k:g}': v for k, v in summary.items()}},
                 SEEDS)
    print(summary.to_dict())


if __name__ == '__main__':
    main()
