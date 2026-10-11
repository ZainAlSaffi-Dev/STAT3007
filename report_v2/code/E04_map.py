"""E4. Grokking time and outcome over width and weight decay at alpha = 1 (Fig. 3, Section 4.3).

Colour is the median log10 of the grokking step at level 0.95 over the seeds that grok, drawn only
when at least 3 of the 5 seeds grok. Each cell is labelled with the seeds that grok out of 5. A cell
in which no seed groks also gets the most common phase of its seeds at the last step, after the
golden report Section 3.5 at threshold 0.9: M memorises only, F forgets, N never fits, and G means
that the test accuracy ends above 0.9 without reaching 0.95. common.phase() gives the rule.

The numbers check against the table of repoduced-code/tier2_reproduction.md, Section 7.
"""
import numpy as np

import common as C

SEEDS = {}

# The alpha = 1 table of tier2_reproduction.md Section 7: (seeds that grok, median T95, phases).
REFERENCE = {
    (50, 0.0): (5, 33000), (100, 0.0): (5, 23000), (200, 1e-5): (5, 106000), (1600, 1e-5): (5, 99000),
    (800, 3e-4): (0, None), (1600, 3e-4): (0, None), (100, 3e-4): (5, 9218), (400, 3e-4): (5, 13894),
}
REFERENCE_PHASES = {(800, 3e-4): 'GGGGM', (1600, 3e-4): 'FFFFF', (50, 1e-3): 'MMMMM', (200, 0.0): 'MMMMM',
                    (100, 3e-3): 'NNNNN'}


def main():
    plt = C.setup_matplotlib()
    import matplotlib
    runs = C.load_runs()
    ckpt = C.load_checkpoints(['run_id', 'step', 'train_acc'])
    cells = C.cell_outcomes(runs, ckpt)
    a1 = cells[cells.alpha == 1.0]

    for (w, d), (k, t) in REFERENCE.items():
        row = a1[(a1.width == w) & (a1.eta_kappa == d)].iloc[0]
        assert row.grokked == k and (t is None or row.median_T == t), (w, d, row.to_dict())
    for (w, d), ph in REFERENCE_PHASES.items():
        assert sorted(a1[(a1.width == w) & (a1.eta_kappa == d)].iloc[0].phases) == sorted(ph), (w, d)

    grid = np.full((len(C.WIDTHS), len(C.DECAYS)), np.nan)
    fig, ax = plt.subplots(figsize=(4.6, 2.5), constrained_layout=True)
    for _, row in a1.iterrows():
        i, j = C.WIDTHS.index(row.width), C.DECAYS.index(row.eta_kappa)
        grid[i, j] = np.log10(row.median_T)
    cmap = matplotlib.colormaps[C.VIOLET_CMAP].copy()
    cmap.set_bad('#f0efec')
    vmin, vmax = np.floor(np.nanmin(grid) * 2) / 2, np.ceil(np.nanmax(grid) * 2) / 2
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin - 0.4, vmax=vmax, aspect='auto', origin='lower')
    for _, row in a1.iterrows():
        i, j = C.WIDTHS.index(row.width), C.DECAYS.index(row.eta_kappa)
        val = grid[i, j]
        dark = np.isfinite(val) and (val - (vmin - 0.4)) / (vmax - vmin + 0.4) > 0.55
        label = f'{row.grokked}/5' + (f'\n{row.letter}' if row.grokked == 0 else '')
        ax.text(j, i, label, ha='center', va='center', fontsize=6.5, color='white' if dark else C.TEXT, linespacing=1.0)
    ax.set_xticks(range(len(C.DECAYS)), [C.decay_tick(d) for d in C.DECAYS])
    ax.set_yticks(range(len(C.WIDTHS)), [str(w) for w in C.WIDTHS])
    ax.set_xlabel(r'weight decay $\eta\lambda$')
    ax.set_ylabel('width $N$')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(r'median $\log_{10} T$')
    cb.outline.set_visible(False)
    C.save_figure(fig, 'E4_map')

    edge = {}
    for w in C.WIDTHS:
        row = a1[(a1.width == w) & (a1.grokked == 5)]
        edge[w] = float(row.eta_kappa.max()) if len(row) else None
    C.write_json('E04', {'alpha1': a1.to_dict('records'), 'largest_decay_all_grok': edge,
                         'no_decay_widths_that_grok': a1[(a1.eta_kappa == 0) & (a1.grokked > 0)].width.tolist()},
                 SEEDS)
    print(a1[['width', 'eta_kappa', 'grokked', 'median_T', 'phases', 'letter']].to_string())
    print('largest eta lambda at which all 5 grok, by width:', edge)


if __name__ == '__main__':
    main()
