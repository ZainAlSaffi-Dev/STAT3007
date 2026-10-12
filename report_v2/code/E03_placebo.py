"""E3. Threshold table with the placebo test (Table 3, Section 4.2), with numbers N6, N7 and N8.

For each statistic S, R and A, and for each group of runs that grok at level 0.95, the script takes
the spread across runs of the statistic at three steps of each run:

- at its own grokking step;
- at its own memorisation step;
- at the grokking step of another run of the group (placebo). The other run is drawn by a random
  permutation with no fixed points. A run is read at the last checkpoint at or before that step.

The spread is the standard deviation across runs (ddof 1). The group chose it on 11 October 2026 in
place of the coefficient of variation; set SPREAD = 'cv' to restore the old measure. The CV is still
written to the JSON for audit. The tightening ratio is the median placebo spread over 2,000
permutations divided by the spread at the true events. The permutation p is the share of
permutations whose placebo spread is at most the true spread. A ratio near 1 and a large p mean that
the statistic is not tied to the event.

Groups: N = 100 over all alpha and eta lambda; alpha = 1 over all N and eta lambda; all runs.

N6: the median of A at step 0 and at grokking, over all runs that grok.
N7: the median of R and of S at grokking by width, over all alpha and eta lambda.
N8: the cells in which no seed groks but whose runs reach the R band of the grokking runs at the same
width. The band is the range (minimum to maximum) of R at grokking over the runs of that width that
grok. A run reaches the band when its largest R over training is at least the band's minimum. The
same is done for A, with the band of all runs that grok. A stricter count uses the median of R at
grokking at that width in place of the minimum.

The method is that of repoduced-code/tier1_v2.ipynb, Section 7, on the report's statistics.
"""
import numpy as np

import common as C

SPREAD = 'sd'
N_PERM = 2_000
SEED = 0
SEEDS = {'permutations': SEED, 'n_permutations': N_PERM}


def derangements(n, k, rng):
    out = np.empty((k, n), dtype=int)
    i = 0
    while i < k:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            out[i] = p
            i += 1
    return out


def placebo(group, wide, rng, level='grok95'):
    """The spreads and the placebo test of each statistic for one group of runs that grok at a level."""
    ids = group.run_id.to_numpy()
    t_grok = group[f't_{level}'].to_numpy()
    t_mem = group.t_mem.to_numpy()
    perms = derangements(len(ids), N_PERM, rng)
    res = {'n_runs': len(ids)}
    for stat in C.STATS:
        w = wide[stat]
        at_grok = C.value_at_step(w, ids, t_grok)
        assert np.allclose(at_grok, group[f'{stat}_t_at_{level}'], rtol=1e-9, atol=1e-12)
        at_mem = C.value_at_step(w, ids, t_mem)
        # Run i read at the grokking step of run perm[i].
        sub = w.loc[ids]
        grid = sub.columns.to_numpy()
        col = np.searchsorted(grid, t_grok[perms], side='right') - 1
        plac = sub.to_numpy()[np.arange(len(ids))[None, :], col]
        r = {}
        for kind in ('sd', 'cv'):
            true = float(C.spread(at_grok, kind))
            p_spread = C.spread(plac, kind)
            r[kind] = {'at_grok': true, 'at_mem': float(C.spread(at_mem, kind)),
                       'placebo_median': float(np.median(p_spread)),
                       'tightening': float(np.median(p_spread) / true),
                       'p': float(np.mean(p_spread <= true))}
        r['mean_at_grok'] = float(np.mean(at_grok))
        r['median_at_grok'] = float(np.median(at_grok))
        r['mean_placebo'] = float(np.mean(plac))
        res[stat] = r
    return res


def main():
    runs = C.load_runs()
    ckpt = C.load_checkpoints(['run_id', 'step', 'S_sum', 'R_sum', 'A_sum'])
    wide = {s: C.pivot(ckpt, C.CHECKPOINT_STATS[s]) for s in C.STATS}
    grok = runs[~runs.censored_grok95]
    groups = {'N100': grok[grok.width == 100], 'alpha1': grok[grok.alpha == 1.0], 'all': grok}

    rng = np.random.default_rng(SEED)
    out = {'spread': SPREAD, 'groups': {}}
    for name, g in groups.items():
        out['groups'][name] = placebo(g, wide, rng)

    # N6.
    a0 = C.value_at_step(wide['A'], grok.run_id, np.zeros(len(grok)))
    out['N6'] = {'A_step0_median': float(np.median(a0)), 'A_grok_median': float(np.median(grok.A_t_at_grok95)),
                 'A_final_median': float(np.median(grok.A_t_final)), 'n_runs': len(grok)}
    # N7.
    out['N7'] = {int(w): {'R_median': float(g.R_t_at_grok95.median()), 'S_median': float(g.S_t_at_grok95.median()),
                          'n_runs': len(g)} for w, g in grok.groupby('width')}
    # N8.
    maxR = ckpt.groupby('run_id')['R_sum'].max()
    maxA = ckpt.groupby('run_id')['A_sum'].max()
    a_lo = grok.A_t_at_grok95.min()
    n8 = []
    for (alpha, width, decay), cell in runs.groupby(['alpha', 'width', 'eta_kappa']):
        if (~cell.censored_grok95).any():
            continue
        band = grok[grok.width == width].R_t_at_grok95
        r_in = int((cell.run_id.map(maxR) >= band.min()).sum()) if len(band) else 0
        a_in = int((cell.run_id.map(maxA) >= a_lo).sum())
        if r_in or a_in:
            n8.append({'alpha': alpha, 'width': int(width), 'eta_kappa': decay, 'runs_in_R_band': r_in,
                       'runs_memorised': int((~cell.censored_mem).sum()),
                       'runs_above_R_median': int((cell.run_id.map(maxR) >= band.median()).sum()) if len(band) else 0,
                       'test_acc_final_median': float(cell.test_acc_final.median()),
                       'runs_in_A_band': a_in, 'R_band_min': float(band.min()) if len(band) else None,
                       'max_R_median': float(cell.run_id.map(maxR).median())})
    out['N8'] = {'cells': n8, 'n_cells_R': sum(c['runs_in_R_band'] > 0 for c in n8),
                 'n_cells_R_memorised': sum(c['runs_in_R_band'] > 0 and c['runs_memorised'] > 0 for c in n8),
                 'n_cells_no_grok_memorised': int(sum((~c.censored_grok95).sum() == 0 and (~c.censored_mem).any()
                                                      for _, c in runs.groupby(['alpha', 'width', 'eta_kappa']))),
                 'n_cells_above_R_median_memorised': sum(c['runs_above_R_median'] > 0 and c['runs_memorised'] > 0
                                                         for c in n8),
                 'n_cells_R_N100': sum(c['runs_in_R_band'] > 0 and c['width'] == 100 for c in n8),
                 'n_cells_no_grok': int(sum((~c.censored_grok95).sum() == 0
                                            for _, c in runs.groupby(['alpha', 'width', 'eta_kappa'])))}
    C.write_json('E03', out, SEEDS)

    for name, g in out['groups'].items():
        print(f"{name} ({g['n_runs']} runs)")
        for s in C.STATS:
            for kind in ('sd', 'cv'):
                r = g[s][kind]
                print(f"  {s} {kind}: grok {r['at_grok']:.4g} mem {r['at_mem']:.4g} placebo {r['placebo_median']:.4g} "
                      f"ratio {r['tightening']:.2f} p {r['p']:.4f}")
    print('N6', out['N6'])
    print('N7', out['N7'])
    print('N8 cells with R in band:', out['N8']['n_cells_R'], 'of', out['N8']['n_cells_no_grok'])
    print('  of the cells that memorise:', out['N8']['n_cells_R_memorised'], 'of', out['N8']['n_cells_no_grok_memorised'])
    for c in n8:
        if c['runs_in_R_band'] and c['runs_memorised']:
            print('   ', c['alpha'], c['width'], c['eta_kappa'], 'R in band', c['runs_in_R_band'], 'above median', c['runs_above_R_median'], 'mem', c['runs_memorised'],
                  'test acc', round(c['test_acc_final_median'], 3))


if __name__ == '__main__':
    main()
