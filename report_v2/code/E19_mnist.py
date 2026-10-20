"""E19. The second task: MNIST digits (Tier 4, Section 4.5, Fig. 6; Appendix F).

The section repeats the questions of Tier 1 and Tier 2 on the MNIST sweep of 19 October 2026
(mnist/sweep/, 560 runs: initial scale s in {3, 5, 7, 9}, width N in {100, 200, 400, 800}, per-step
weight decay eta lambda in {0, 1e-6, 3e-6, 6e-6, 1e-5, 2e-5, 4e-5}, 5 seeds, 500,000 steps). The
setup follows Liu, Michaud and Tegmark (2023), Omnigrok, Section 4: 1,000 training images, a
three-layer ReLU network of width 200, MSE on one-hot targets, and the initial weights multiplied
by s. Our network has no biases, so its tangent kernel has a closed form, and it trains by SGD
without momentum under a curvature learning-rate rule (mnist/README.md). The sweep tables are
committed in git, so this script reads them from the repository rather than from code/data.

Events follow the report: memorisation at training accuracy 0.99 and grokking at a test level,
counted from step 0, censored at the budget of 500,000. MNIST does not reach 0.95 within the
budget, so the level here is 0.80 (mnist_sweep.py, PRIMARY_LEVEL).

What the script computes.

- Scaling (Tier 2's question). The log-normal accelerated failure time model of report Eq. 9,
  log T = c + a log N + b log(eta lambda) + gamma log N log(eta lambda), fitted as in E06 with a
  cell-then-seed bootstrap. The main row is s = 7, N in {200, 400, 800}, eta lambda in
  {3e-6, 6e-6, 1e-5}, where every seed groks. Check rows: s = 5 and s = 9, the decays up to 4e-5,
  level 0.85, and the memorisation step.
- What is fixed at grokking (Tier 1's question). The weight norm at grokking in units of the
  default initialisation norm, n_c = ||theta_T|| / ||theta_0|| * s, over every run at N >= 200 with
  decay above zero that reaches the level, and the law of Liu et al. (2023), Section 2:
  T = ln(s / n_c) / (eta lambda) when the norm decays as exp(-eta lambda t). The placebo test of
  E03 on S, R and A, read from report_kernel.parquet at another run's grokking step, at a fixed
  width (N = 200), across widths (s = 7), and over all runs that grok.
- The medians of R, S and A at grokking by width at s = 7, and A at step 0.

Figure E19: (a) the grokking step against eta lambda, medians over seeds, coloured by s with one
marker per width and slope -1 guides; (b) n_c against s, one point per run; (c) R at grokking
against width at s = 7.
"""
import warnings

import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter

import common as C

SEED = 0
N_BOOT = 1_000
N_PERM = 2_000
SEEDS = {'bootstrap': SEED, 'n_boot': N_BOOT, 'permutations': SEED, 'n_permutations': N_PERM}
SWEEP = C.REPO / 'mnist' / 'sweep'
BUDGET = 500_000
LEVEL = 'acc80'
SCALES = [3.0, 5.0, 7.0, 9.0]
WIDTHS = [100, 200, 400, 800]
DECAYS = [0.0, 1e-6, 3e-6, 6e-6, 1e-5, 2e-5, 4e-5]
MAIN_SCALE = 7.0
MAIN_WIDTHS = [200, 400, 800]
MAIN_DECAYS = [3e-6, 6e-6, 1e-5]
X0 = float(np.mean(np.log(MAIN_WIDTHS)))
Y0 = float(np.mean(np.log(MAIN_DECAYS)))
ROWS = {
    'main': (MAIN_SCALE, MAIN_DECAYS, LEVEL),
    'scale5': (5.0, MAIN_DECAYS, LEVEL),
    'scale9': (9.0, MAIN_DECAYS, LEVEL),
    'wide': (MAIN_SCALE, MAIN_DECAYS + [2e-5, 4e-5], LEVEL),
    'level85': (MAIN_SCALE, MAIN_DECAYS, 'acc85'),
    'mem': (MAIN_SCALE, MAIN_DECAYS, 'mem'),
}
NAMES = {'x': 'a', 'y': 'b', 'xy': 'gamma', 'Intercept': 'c'}


def load():
    runs = pd.read_parquet(SWEEP / 'runs.parquet')
    assert len(runs) == 560 and (runs.status == 'ok').all()
    kernel = pd.read_parquet(SWEEP / 'report_kernel.parquet').sort_values(['run_id', 'step'])
    s0 = kernel[kernel.step == 0]
    assert np.allclose(s0.S_t, 1.0) and np.allclose(s0.R_t, 0.0), 'S and R must be 1 and 0 at step 0'
    runs = runs.sort_values(['init_scale', 'width', 'eta_kappa', 'seed']).reset_index(drop=True)
    return runs, kernel


def events(runs, level):
    """The event step of each run and whether it happened, for a level tag or 'mem'."""
    if level == 'mem':
        t = runs.t_mem.to_numpy(dtype=float)
        hit = runs.memorised.to_numpy(dtype=bool)
    else:
        t = runs[f't_gen_{level}'].to_numpy(dtype=float)
        hit = ~runs[f'censored_{level}'].to_numpy(dtype=bool)
    t = np.where(hit, t, BUDGET)
    return t, hit


# ---------------------------------------------------------------------------------------------
# Scaling, as in E06.

def design(df, level):
    d = pd.DataFrame({'x': np.log(df.width.to_numpy()) - X0, 'y': np.log(df.eta_kappa.to_numpy()) - Y0})
    d['xy'] = d.x * d.y
    d['T'], hit = events(df, level)
    d['E'] = hit.astype(int)
    d['cell'] = list(zip(df.width, df.eta_kappa))
    return d.reset_index(drop=True)


def fit(d):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = LogNormalAFTFitter()
        m.fit(d[['x', 'y', 'xy', 'T', 'E']], duration_col='T', event_col='E')
    mu = m.params_.loc['mu_']
    out = {NAMES[k]: float(v) for k, v in mu.items()}
    out['sigma'] = float(np.exp(m.params_.loc['sigma_'].iloc[0]))
    return out


def identified(pick):
    """True when the drawn cells can identify the four coefficients of the plane.

    The main range has only 9 cells, so a draw can hold too few distinct cells. The design
    x, y, xy, 1 needs the distinct cells to span four dimensions, which the rank of the design
    matrix of the distinct cells checks directly. E06 needs only two widths and two decays, since
    its 18 cells make a rank deficit rare, but with 9 cells it is common and the fit of a
    deficient draw returns arbitrary coefficients.
    """
    cells = sorted(set(pick))
    x = np.log([c[0] for c in cells]) - X0
    y = np.log([c[1] for c in cells]) - Y0
    design = np.column_stack([np.ones(len(cells)), x, y, x * y])
    return np.linalg.matrix_rank(design) == 4


def bootstrap(d, rng):
    cells = sorted(set(d.cell))
    by_cell = {c: d.index[d.cell == c].to_numpy() for c in cells}
    draws, skipped = [], 0
    while len(draws) < N_BOOT:
        pick = [cells[i] for i in rng.integers(len(cells), size=len(cells))]
        if not identified(pick):
            skipped += 1
            continue
        idx = np.concatenate([rng.choice(by_cell[c], size=len(by_cell[c]), replace=True) for c in pick])
        b = d.loc[idx].reset_index(drop=True)
        if b.E.sum() < 5:
            skipped += 1
            continue
        try:
            est = fit(b)
        except Exception:
            skipped += 1
            continue
        # A fit that did not converge leaves a coefficient far outside the range of any draw of
        # these data, so it is dropped and counted.
        if not all(np.isfinite(v) for v in est.values()) or max(abs(est[k]) for k in ('a', 'b', 'gamma')) > 5:
            skipped += 1
            continue
        draws.append(est)
    df = pd.DataFrame(draws)
    return {k: [float(df[k].quantile(0.025)), float(df[k].quantile(0.975))] for k in df.columns}, skipped


def scaling_row(args):
    """One row of the fit with its bootstrap, run in its own process; the rng is seeded by the row."""
    name, df, level, seed = args
    d = design(df, level)
    est = fit(d)
    ci, skipped = bootstrap(d, np.random.default_rng(seed))
    return name, {'event': level, 'n_runs': int(len(d)), 'n_censored': int((d.E == 0).sum()), 'estimate': est,
                  'ci_cells': ci, 'boot_redrawn_cells': skipped, 'bootstrap_seed': seed}


# ---------------------------------------------------------------------------------------------
# The placebo test, as in E03, on the kernel table.

def derangements(n, k, rng):
    out = np.empty((k, n), dtype=int)
    i = 0
    while i < k:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            out[i] = p
            i += 1
    return out


def placebo(group, wide, rng):
    ids = group.run_id.to_numpy()
    t_grok, hit = events(group, LEVEL)
    assert hit.all()
    t_mem = group.t_mem.to_numpy(dtype=float)
    perms = derangements(len(ids), N_PERM, rng)
    res = {'n_runs': int(len(ids))}
    for stat, col in (('S', 'S_t'), ('R', 'R_t'), ('A', 'A_t')):
        w = wide[col]
        at_grok = C.value_at_step(w, ids, t_grok)
        at_mem = C.value_at_step(w, ids, t_mem)
        sub = w.loc[ids]
        grid = sub.columns.to_numpy()
        colidx = np.searchsorted(grid, t_grok[perms], side='right') - 1
        plac = sub.to_numpy()[np.arange(len(ids))[None, :], colidx]
        true = float(C.spread(at_grok, 'sd'))
        p_spread = C.spread(plac, 'sd')
        res[stat] = {'at_grok': true, 'at_mem': float(C.spread(at_mem, 'sd')),
                     'placebo_median': float(np.median(p_spread)), 'tightening': float(np.median(p_spread) / true),
                     'p': float(np.mean(p_spread <= true)), 'median_at_grok': float(np.median(at_grok)),
                     'median_at_mem': float(np.median(at_mem))}
    return res


# ---------------------------------------------------------------------------------------------

def main():
    runs, kernel = load()
    rng = np.random.default_rng(SEED)
    out = {'design': {'scales': SCALES, 'widths': WIDTHS, 'decays': DECAYS, 'seeds': 5, 'budget': BUDGET,
                      'level': 0.80, 'n_runs': int(len(runs))},
           'centre': {'x0_logN': X0, 'y0_log_decay': Y0, 'N0': float(np.exp(X0)), 'decay0': float(np.exp(Y0))},
           'rows': {}}

    # Scaling rows, one process each. Row i uses bootstrap seed SEED + i.
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for i, (name, (scale, decays, level)) in enumerate(ROWS.items()):
        df = runs[(runs.init_scale == scale) & runs.width.isin(MAIN_WIDTHS) & runs.eta_kappa.isin(decays)]
        jobs.append((name, df, level, SEED + i))
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = dict(pool.map(scaling_row, jobs))
    for name, (scale, decays, level) in ROWS.items():
        row = results[name]
        out['rows'][name] = {'scale': scale, 'decays': decays, **row}
        est, ci = row['estimate'], row['ci_cells']
        print(f"{name:8s} n={row['n_runs']} cens={row['n_censored']} redrawn={row['boot_redrawn_cells']}  " + '  '.join(
            f"{k} {est[k]:+.3f} [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]" for k in ('a', 'b', 'gamma', 'sigma')), flush=True)

    # The critical norm and the law of Liu et al. (2023).
    T, hit = events(runs, LEVEL)
    runs = runs.assign(T=T, hit=hit)
    law = runs[(runs.width >= 200) & (runs.eta_kappa > 0) & runs.hit].copy()
    law['n_c'] = law.weight_norm_at_gen / law.init_weight_norm * law.init_scale
    law['decay_T'] = law.eta_kappa * law['T']
    law['pure_decay_ratio'] = np.exp(-law.decay_T)                     # ||theta_T|| / ||theta_0|| if only decay acts
    law['measured_ratio'] = law.weight_norm_at_gen / law.init_weight_norm
    n_c = float(law.n_c.mean())
    law['predicted_decay_T'] = np.log(law.init_scale / n_c)
    corr = float(np.corrcoef(law.predicted_decay_T, law.decay_T)[0, 1])
    strong = law[(law.init_scale >= 5) & law.eta_kappa.isin(MAIN_DECAYS)]
    out['law'] = {
        'n_runs': int(len(law)), 'n_c_mean': n_c, 'n_c_sd': float(law.n_c.std(ddof=1)),
        'n_c_min': float(law.n_c.min()), 'n_c_max': float(law.n_c.max()),
        'n_c_sd_over_mean': float(law.n_c.std(ddof=1) / n_c),
        'n_c_by_scale': {f'{s:g}': float(g.n_c.median()) for s, g in law.groupby('init_scale')},
        'n_c_by_width': {str(int(w)): float(g.n_c.median()) for w, g in law.groupby('width')},
        'decay_T_by_scale': {f'{s:g}': float(g.decay_T.median()) for s, g in law.groupby('init_scale')},
        'decay_T_sd_within_scale_max': float(law.groupby('init_scale').decay_T.std(ddof=1).max()),
        'pure_decay_max_abs_gap': float((law.pure_decay_ratio - law.measured_ratio).abs().max()),
        'corr_predicted_measured': corr,
        'strong_n_runs': int(len(strong)), 'strong_n_c_mean': float(strong.n_c.mean()),
        'strong_n_c_sd': float(strong.n_c.std(ddof=1)),
        'norm_at_mem_sd_over_mean': float((law.weight_norm_at_mem / law.init_weight_norm * law.init_scale).std(ddof=1)
                                          / (law.weight_norm_at_mem / law.init_weight_norm * law.init_scale).mean()),
    }
    print(f"law: n_c = {n_c:.3f} +- {out['law']['n_c_sd']:.3f} over {len(law)} runs; corr {corr:.4f}; "
          f"pure-decay gap {out['law']['pure_decay_max_abs_gap']:.4f}")

    # Width 100 and the cells that do not follow the law.
    w100 = runs[(runs.width == 100) & runs.hit & (runs.eta_kappa > 0)]
    out['width100'] = {'n_reached': int(len(w100)), 'n_strong': int((runs[runs.width == 100].verdict == 'strong').sum()),
                       'decay_T_min': float((w100.eta_kappa * w100['T']).min()),
                       'decay_T_max': float((w100.eta_kappa * w100['T']).max()),
                       'test_acc_at_mem_median': float(w100.test_acc_at_mem.median())}
    out['verdicts'] = {k: int(v) for k, v in runs.verdict.value_counts().items()}
    for s in (3.0, 7.0):
        g = runs[(runs.init_scale == s) & (runs.width >= 200) & runs.memorised]
        out[f'scale{s:g}'] = {'test_acc_at_mem_median': float(g.test_acc_at_mem.median()), 'n_runs': int(len(g))}
    out['no_decay'] = {'n_reached': int(runs[(runs.eta_kappa == 0) & runs.hit].shape[0]),
                       'n_runs': int((runs.eta_kappa == 0).sum())}

    # Statistics at grokking by width at s = 7 on the main decays, and A at step 0.
    m7 = runs[(runs.init_scale == MAIN_SCALE) & runs.eta_kappa.isin(MAIN_DECAYS) & runs.hit & (runs.width >= 200)]
    out['at_grok_by_width'] = {str(int(w)): {'R': float(g.R_sum_at_gen.median()), 'S': float(g.S_sum_at_gen.median()),
                                             'A': float(g.A_sum_at_gen.median()), 'A0': float(g.A_0.median()),
                                             'n_c': float((g.weight_norm_at_gen / g.init_weight_norm * g.init_scale).median()),
                                             'T_median': float(g['T'].median()), 'n_runs': int(len(g))}
                               for w, g in m7.groupby('width')}
    allgrok = runs[(runs.width >= 200) & (runs.eta_kappa > 0) & runs.hit]
    out['A'] = {'A0_median': float(allgrok.A_0.median()), 'A_grok_median': float(allgrok.A_sum_at_gen.median()),
                'A_mem_median': float(allgrok.A_sum_at_mem.median()), 'n_runs': int(len(allgrok))}

    # Placebo.
    wide = {col: kernel.pivot(index='run_id', columns='step', values=col) for col in ('S_t', 'R_t', 'A_t')}
    groups = {'N200': allgrok[allgrok.width == 200], 'scale7': allgrok[allgrok.init_scale == MAIN_SCALE],
              'all': allgrok}
    out['placebo'] = {k: placebo(g, wide, rng) for k, g in groups.items()}
    for k, g in out['placebo'].items():
        print(f"placebo {k} ({g['n_runs']} runs): " + '  '.join(
            f"{s} ratio {g[s]['tightening']:.2f} p {g[s]['p']:.3f}" for s in ('S', 'R', 'A')))

    # Compute.
    import json
    manifest = json.loads((SWEEP / 'manifest.json').read_text())
    out['compute'] = {'summed_hours': manifest['total_seconds'] / 3600, 'gpu': 'NVIDIA H100',
                      'written': manifest['written'], 'torch': manifest['versions']['torch']}

    C.write_json('E19', out, SEEDS)
    figure(runs, law, m7)


def mnist_phase(runs):
    """The phase letters of common.phase() for the MNIST runs, from the run table alone.

    G: training and test accuracy both end above 0.9. M: only training accuracy does. F: the run
    memorised (training accuracy reached 0.99 at a checkpoint) and ends below 0.9. N: none of these.
    The sweep's checkpoint table is not in git, so F uses the memorisation flag in place of the peak
    training accuracy that common.phase() reads from the checkpoints.
    """
    th = C.PHASE_THRESHOLD
    fitted = runs.train_acc_final.to_numpy() > th
    crossed = runs.memorised.to_numpy(dtype=bool)
    return np.select([fitted & (runs.test_acc_final.to_numpy() > th), fitted, crossed], ['G', 'M', 'F'], 'N')


def figure(runs, law, m7):
    plt = C.setup_matplotlib()
    import matplotlib
    cmap = matplotlib.colormaps[C.VIOLET_CMAP]
    colours = {s: cmap(v) for s, v in zip(SCALES, np.linspace(0.45, 0.95, len(SCALES)))}
    markers = {200: 'o', 400: 's', 800: '^'}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.1, 0.8, 1.45]})

    ax = axes[0]
    grok = runs[(runs.width >= 200) & (runs.eta_kappa > 0)]
    xs = np.array([1e-6, 4e-5])
    for s in SCALES:
        for w in MAIN_WIDTHS:
            cell = grok[(grok.init_scale == s) & (grok.width == w)].groupby('eta_kappa')
            med = cell.apply(lambda g: g['T'].median() if g.hit.all() else np.nan, include_groups=False)
            ax.plot(med.index, med.values, marker=markers[w], color=colours[s], linestyle='none',
                    markersize=3.5, markerfacecolor=colours[s] if w == 400 else 'white', markeredgewidth=0.8)
        # The law T = ln(s / n_c) / (eta lambda), with n_c the mean over all runs.
        n_c = law.n_c.mean()
        ax.plot(xs, np.log(s / n_c) / xs, color=colours[s], linewidth=0.8, alpha=0.8)
    ax.axhline(BUDGET, color=C.MUTED, linewidth=0.6, linestyle=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'weight decay $\eta\lambda$')
    ax.set_ylabel('grokking step $T$')
    ax.set_title('(a) time against decay', loc='left')
    h = [matplotlib.lines.Line2D([], [], color=colours[s], linewidth=1.2, label=f'$s={s:g}$') for s in SCALES]
    h += [matplotlib.lines.Line2D([], [], color=C.MUTED, marker=markers[w], linestyle='none', markersize=3.5,
                                  markerfacecolor=C.MUTED if w == 400 else 'white', label=f'$N={w}$')
          for w in MAIN_WIDTHS]
    ax.legend(handles=h, ncol=2, loc='lower left', fontsize=5.5, handlelength=1.4, columnspacing=0.8)

    ax = axes[1]
    for s in SCALES:
        g = law[law.init_scale == s]
        jitter = (g.width.map({200: -0.25, 400: 0.0, 800: 0.25}) + np.random.default_rng(1).normal(0, 0.05, len(g)))
        ax.plot(s + jitter, g.n_c, linestyle='none', marker='.', markersize=3, color=colours[s], alpha=0.8)
    ax.axhline(law.n_c.mean(), color=C.TEXT, linewidth=0.8)
    ax.set_xticks(SCALES, [f'{s:g}' for s in SCALES])
    ax.set_xlabel('initial scale $s$')
    ax.set_ylabel(r'weight norm at $T$, $n_c$')
    ax.set_ylim(1.7, 2.5)
    ax.set_title('(b) norm at grokking', loc='left')

    ax = axes[2]
    # The time and outcome map of Fig. 3 (E04), at s = 7 over every width and decay. A tile shows
    # the median grokking step over the seeds when at least 3 of 5 grok, and the label gives the
    # seeds that grok out of 5, with the most common phase letter when none does.
    s7 = runs[runs.init_scale == MAIN_SCALE].copy()
    s7['phase'] = mnist_phase(s7)
    grid = np.full((len(WIDTHS), len(DECAYS)), np.nan)
    labels = {}
    for (w, d), cell in s7.groupby(['width', 'eta_kappa']):
        i, j = WIDTHS.index(int(w)), DECAYS.index(float(d))
        k = int(cell.hit.sum())
        if k >= 3:
            grid[i, j] = np.log10(np.median(cell.loc[cell.hit, 'T']))
        top = cell.phase.value_counts().index[0] if k == 0 else ''
        labels[(i, j)] = f'{k}/5' + (f'\n{top}' if k == 0 else '')
    cmap = matplotlib.colormaps[C.VIOLET_CMAP].copy()
    cmap.set_bad('#f0efec')
    vmin, vmax = np.floor(np.nanmin(grid) * 2) / 2, np.ceil(np.nanmax(grid) * 2) / 2
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin - 0.4, vmax=vmax, aspect='auto', origin='lower')
    for (i, j), label in labels.items():
        val = grid[i, j]
        dark = np.isfinite(val) and (val - (vmin - 0.4)) / (vmax - vmin + 0.4) > 0.55
        ax.text(j, i, label, ha='center', va='center', fontsize=5.5, color='white' if dark else C.TEXT,
                linespacing=1.0)
    ax.set_xticks(range(len(DECAYS)), [C.decay_tick(d) for d in DECAYS], fontsize=5.5)
    ax.set_yticks(range(len(WIDTHS)), [str(w) for w in WIDTHS])
    ax.set_xlabel(r'weight decay $\eta\lambda$')
    ax.set_ylabel('width $N$')
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(r'median $\log_{10} T$')
    cb.outline.set_visible(False)
    ax.set_title('(c) time and outcome at $s=7$', loc='left')
    C.save_figure(fig, 'E19_mnist')


if __name__ == '__main__':
    main()
