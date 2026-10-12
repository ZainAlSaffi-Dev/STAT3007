"""E16. Sensitivity to the grokking level (Appendix F).

Repeats the placebo test of E3 at levels 0.80 and 0.90 (2,000 permutations, seed 0, standard
deviation as the spread) and collects the E6 fits at those levels. Writes two LaTeX tables.
"""
import numpy as np

import common as C
from E03_placebo import N_PERM, SEED, placebo

SEEDS = {'permutations': SEED, 'n_permutations': N_PERM}


def main():
    runs = C.load_runs()
    ckpt = C.load_checkpoints(['run_id', 'step', 'S_sum', 'R_sum', 'A_sum'])
    wide = {s: C.pivot(ckpt, C.CHECKPOINT_STATS[s]) for s in C.STATS}
    out = {}
    for level in ('grok80', 'grok90', 'grok95'):
        # One generator per level, used in the order of E03, so that the 0.95 rows equal Table 3.
        rng = np.random.default_rng(SEED)
        g = runs[~runs[f'censored_{level}']]
        out[level] = {name: placebo(sub, wide, rng, level) for name, sub in
                      (('N100', g[g.width == 100]), ('alpha1', g[g.alpha == 1.0]))}
    e3 = C.read_json('E03')['groups']
    for name in ('N100', 'alpha1'):
        for s in C.STATS:
            for key in ('at_grok', 'placebo_median', 'p'):
                assert abs(out['grok95'][name][s]['sd'][key] - e3[name][s]['sd'][key]) < 1e-12, (name, s, key)
    e6 = C.read_json('E06')['rows']

    lines = ['\\begin{tabular}{lll' + 'r' * 4 + '}', '\\toprule',
             'Level & Runs & Statistic & SD at grokking & Placebo SD & Ratio & $p$ \\\\']
    for level, lab in (('grok80', '0.80'), ('grok90', '0.90'), ('grok95', '0.95')):
        for name, glab in (('N100', '$N=100$'), ('alpha1', '$\\alpha=1$')):
            lines.append('\\midrule' if name == 'N100' else '\\addlinespace')
            for i, s in enumerate(C.STATS):
                r = out[level][name][s]['sd']
                first = f"{lab} & {glab} ({out[level][name]['n_runs']})" if i == 0 else ' & '
                lines.append(f"{first} & ${s}_t$ & {r['at_grok']:.3g} & {r['placebo_median']:.3g} & "
                             f"{r['tightening']:.2f} & {r['p']:.3f} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    C.write_table('E16_thresholds', '\n'.join(lines))

    def cell(row, k):
        e, ci = row['estimate'][k], row['ci_cells'][k]
        return f'${e:+.2f}$ [${ci[0]:+.2f}$, ${ci[1]:+.2f}$]'
    lines = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Level & $a$ & $b$ & $\\gamma$ & $\\sigma$ \\\\', '\\midrule']
    for key, lab in (('level80', '0.80'), ('level90', '0.90'), ('main', '0.95')):
        r = e6[key]
        lines.append(f"{lab} & {cell(r, 'a')} & {cell(r, 'b')} & {cell(r, 'gamma')} & {r['estimate']['sigma']:.2f} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    C.write_table('E16_scaling', '\n'.join(lines))
    C.write_json('E16', out, SEEDS)
    for level in out:
        for name in out[level]:
            print(level, name, out[level][name]['n_runs'],
                  {s: (round(out[level][name][s]['sd']['tightening'], 2), round(out[level][name][s]['sd']['p'], 3))
                   for s in C.STATS})


if __name__ == '__main__':
    main()
