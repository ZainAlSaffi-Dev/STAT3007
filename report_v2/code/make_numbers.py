"""Collect the numbers of every item into overleaf/numbers.tex as LaTeX macros.

Each macro is a \\newcommand whose name says where the number comes from, such as \\EthreeNhundredRratio
for the tightening ratio of R_t in the N = 100 group of E3. LaTeX macro names hold letters only, so
digits are spelt out. A comment beside each macro gives the JSON file and the key it came from.
main.tex reads the file with \\input{numbers}, so no number in the report is typed by hand.
"""
import re

import common as C

WORDS = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
         '8': 'eight', '9': 'nine'}
SPECIAL = {'50': 'fifty', '100': 'hundred', '200': 'twohundred', '400': 'fourhundred', '800': 'eighthundred',
           '1600': 'sixteenhundred', '0.5': 'half', '1': 'one', '2': 'two', '80': 'eighty', '90': 'ninety',
           '95': 'ninetyfive'}


def name(*parts):
    out = ''
    for p in parts:
        p = str(p)
        if p in SPECIAL:
            p = SPECIAL[p]
        p = re.sub(r'\d', lambda m: WORDS[m.group()], p)
        p = re.sub(r'[^A-Za-z]', '', p)
        out += p[:1].upper() + p[1:]
    return out


def steps(x):
    return f'{x:,.0f}'.replace(',', '\\,')


def signed(x, d=2):
    """A signed number that works in and out of math mode."""
    return f'\\ensuremath{{{x:+.{d}f}}}'


def pval(p, n=2000):
    """A permutation p; 0 of n permutations is written as below 1/n."""
    return f'\\ensuremath{{<{1 / n:.4f}}}' if p == 0 else f'{p:.3f}'


def sig(x, n=2):
    return f'{float(f"{x:.{n}g}"):g}'


class Macros:
    def __init__(self):
        self.lines = []
        self.names = set()

    def add(self, macro, value, source):
        assert macro not in self.names, macro
        self.names.add(macro)
        self.lines.append(f'\\newcommand{{\\{macro}}}{{{value}}}% {source}')


def main():
    M = Macros()
    e1 = C.read_json('E01')
    for a, v in e1['N5'].items():
        for k in ('mem_median', 'mem_min', 'mem_max', 'grok95_median', 'grok95_min', 'grok95_max'):
            if v[k] is not None:
                M.add(name('Nfive', 'a', a, k.replace('95', '').replace('_', ' ').title()), steps(v[k]),
                      f'E01.json N5.{a}.{k}')
        M.add(name('Nfive', 'a', a, 'Grokked'), v['grokked'], f'E01.json N5.{a}.grokked')
    for a, v in e1['N2'].items():
        for k in ('mem', 'grok95', 'final'):
            if v[k] is not None:
                M.add(name('Ntwo', 'a', a, k.replace('95', '')), f'{100 * v[k]:.0f}\\%', f'E01.json N2.{a}.{k}')
        M.add(name('Ntwo', 'a', a, 'Sfinal'), f'{v["S_final_median"]:.1f}', f'E01.json N2.{a}.S_final_median')
        M.add(name('Ntwo', 'a', a, 'Rfinal'), f'{v["R_final_median"]:.3f}', f'E01.json N2.{a}.R_final_median')

    e3 = C.read_json('E03')
    for g, gname in (('N100', 'Nhundred'), ('alpha1', 'Aone'), ('all', 'All')):
        G = e3['groups'][g]
        M.add(name('Ethree', gname, 'Runs'), G['n_runs'], f'E03.json groups.{g}.n_runs')
        for s in C.STATS:
            r = G[s]['sd']
            M.add(name('Ethree', gname, s, 'sd'), sig(r['at_grok']), f'E03.json groups.{g}.{s}.sd.at_grok')
            M.add(name('Ethree', gname, s, 'sdmem'), sig(r['at_mem']), f'E03.json groups.{g}.{s}.sd.at_mem')
            M.add(name('Ethree', gname, s, 'sdplacebo'), sig(r['placebo_median']),
                  f'E03.json groups.{g}.{s}.sd.placebo_median')
            M.add(name('Ethree', gname, s, 'ratio'), f'{r["tightening"]:.2f}', f'E03.json groups.{g}.{s}.sd.tightening')
            M.add(name('Ethree', gname, s, 'p'), pval(r['p']), f'E03.json groups.{g}.{s}.sd.p')
            M.add(name('Ethree', gname, s, 'cvratio'), f'{G[s]["cv"]["tightening"]:.2f}',
                  f'E03.json groups.{g}.{s}.cv.tightening (audit only)')
    n6 = e3['N6']
    for k, lab in (('A_step0_median', 'Azero'), ('A_grok_median', 'Agrok'), ('A_final_median', 'Afinal')):
        M.add(name('Nsix', lab), f'{n6[k]:.3f}', f'E03.json N6.{k}')
    for w, v in e3['N7'].items():
        M.add(name('Nseven', 'R', 'w', w), f'{v["R_median"]:.3f}', f'E03.json N7.{w}.R_median')
        M.add(name('Nseven', 'S', 'w', w), f'{v["S_median"]:.2f}', f'E03.json N7.{w}.S_median')
    n8 = e3['N8']
    for k in ('n_cells_no_grok', 'n_cells_R', 'n_cells_no_grok_memorised', 'n_cells_R_memorised',
              'n_cells_above_R_median_memorised'):
        M.add(name('Neight', k.replace('n_cells', '').replace('_', ' ').title()), n8[k], f'E03.json N8.{k}')

    e4 = C.read_json('E04')
    for w, d in e4['largest_decay_all_grok'].items():
        M.add(name('Efour', 'Edge', 'w', w), C.fmt_decay(d), f'E04.json largest_decay_all_grok.{w}')

    e6 = C.read_json('E06')
    for row, rname in (('main', 'Main'), ('with3e-4', 'Wide'), ('level80', 'Leighty'), ('level90', 'Lninety'),
                       ('mem', 'Mem')):
        R = e6['rows'][row]
        for k, kn in (('a', 'a'), ('b', 'b'), ('gamma', 'g')):
            M.add(name('Esix', rname, kn), signed(R['estimate'][k]), f'E06.json rows.{row}.estimate.{k}')
            M.add(name('Esix', rname, kn, 'lo'), signed(R['ci_cells'][k][0]), f'E06.json rows.{row}.ci_cells.{k}')
            M.add(name('Esix', rname, kn, 'hi'), signed(R['ci_cells'][k][1]), f'E06.json rows.{row}.ci_cells.{k}')
        M.add(name('Esix', rname, 's'), f'{R["estimate"]["sigma"]:.2f}', f'E06.json rows.{row}.estimate.sigma')
        M.add(name('Esix', rname, 'Within'), f'{R["within_cell_sd_median"]:.2f}', f'E06.json rows.{row}.within_cell_sd_median')
        M.add(name('Esix', rname, 'Misfit'), f'{R["cell_misfit_sd"]:.2f}', f'E06.json rows.{row}.cell_misfit_sd')
        M.add(name('Esix', rname, 'Runs'), R['n_runs'], f'E06.json rows.{row}.n_runs')
        M.add(name('Esix', rname, 'Cens'), R['n_censored'], f'E06.json rows.{row}.n_censored')
    for k, kn in (('a', 'a'), ('b', 'b'), ('gamma', 'g')):
        ci = e6['rows']['main']['ci_seeds'][k]
        M.add(name('Esix', 'Main', kn, 'seedlo'), signed(ci[0]), f'E06.json rows.main.ci_seeds.{k}')
        M.add(name('Esix', 'Main', kn, 'seedhi'), signed(ci[1]), f'E06.json rows.main.ci_seeds.{k}')

    e7 = C.read_json('E07')
    for col, cn in (('logR_t', 'logR'), ('S_t', 'S'), ('A_t', 'A')):
        for by, bn in (('alpha', 'alpha'), ('width', 'width'), ('eta_kappa', 'decay')):
            M.add(name('Nnine', cn, bn), f'{e7["N9_grokked"][col][by]:.2f}', f'E07.json N9_grokked.{col}.{by}')
            M.add(name('Nnine', 'All', cn, bn), f'{e7["N9_all"][col][by]:.2f}', f'E07.json N9_all.{col}.{by}')

    e8 = C.read_json('E08')['ladder']
    for model, mn in (('A only', 'Aonly'), ('S only', 'Sonly'), ('log R only', 'Ronly'), ('linear AFT', 'Linear'),
                      ('AFT with interactions', 'Inter'), ('kernel ridge', 'Krr'), ('XGBoost', 'Xgb'),
                      ('GRU on history', 'Gru'), ('settings reference', 'Ref'), ('width reference', 'Widthref'),
                      ('decay reference', 'Decayref')):
        v = e8[model]
        M.add(name('Eeight', mn), f'{v["mean"]:.2f}', f'E08.json ladder.{model}.mean')
        if 'seed_range' in v:
            M.add(name('Eeight', mn, 'lo'), f'{v["seed_range"][0]:.2f}', f'E08.json ladder.{model}.seed_range')
            M.add(name('Eeight', mn, 'hi'), f'{v["seed_range"][1]:.2f}', f'E08.json ladder.{model}.seed_range')

    e9 = C.read_json('E09')['gamma_SR']
    M.add('EnineGamma', signed(e9['estimate']), 'E09.json gamma_SR.estimate')
    M.add('EnineGammalo', signed(e9['ci'][0]), 'E09.json gamma_SR.ci')
    M.add('EnineGammahi', signed(e9['ci'][1]), 'E09.json gamma_SR.ci')

    e10 = C.read_json('E10')
    for sub, sn in (('S', 'S'), ('log R', 'R'), ('A', 'A'), ('S + log R', 'SR'), ('S + A', 'SA'), ('log R + A', 'RA'),
                    ('S + log R + A', 'SRA')):
        v = e10['subsets'][sub]
        M.add(name('Eten', sn, 'Rsq'), f'{v["R2"]:.2f}', f'E10.json subsets.{sub}.R2')
        M.add(name('Eten', sn, 'C'), f'{v["concordance"]:.2f}', f'E10.json subsets.{sub}.concordance')
    for q, qn in (('logT', 'logT'), ('S_t', 'S'), ('logR_t', 'logR'), ('A_t', 'A')):
        M.add(name('Eten', 'Between', qn), f'{e10["between_cells"][q]:.2f}', f'E10.json between_cells.{q}')
    M.add('EtenRuns', e10['n_runs'], 'E10.json n_runs')
    M.add('EtenCells', e10['n_cells'], 'E10.json n_cells')

    e13 = C.read_json('E13')['by_alpha']
    for al, v in e13.items():
        M.add(name('Ethirteen', 'a', al, 'AllGrok'), v['cells_all_grok'], f'E13.json by_alpha.{al}.cells_all_grok')
        M.add(name('Ethirteen', 'a', al, 'NoneGrok'), v['cells_none_grok'], f'E13.json by_alpha.{al}.cells_none_grok')

    e15 = C.read_json('E15')
    ep = sorted(e15['gru_epochs'].values())
    M.add('EfifteenGruEpochsMin', ep[0], 'E15.json gru_epochs (min)')
    M.add('EfifteenGruEpochsMax', ep[-1], 'E15.json gru_epochs (max)')

    e16 = C.read_json('E16')
    for level, ln in (('grok80', 'eighty'), ('grok90', 'ninety')):
        for g, gname in (('N100', 'Nhundred'), ('alpha1', 'Aone')):
            G = e16[level][g]
            M.add(name('Esixteen', ln, gname, 'Runs'), G['n_runs'], f'E16.json {level}.{g}.n_runs')
            for st in C.STATS:
                r = G[st]['sd']
                M.add(name('Esixteen', ln, gname, st, 'ratio'), f'{r["tightening"]:.2f}',
                      f'E16.json {level}.{g}.{st}.sd.tightening')
                M.add(name('Esixteen', ln, gname, st, 'p'), pval(r['p']), f'E16.json {level}.{g}.{st}.sd.p')

    n = C.read_json('N')
    M.add('NoneSummedHours', f'{n["N1"]["summed_run_hours"]:.1f}', 'N.json N1.summed_run_hours')
    M.add('NoneSummedSeconds', steps(n['N1']['summed_run_seconds']), 'N.json N1.summed_run_seconds')
    M.add('NoneElapsedHours', f'{n["N1"]["elapsed_hours"]:.1f}', 'N.json N1.elapsed_hours')
    M.add('NoneWorkers', n['N1']['workers'], 'N.json N1.workers')
    M.add('NthreeMem', f'{100 * n["N3"]["mem"]["max"]:.1f}\\%', 'N.json N3.mem.max')
    M.add('NthreeGrok', f'{100 * n["N3"]["grok95"]["max"]:.1f}\\%', 'N.json N3.grok95.max')
    M.add('NfourMin', f'{n["N4"]["min"]:.3f}', 'N.json N4.min')
    M.add('NfourMax', f'{n["N4"]["max"]:.3f}', 'N.json N4.max')

    path = C.OVERLEAF / 'numbers.tex'
    header = ['% Generated by report_v2/code/make_numbers.py from report_v2/code/out/numbers/*.json.',
              '% Do not edit by hand: rerun python make_all.py instead.']
    path.write_text('\n'.join(header + M.lines) + '\n')
    print(f'make_numbers: {len(M.lines)} macros in {path}')


if __name__ == '__main__':
    main()
