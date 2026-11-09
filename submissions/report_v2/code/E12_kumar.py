"""E12. The baseline against Kumar et al. (2024), arXiv:2310.06110v3 (Appendix B).

The settings of Kumar et al. below were read from the v3 PDF on 11 October 2026: Appendix 8.3 (pages
13 and 14) for the task and training, Section 3 (page 3) for the loss, and Appendix 8.1 (page 13)
for the centred predictor and the learning rate eta_0 / alpha^2. Their Figure 2 (page 4) shows one run
for each alpha. Its event steps are read by eye from the plot, so they are approximate; the paper
gives no numbers for them. Kumar et al. also sweep alpha = 1.5, which is not in our grid.

The table sets each setting beside ours and gives our events by alpha at N = 100, eta lambda = 0
(median and range over 5 seeds, from E01).
"""
import common as C

SEEDS = {}

# (setting, Kumar et al. v3, ours, where in Kumar et al.)
SETTINGS = [
    ('Task', '$a+b \\bmod p$, $p=23$', '$a+b \\bmod p$, $p=23$', 'App. 8.3'),
    ('Encoding', 'two one-hot codes, $D=2p$', 'two one-hot codes, $D=2p$', 'App. 8.3'),
    ('Output', '$p$ outputs, one-hot target', '$p$ outputs, one-hot target', 'App. 8.3'),
    ('Training pairs', '90\\% of $p^2$', '90\\% of $p^2$ (476 of 529)', 'App. 8.3'),
    ('Width', '$N=100$', '$N=100$ (grid: 50 to 1600)', 'App. 8.3'),
    ('Network', 'one hidden layer', 'one hidden layer, no biases, rectifier', 'App. 8.3'),
    ('Parameterisation', 'not stated', 'NTK, weights from $\\mathcal{N}(0,1)$', ''),
    ('Loss', 'mean squared error', 'mean squared error', 'Sec. 3'),
    ('Optimiser', 'vanilla full-batch GD, no weight decay', 'full-batch GD; $\\eta\\lambda=0$ at the baseline', 'App. 8.3'),
    ('Predictor', '$\\alpha[f(x;\\theta)-f(x;\\theta_0)]$', 'the same', 'App. 8.1'),
    ('Learning rate', '$\\eta=100$; $\\eta=\\eta_0/\\alpha^2$', '$\\eta=100/\\alpha^2$', 'App. 8.1, 8.3'),
    ('$\\alpha$', '$\\{0.5,1,1.5,2\\}$', '$\\{0.5,1,2\\}$', 'Fig. 2(c)'),
    ('Budget', '100\\,000 epochs (Fig. 2(c))', '200\\,000 steps', 'Fig. 2(c)'),
    ('Seeds', 'one run shown', '5 per cell', 'Fig. 2'),
]
# Read by eye from Figure 2(a) and 2(c) of v3: the step at which each curve reaches about 100%.
FIGURE_READING = {
    0.5: {'train_100': '$\\approx$4\\,000', 'test_100': '$\\approx$5\\,000'},
    1.0: {'train_100': '$\\approx$20\\,000 to 25\\,000', 'test_100': '$\\approx$25\\,000 to 27\\,000'},
    1.5: {'train_100': '$\\approx$65\\,000', 'test_100': '$\\approx$70\\,000'},
    2.0: {'train_100': '$\\approx$90\\,000', 'test_100': 'about 25\\% at 100\\,000'},
}


def k(x):
    return f'{x:,.0f}'.replace(',', '\\,')


def main():
    n5 = C.read_json('E01')['N5']
    rows = ['\\begin{tabular}{L{2.7cm}L{4.2cm}L{4.5cm}L{1.9cm}}', '\\toprule',
            'Setting & \\citet{kumarGrokkingTransitionLazy2024} & This report & Where \\\\', '\\midrule']
    rows += [f'{a} & {b} & {c} & {d} \\\\' for a, b, c, d in SETTINGS]
    rows += ['\\bottomrule', '\\end{tabular}']
    C.write_table('E12_settings', '\n'.join(rows))

    ev = ['\\begin{tabular}{lllll}', '\\toprule',
          '& \\multicolumn{2}{c}{\\citet{kumarGrokkingTransitionLazy2024}, read from Fig.~2} & '
          '\\multicolumn{2}{c}{This report, median [range] over 5 seeds} \\\\',
          '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
          '$\\alpha$ & Train 100\\% & Test 100\\% & Memorisation & Grokking (0.95) \\\\', '\\midrule']
    for a in (0.5, 1.0, 1.5, 2.0):
        fr = FIGURE_READING[a]
        if a == 1.5:
            ours = ('not in the grid', '')
        else:
            v = n5[f'{a:g}']
            mem = f"{k(v['mem_median'])} [{k(v['mem_min'])}, {k(v['mem_max'])}]"
            grok = (f"{k(v['grok95_median'])} [{k(v['grok95_min'])}, {k(v['grok95_max'])}], {v['grokked']}/5"
                    if v['grokked'] else f"0/5 by 200\\,000")
            ours = (mem, grok)
        ev.append(f"{a:g} & {fr['train_100']} & {fr['test_100']} & {ours[0]} & {ours[1]} \\\\")
    ev += ['\\bottomrule', '\\end{tabular}']
    C.write_table('E12_events', '\n'.join(ev))
    C.write_json('E12', {'settings': SETTINGS, 'figure_reading': FIGURE_READING, 'ours': n5,
                         'source': 'arXiv:2310.06110v3, App. 8.1 and 8.3, Sec. 3, Fig. 2; read 11 Oct 2026'}, SEEDS)


if __name__ == '__main__':
    main()
