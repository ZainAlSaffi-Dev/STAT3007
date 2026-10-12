"""Shared code for the report_v2 figures and numbers.

Every item script imports this module. It holds the paths, the loaders of the slim data
extract, the event definitions, the cell grouping, the colours and the writers of the outputs.

The events follow the golden report, Section 3.5. The memorisation step of a run is the first
checkpoint at which its training accuracy reaches 0.99. Its grokking step at level l is the first
checkpoint at which its test accuracy reaches l, counted from step 0. A run that does not reach a
level within the budget of 200,000 steps is kept and censored at the budget. These events are
already in report_runs.csv, and extract.py checks them against the checkpoints.

The kernel statistics are S, R and A of the tangent kernel of the summed logits over sqrt(p), on
the 53 test pairs of each run (report Eq. 3). In the checkpoints they are S_sum, R_sum and A_sum.
The columns S_t, R_t, A_t, S_c and R_c of the full checkpoints.parquet belong to an older probe
kernel and are never read here.
"""
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
FULL_DATA = Path(os.environ.get('STAT3007_DATA', REPO / 'data'))
SLIM = CODE / 'data'
OUT = CODE / 'out'
OVERLEAF = CODE.parent / 'overleaf'

BUDGET = 200_000
T_STAR = 2_430
LEVELS = {'grok80': 0.80, 'grok90': 0.90, 'grok95': 0.95}
MAIN_LEVEL = 'grok95'
ALPHAS = [0.5, 1.0, 2.0]
WIDTHS = [50, 100, 200, 400, 800, 1600]
DECAYS = [0.0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
STATS = ['S', 'R', 'A']
CHECKPOINT_STATS = {'S': 'S_sum', 'R': 'R_sum', 'A': 'A_sum'}
PHASE_THRESHOLD = 0.9

PACKAGES = ['numpy', 'pandas', 'pyarrow', 'scipy', 'matplotlib', 'lifelines', 'scikit-learn', 'xgboost',
            'torch']


# ---------------------------------------------------------------------------------------------
# Loaders. They read the slim extract in code/data only, so a reader needs nothing else.

def load_runs():
    """One row per run (720), with the settings, the events and the statistics at each event."""
    runs = pd.read_csv(SLIM / 'report_runs.csv')
    return runs.sort_values(['alpha', 'width', 'eta_kappa', 'seed']).reset_index(drop=True)


def load_checkpoints(columns=None):
    """The 307 checkpoints of every run, sorted by run and step."""
    df = pd.read_parquet(SLIM / 'checkpoints_slim.parquet', columns=columns)
    return df.sort_values(['run_id', 'step']).reset_index(drop=True)


def load_kernel():
    """S, R, D and A at the 48 steps whose weights were saved."""
    return pd.read_parquet(SLIM / 'report_kernel.parquet').sort_values(['run_id', 'step']).reset_index(drop=True)


def load_manifest():
    return json.loads((SLIM / 'manifest.json').read_text())


def at_tstar(runs, kernel=None):
    """The runs joined to S, R and A at t* = 2,430, with log R."""
    kernel = load_kernel() if kernel is None else kernel
    k = kernel[kernel.step == T_STAR][['run_id', 'S_t', 'R_t', 'A_t']]
    out = runs.merge(k, on='run_id', how='left', validate='one_to_one')
    out['logR_t'] = np.log(out['R_t'])
    return out


def pivot(checkpoints, column):
    """A run by step matrix of one checkpoint column. Every run shares one step grid."""
    wide = checkpoints.pivot(index='run_id', columns='step', values=column)
    return wide


def value_at_step(wide, run_ids, steps):
    """The value of each run at a step: the last checkpoint at or before it. wide comes from pivot()."""
    grid = wide.columns.to_numpy()
    idx = np.searchsorted(grid, np.asarray(steps), side='right') - 1
    rows = wide.index.get_indexer(run_ids)
    if (rows < 0).any():
        raise KeyError('a run is missing from the checkpoint table')
    return wide.to_numpy()[rows, idx]


# ---------------------------------------------------------------------------------------------
# Cells and outcomes.

def cell_key(df):
    return list(zip(df['alpha'], df['width'], df['eta_kappa']))


def cell_label(df):
    return df['alpha'].map('{:g}'.format) + '_' + df['width'].astype(str) + '_' + df['eta_kappa'].map('{:g}'.format)


def phase(runs, checkpoints):
    """The phase of each run at its last step, after the golden report Section 3.5 with threshold 0.9.

    G: training and test accuracy both end above 0.9. M: only training accuracy does. F: training
    accuracy rose above 0.9 at some checkpoint and ends below it. N: none of these. The rule is the
    one of grid/src/deeplearning/training.py on branch feat/grid.
    """
    peak = checkpoints.groupby('run_id')['train_acc'].max()
    th = PHASE_THRESHOLD
    fitted = runs['train_acc_final'] > th
    crossed = runs['run_id'].map(peak) > th
    return pd.Series(np.select([fitted & (runs['test_acc_final'] > th), fitted, crossed], ['G', 'M', 'F'], 'N'),
                     index=runs.index)


def spread(values, kind='sd'):
    """The spread of a statistic across runs: the standard deviation (ddof 1), or the CV for audit."""
    v = np.asarray(values, dtype=float)
    sd = np.std(v, ddof=1, axis=-1)
    return sd if kind == 'sd' else sd / np.mean(v, axis=-1)


# ---------------------------------------------------------------------------------------------
# Colours. One hue per statistic in every figure, from the validated reference palette
# (S blue, R orange, A aqua). An ordered setting inside a statistic's panel is drawn as light to
# dark steps of that statistic's hue. Width and the grokking time use one violet ramp. Accuracy
# uses greys so that it never reads as a statistic.

STAT_COLOUR = {'S': '#2a78d6', 'R': '#eb6834', 'A': '#1baf7a'}
STAT_RAMP = {
    'S': ['#b7d3f6', '#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#184f95', '#104281', '#0d366b'],
    'R': ['#f8cdb8', '#f5b394', '#f08c5f', '#eb6834', '#c9501f', '#a33f17', '#7f3011', '#5e230c'],
    'A': ['#b9e8d3', '#8fd9b9', '#4fc496', '#1baf7a', '#168f64', '#11704e', '#0c553b', '#083d2a'],
}
TRAIN_COLOUR = '#8a8986'
TEST_COLOUR = '#0b0b0b'
TEXT = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'
VIOLET_CMAP = 'Purples'
STAT_CMAP = {'S': 'Blues', 'R': 'Oranges', 'A': 'Greens'}
STAT_NAME = {'S': 'scale $S_t$', 'R': 'rotation $R_t$', 'A': 'alignment $A_t$'}


def ramp(stat, n):
    """n ordered steps of a statistic's hue, light to dark, evenly spread over its ramp."""
    r = STAT_RAMP[stat]
    return [r[int(round(i))] for i in np.linspace(0, len(r) - 1, n)]


def width_colours():
    import matplotlib
    cmap = matplotlib.colormaps[VIOLET_CMAP]
    return {w: cmap(x) for w, x in zip(WIDTHS, np.linspace(0.4, 1.0, len(WIDTHS)))}


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': MUTED, 'axes.labelcolor': TEXT, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'lines.linewidth': 1.5, 'lines.markersize': 4, 'figure.dpi': 150, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'cm', 'font.family': 'serif',
        'savefig.dpi': 200,
        # A fixed creation date keeps the PDFs byte-identical between runs.
        'pdf.compression': 6,
    })
    return plt


# ---------------------------------------------------------------------------------------------
# Outputs. Every script writes only into code/out. make_all.py copies the figures and tables into
# the Overleaf folder.

def versions():
    v = {'python': sys.version.split()[0], 'platform': platform.platform()}
    for p in PACKAGES:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            v[p] = None
    return v


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def write_json(item, payload, seeds=None):
    """Write the numbers of an item to out/numbers/<item>.json, with the seeds and the versions."""
    path = OUT / 'numbers' / f'{item}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {'item': item, 'seeds': seeds or {}, 'versions': versions(), 'results': payload}
    path.write_text(json.dumps(_clean(body), indent=1, sort_keys=False) + '\n')
    return path


def read_json(item):
    return json.loads((OUT / 'numbers' / f'{item}.json').read_text())['results']


def save_figure(fig, name):
    """Write a figure as out/figures/<name>.pdf, with a PNG preview beside it."""
    d = OUT / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    meta = {'CreationDate': None, 'Creator': 'report_v2/code', 'Producer': None}
    fig.savefig(d / f'{name}.pdf', metadata=meta)
    fig.savefig(d / f'{name}.png', metadata={'Software': None})
    return d / f'{name}.pdf'


def write_table(name, tex):
    """Write a LaTeX table body to out/tables/<name>.tex."""
    d = OUT / 'tables'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{name}.tex').write_text(tex.rstrip() + '\n')
    return d / f'{name}.tex'


def fmt_decay(x):
    """eta lambda for a table or a tick label, such as 0, 3e-6 or 1e-4, in LaTeX."""
    if x == 0:
        return '\\ensuremath{0}'
    m, e = f'{x:.0e}'.split('e')
    e = int(e)
    return f'\\ensuremath{{10^{{{e}}}}}' if m == '1' else f'\\ensuremath{{{m}\\times10^{{{e}}}}}'


def decay_tick(x):
    if x == 0:
        return '0'
    m, e = f'{x:.0e}'.split('e')
    return f'$10^{{{int(e)}}}$' if m == '1' else f'${m}{{\\cdot}}10^{{{int(e)}}}$'


def cell_outcomes(runs, checkpoints, level=MAIN_LEVEL):
    """One row per cell: the seeds that grok at the level, the median grokking step over the seeds
    that grok (kept only when at least 3 of 5 grok), the phase letters of the seeds, and a letter for
    a cell in which no seed groks. That letter is the most common phase among its seeds. A G in a
    cell with no grokking seed means that the test accuracy ends above 0.9 without reaching the level.
    """
    r = runs.copy()
    r['phase'] = phase(r, checkpoints)
    rows = []
    for (alpha, width, decay), cell in r.groupby(['alpha', 'width', 'eta_kappa']):
        cell = cell.sort_values('seed')
        hit = ~cell[f'censored_{level}']
        k = int(hit.sum())
        letters = ''.join(cell.phase)
        counts = cell.phase.value_counts()
        top = counts.index[0] if k == 0 else ''
        rows.append({'alpha': alpha, 'width': int(width), 'eta_kappa': decay, 'grokked': k, 'seeds': len(cell),
                     'median_T': float(np.median(cell.loc[hit, f't_{level}'])) if k >= 3 else np.nan,
                     'phases': letters, 'letter': top})
    return pd.DataFrame(rows)


def between_share(df, column, by):
    """The share of the spread of a column that lies between groups: 1 - SS_within / SS_total."""
    v = df[column]
    total = ((v - v.mean()) ** 2).sum()
    within = ((v - df.groupby(by)[column].transform('mean')) ** 2).sum()
    return float(1 - within / total)
