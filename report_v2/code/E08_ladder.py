"""E8. The model ladder on a held-out alpha (Fig. 6, Section 4.4).

Each model is fitted on the runs of two values of alpha and scored on the third, by the concordance
index with censoring. The ladder is: each statistic alone in a linear AFT; the linear AFT on all
three; the AFT with interactions; kernel ridge (fitted on the training runs that grok); XGBoost with
the AFT objective (5 seeds); and a GRU that reads the statistics at the 25 saved steps up to t*
(3 seeds). The reference is XGBoost given the settings N and eta lambda, which knows more than a
general network would report. models.py holds the settings.

Since 17 October 2026 the ladder also splits the settings reference into its two parts: XGBoost on
log N alone ('width reference') and on log10 eta lambda alone ('decay reference'), 5 seeds each. They
are not drawn. Section 4.4 compares them with the full reference in the text.

Since 17 October 2026 the figure shows only the rungs in SHOWN: each statistic alone, XGBoost on all
three statistics at t*, and the GRU on their history up to t*, with the settings reference as a line.
E08.json still keeps the scores of every rung of LADDER.

The method is that of model_fitting/3_kernel_only.ipynb, Parts 3 and 5.
"""
import numpy as np
import pandas as pd

import common as C
import models as M

SEEDS = {'xgboost': M.XGB_SEEDS, 'gru': M.GRU_SEEDS, 'validation_split': 'numpy default_rng(seed)'}
LADDER = ['A only', 'S only', 'log R only', 'linear AFT', 'AFT with interactions', 'kernel ridge', 'XGBoost', 'GRU on history',
          'settings reference', 'width reference', 'decay reference']
PARTS = {'settings reference': [0, 1], 'width reference': [0], 'decay reference': [1]}
SHOWN = {'A only': '$A_t$ alone', 'S only': '$S_t$ alone', 'log R only': '$\\log R_t$ alone',
         'XGBoost': 'all three at $t^\\ast$ (trees)', 'GRU on history': 'all three up to $t^\\ast$ (recurrent)'}


def main():
    data = M.load_data()
    hist = M.history(data)
    alphas = sorted(data.alpha.unique())
    log_T = np.log(data['T'].to_numpy())
    folds = {a: ((data.alpha != a).to_numpy(), (data.alpha == a).to_numpy()) for a in alphas}
    scores = {name: {} for name in LADDER}
    seed_scores = {'XGBoost': {}, 'GRU on history': {}, 'settings reference': {}, 'width reference': {},
                   'decay reference': {}}
    chosen = {}

    det = {n: np.zeros(len(data)) for n in LADDER[:6]}
    for a, (train, test) in folds.items():
        for name, inter in (('linear AFT', False), ('AFT with interactions', True)):
            X = M.design(data, inter, ref=data[train])
            det[name][test] = M.fit_aft(data[train], X[train], M.AFT_PENALIZER_CV).predict_median(X[test]).to_numpy().ravel()
        X = M.design(data, False, ref=data[train])
        for stat, name in zip(M.COVS, ('A only', 'S only', 'log R only')):
            det[name][test] = M.fit_aft(data[train], X.loc[train, [stat]], M.AFT_PENALIZER_CV).predict_median(
                X.loc[test, [stat]]).to_numpy().ravel()
        g = train & (data.grokked == 1).to_numpy()
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV, GroupKFold
        krr = GridSearchCV(KernelRidge(kernel='rbf'), M.KRR_GRID, cv=GroupKFold(5))
        krr.fit(X[g], log_T[g], groups=data.loc[g, 'cell'])
        det['kernel ridge'][test] = np.exp(krr.predict(X[test]))
    for name, p in det.items():
        scores[name] = {f'{a:g}': M.concordance(data, p, folds[a][1]) for a in alphas}

    Xs = M.settings_matrix(data)
    for seed in M.XGB_SEEDS:
        for name in ('XGBoost', *PARTS):
            p = np.zeros(len(data))
            for a, (train, test) in folds.items():
                X = M.design(data, False, ref=data[train]).to_numpy() if name == 'XGBoost' else Xs[:, PARTS[name]]
                booster, best = M.tuned_xgb(data, X, train, seed)
                import xgboost as xgb
                p[test] = booster.predict(xgb.DMatrix(X[test]))
                if seed == 0:
                    chosen[f'{name}, alpha {a:g} held out'] = {
                        'max_depth': best[1]['max_depth'], 'eta': best[1]['eta'],
                        'min_child_weight': best[1]['min_child_weight'], 'rounds': best[2],
                        'validation_concordance': best[0]}
            seed_scores[name][seed] = {f'{a:g}': M.concordance(data, p, folds[a][1]) for a in alphas}
        print(f'xgboost seed {seed} done', flush=True)
    epochs = {}
    for seed in M.GRU_SEEDS:
        p = np.zeros(len(data))
        for a, (train, test) in folds.items():
            z = (hist - hist[train].mean(axis=(0, 1))) / hist[train].std(axis=(0, 1))
            predict, epochs[f'{seed}_{a:g}'] = M.train_gru(data, z, train, seed)
            p[test] = predict(z[test])
        seed_scores['GRU on history'][seed] = {f'{a:g}': M.concordance(data, p, folds[a][1]) for a in alphas}
        print(f'gru seed {seed} done', flush=True)
    for name, by_seed in seed_scores.items():
        df = pd.DataFrame(by_seed).T
        scores[name] = df.mean().to_dict()
        scores[name + ' seed range'] = [float(df.mean(axis=1).min()), float(df.mean(axis=1).max())]

    summary = {name: {'folds': scores[name], 'mean': float(np.mean(list(scores[name].values())))} for name in LADDER}
    for name in seed_scores:
        summary[name]['seed_range'] = scores[name + ' seed range']
    out = {'ladder': summary, 'seed_scores': seed_scores, 'xgboost_chosen_seed0': chosen, 'gru_epochs': epochs,
           'n_runs': len(data), 'grokked_by_alpha': data.groupby('alpha').grokked.sum().to_dict()}
    C.write_json('E08', out, SEEDS)
    for name in LADDER:
        s = summary[name]
        print(f"{name:24s} mean {s['mean']:.3f} folds " + ' '.join(f'{v:.3f}' for v in s['folds'].values())
              + (f"  seeds {s['seed_range'][0]:.3f}-{s['seed_range'][1]:.3f}" if 'seed_range' in s else ''))
    plot(summary)


def plot(summary):
    plt = C.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.6, 2.1), constrained_layout=True)
    colour = {'A only': C.STAT_COLOUR['A'], 'S only': C.STAT_COLOUR['S'], 'log R only': C.STAT_COLOUR['R']}
    names = list(SHOWN)
    for i, name in enumerate(names):
        s = summary[name]
        ax.barh(i, s['mean'] - 0.5, left=0.5, height=0.62, color=colour.get(name, '#8a8986'), zorder=2)
        ax.scatter(list(s['folds'].values()), [i] * len(s['folds']), s=14, facecolor='white', edgecolor=C.TEXT,
                   lw=0.8, zorder=4)
        if 'seed_range' in s:
            ax.hlines(i, *s['seed_range'], color=C.TEXT, lw=2.2, zorder=3)
        ax.text(0.905, i, f"{s['mean']:.2f}", va='center', ha='right', fontsize=6.5, color=C.TEXT)
    ref = summary['settings reference']['mean']
    ax.axvline(ref, color=C.TEXT, ls='--', lw=0.9, zorder=1)
    ax.text(ref - 0.006, -0.85, f'given the settings, {ref:.2f}', ha='right', va='center', fontsize=6.5,
            color=C.TEXT)
    ax.axvline(0.5, color=C.MUTED, lw=0.6, ls=':')
    ax.set_yticks(range(len(names)), [SHOWN[n] for n in names])
    ax.set_xlim(0.4, 0.91)
    ax.set_ylim(len(names) - 0.4, -1.2)
    ax.set_xlabel('concordance on the held-out $\\alpha$')
    ax.grid(axis='y', visible=False)
    C.save_figure(fig, 'E8_ladder')


if __name__ == '__main__':
    main()
