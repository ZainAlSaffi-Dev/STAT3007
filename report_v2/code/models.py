"""The prediction models of Tier 3 (E8, E9, E10), ported from model_fitting/3_kernel_only.ipynb and
model_fitting/4_nonlinear_exploration.ipynb. Their settings are the ones listed in Appendix D.

The data are the 630 runs with eta lambda at most 1e-3, with S_t, log R_t and A_t read at
t* = 2,430. T is the grokking step at level 0.95, or the budget of 200,000 for a censored run.
"""
import copy
import itertools

import xgboost as xgb  # import XGBoost before PyTorch, or their two copies of OpenMP clash
import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter
from lifelines.utils import concordance_index
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

KERNEL = ['A_t', 'S_t', 'R_t']        # as stored
COVS = ['A_t', 'S_t', 'logR_t']       # what the models use
PAIRS = [('A_t', 'S_t'), ('A_t', 'logR_t'), ('S_t', 'logR_t')]
AFT_PENALIZER_CV = 0.01
AFT_PENALIZER_FULL = 1e-4
KRR_GRID = {'alpha': [0.01, 0.1, 1], 'gamma': [0.1, 0.5, 2]}
XGB_BASE = {'objective': 'survival:aft', 'aft_loss_distribution': 'normal', 'aft_loss_distribution_scale': 1.0,
            'subsample': 0.8, 'eval_metric': 'aft-nloglik', 'nthread': 1}
XGB_GRID = [{'max_depth': d, 'eta': e, 'min_child_weight': w}
            for d, e, w in itertools.product([1, 2, 3, 4], [0.03, 0.1], [1, 5, 10])]
XGB_ROUNDS, XGB_EARLY = 1000, 50
VALIDATION_SHARE = 0.2
GRU_HIDDEN, GRU_LR, GRU_WD, GRU_EPOCHS, GRU_EVERY = 16, 1e-2, 1e-4, 800, 10
XGB_SEEDS = list(range(5))
GRU_SEEDS = list(range(3))


def load_data():
    runs = C.at_tstar(C.load_runs())
    data = runs[runs.eta_kappa <= 1e-3].reset_index(drop=True)
    data['grokked'] = (~data.censored_grok95).astype(int)
    data['T'] = data.t_grok95.astype(float)
    data['cell'] = C.cell_label(data)
    assert len(data) == 630 and data['T'].min() > C.T_STAR
    return data


def history(data, kernel=None):
    """Runs by the 25 saved steps up to t* by (A, S, R), with R raw since it is 0 at step 0."""
    kernel = C.load_kernel() if kernel is None else kernel
    before = kernel[kernel.step <= C.T_STAR].pivot(index='run_id', columns='step', values=KERNEL)
    h = np.stack([before[k].loc[data.run_id].to_numpy() for k in KERNEL], axis=-1)
    assert h.shape[1] == 25
    return h


def design(df, interactions=True, ref=None, cols=COVS):
    """z-scores against ref, with the products of the z-scores when interactions is True."""
    ref = df if ref is None else ref
    z = (df[cols] - ref[cols].mean()) / ref[cols].std()
    X = z.copy()
    if interactions:
        for a, b in PAIRS:
            if a in cols and b in cols:
                X[f'{a}*{b}'] = z[a] * z[b]
    return X


def fit_aft(df, X, penalizer):
    m = LogNormalAFTFitter(penalizer=penalizer)
    return m.fit(X.assign(T=df['T'].to_numpy(), grokked=df['grokked'].to_numpy()), 'T', 'grokked')


def kernel_ridge():
    """RBF kernel ridge on standardised inputs and target, with the penalty and width chosen by grouped CV."""
    grid = {f'kernelridge__{k}': v for k, v in KRR_GRID.items()}
    return TransformedTargetRegressor(
        GridSearchCV(make_pipeline(StandardScaler(), KernelRidge(kernel='rbf')), grid, cv=GroupKFold(5)),
        transformer=StandardScaler())


def validation_split(data, train, seed):
    cells = data.loc[train, 'cell'].unique()
    held = np.random.default_rng(seed).choice(cells, round(VALIDATION_SHARE * len(cells)), replace=False)
    val = train & data['cell'].isin(held).to_numpy()
    return train & ~val, val


def _aft_matrix(data, X, rows):
    m = xgb.DMatrix(X[rows])
    T = data['T'].to_numpy()
    m.set_float_info('label_lower_bound', T[rows])
    m.set_float_info('label_upper_bound', np.where(data['grokked'] == 1, T, np.inf)[rows])
    return m


def tuned_xgb(data, X, train, seed):
    """Try every setting on 20% of the training cells, keep the best by concordance, refit on all."""
    fit, val = validation_split(data, train, seed)
    best = (-1, None, None)
    T, E = data['T'].to_numpy(), data['grokked'].to_numpy()
    for setting in XGB_GRID:
        params = {**XGB_BASE, **setting, 'seed': seed}
        booster = xgb.train(params, _aft_matrix(data, X, fit), XGB_ROUNDS,
                            evals=[(_aft_matrix(data, X, val), 'validation')],
                            early_stopping_rounds=XGB_EARLY, verbose_eval=False)
        rounds = booster.best_iteration + 1
        c = concordance_index(T[val], booster.predict(_aft_matrix(data, X, val), iteration_range=(0, rounds)), E[val])
        if c > best[0]:
            best = (c, params, rounds)
    return xgb.train(best[1], _aft_matrix(data, X, train), best[2]), best


def settings_matrix(data):
    return np.column_stack([np.log(data['width']),
                            np.log10(data['eta_kappa'].where(data['eta_kappa'] > 0, 1e-6))])


def train_gru(data, z, train, seed):
    import torch
    torch.set_num_threads(1)

    def aft_loss(mu, log_sigma, y, grokked):
        s = (y - mu) / log_sigma.exp()
        normal = torch.distributions.Normal(0.0, 1.0)
        return -(grokked * (normal.log_prob(s) - log_sigma) + (1 - grokked) * torch.log(normal.cdf(-s) + 1e-12)).mean()

    class GRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(3, GRU_HIDDEN, batch_first=True)
            self.head = torch.nn.Linear(GRU_HIDDEN, 1)
            self.log_sigma = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return self.head(self.gru(x)[1][-1]).squeeze(-1)

    fit, val = validation_split(data, train, seed)
    log_T = np.log(data['T'].to_numpy())
    y = (log_T - log_T[train].mean()) / log_T[train].std()
    Z, Y, E = (torch.tensor(np.asarray(v), dtype=torch.float32) for v in (z, y, data['grokked']))
    torch.manual_seed(seed)
    net = GRU()
    opt = torch.optim.Adam(net.parameters(), lr=GRU_LR, weight_decay=GRU_WD)
    best = (np.inf, None, 0)
    for epoch in range(1, GRU_EPOCHS + 1):
        opt.zero_grad()
        aft_loss(net(Z[fit]), net.log_sigma, Y[fit], E[fit]).backward()
        opt.step()
        if epoch % GRU_EVERY == 0:
            with torch.no_grad():
                loss = aft_loss(net(Z[val]), net.log_sigma, Y[val], E[val]).item()
            if loss < best[0]:
                best = (loss, copy.deepcopy(net.state_dict()), epoch)
    net.load_state_dict(best[1])

    def predict(x):
        with torch.no_grad():
            return net(torch.tensor(x, dtype=torch.float32)).numpy()
    return predict, best[2]


def concordance(data, pred, rows):
    return float(concordance_index(data.loc[rows, 'T'], pred[rows], data.loc[rows, 'grokked']))


def settings():
    """The settings as Appendix D reports them."""
    return {'aft_penalizer_cross_validation': AFT_PENALIZER_CV, 'aft_penalizer_full_data': AFT_PENALIZER_FULL,
            'kernel_ridge_grid': KRR_GRID, 'kernel_ridge_inner_cv': 'GroupKFold(5) by cell',
            'xgboost_base': XGB_BASE, 'xgboost_grid': {'max_depth': [1, 2, 3, 4], 'eta': [0.03, 0.1],
                                                       'min_child_weight': [1, 5, 10]},
            'xgboost_rounds': XGB_ROUNDS, 'xgboost_early_stopping': XGB_EARLY,
            'validation_share_of_training_cells': VALIDATION_SHARE, 'xgboost_seeds': XGB_SEEDS,
            'gru': {'hidden': GRU_HIDDEN, 'lr': GRU_LR, 'weight_decay': GRU_WD, 'max_epochs': GRU_EPOCHS,
                    'validate_every': GRU_EVERY, 'optimiser': 'Adam, full batch', 'loss': 'log-normal AFT',
                    'inputs': '25 saved steps up to t*, A, S and raw R, z-scored on the training fold'},
            'gru_seeds': GRU_SEEDS}
