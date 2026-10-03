"""Rebuild the kernel statistics of the report, Equations 5 and 6, from the saved weights.

The report measures the kernel of g = (1 / sqrt(p)) sum_c f_c, the sum of the logits of
Equation 2 divided by sqrt(p). It takes the statistics on the test pairs only. S_t is the
ratio of Frobenius norms, R_t is the uncentred kernel distance of Fort et al. (2020), D_t is
the relative change, and A_t is the centred alignment with Y Y^T. Each run has its own split,
so each run uses its own test pairs. This script computes the four numbers at the 48 steps
whose weights were saved, for all 720 runs. The checkpoint columns S_sum, R_sum, D_sum and
A_sum hold the same numbers at every checkpoint.

Run from model_fitting/:  ../.venv/bin/python report_kernel.py
It writes data/report_kernel.parquet with the columns run_id, step, S_t, R_t, D_t, A_t.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'repoduced-code'))
from RepoducedCode import make_modular_addition_dataset  # noqa: E402
from ntk_lib import report_statistics as statistics, sum_kernel  # noqa: E402

P = 23
DATA = Path(__file__).resolve().parent / 'data'


def test_pairs(data_seed):
    """The test pairs and their targets for one split, as float64 numpy arrays."""
    _, (X, Y) = make_modular_addition_dataset(p=P, train_fraction=0.9, seed=data_seed)
    return X.double().numpy(), Y.double().numpy()


def run_rows(run_id, data_seed):
    X_test, Y_test = test_pairs(data_seed)
    w = np.load(DATA / 'runs' / f'{run_id}_weights.npz')
    W1, W2 = w['W1'].astype(float), w['W2'].astype(float)
    kernels = [sum_kernel(W1[k], W2[k], X_test) for k in range(len(w['steps']))]
    return [{'run_id': run_id, 'step': int(s), **statistics(K, kernels[0], Y_test)}
            for s, K in zip(w['steps'], kernels)]


def check():
    """Closed form against autograd on one checkpoint, and the identity of report Equation 7."""
    import torch
    X_TEST, _ = test_pairs(43)
    w = np.load(DATA / 'runs' / 'ntk_N100_a1_wd0_s1_weights.npz')
    W1 = torch.tensor(w['W1'][20], dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor(w['W2'][20], dtype=torch.float64, requires_grad=True)
    X = torch.tensor(X_TEST)
    grads = []
    for x in X:
        g = (torch.relu(x @ W1.T / np.sqrt(2 * P)) @ W2.T / np.sqrt(100)).sum() / np.sqrt(P)
        grads.append(torch.cat([t.flatten() for t in torch.autograd.grad(g, (W1, W2))]))
    J = torch.stack(grads)
    K_auto = (J @ J.T).numpy()
    K_closed = sum_kernel(w['W1'][20].astype(float), w['W2'][20].astype(float), X_TEST)
    assert np.abs(K_closed - K_auto).max() <= 1e-6 * np.abs(K_auto).max(), 'closed form differs from autograd'
    rows = pd.DataFrame(run_rows('ntk_N100_a1_wd0_s1', 43))
    gap = rows['D_t'] ** 2 - (rows['S_t'] ** 2 - 2 * rows['S_t'] * (1 - rows['R_t']) + 1)
    assert gap.abs().max() < 1e-9, 'Equation 7 does not hold'
    print('checks passed: closed form matches autograd, Equation 7 holds')


if __name__ == '__main__':
    check()
    runs = pd.read_parquet(DATA / 'runs.parquet', columns=['run_id', 'data_seed'])
    out = pd.DataFrame([row for r in runs.itertuples() for row in run_rows(r.run_id, r.data_seed)])
    out.to_parquet(DATA / 'report_kernel.parquet', index=False)
    print(out.shape, 'rows written to data/report_kernel.parquet')
