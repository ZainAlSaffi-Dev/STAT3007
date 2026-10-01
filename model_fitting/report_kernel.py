"""Rebuild the kernel statistics of the report, Equations 5 and 6, from the saved weights.

The report measures the kernel of g = (1 / sqrt(p)) sum_c f_c, the sum of the logits of
Equation 2 divided by sqrt(p). It takes the statistics on the test pairs only. S_t is the
ratio of Frobenius norms, R_t is the uncentred kernel distance of Fort et al. (2020), D_t is
the relative change, and A_t is the centred alignment with Y Y^T. The dataset columns use
another kernel and another probe, so this script recomputes the four numbers at the 48 steps
whose weights were saved, for all 720 runs.

Run from model_fitting/:  ../.venv/bin/python report_kernel.py
It writes data/report_kernel.parquet with the columns run_id, step, S_t, R_t, D_t, A_t.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'repoduced-code'))
from RepoducedCode import make_modular_addition_dataset  # noqa: E402

P = 23
DATA = Path(__file__).resolve().parent / 'data'

# Every run used data seed 42, so every run has the same 53 test pairs.
_, (X_TEST, Y_TEST) = make_modular_addition_dataset(p=P, train_fraction=0.9, seed=42)
X_TEST, Y_TEST = X_TEST.double().numpy(), Y_TEST.double().numpy()


def sum_kernel(W1, W2, X):
    """Kernel of g = sum_c f_c / sqrt(p) for f = W2 relu(W1 x / sqrt(D)) / sqrt(N), in closed form.

    g has one output whose second-layer weight is u = W2.sum(0) / sqrt(p). Its gradient with
    respect to W2[c, i] is relu(h_i) / sqrt(N p) for every c, which sums to Z Z^T / N over c.
    Its gradient with respect to row i of W1 is u_i 1[h_i > 0] x / sqrt(D N).
    """
    D, N = X.shape[1], W1.shape[0]
    h = X @ W1.T / np.sqrt(D)
    Z, M = np.maximum(h, 0), (h > 0).astype(float)
    u = W2.sum(0) / np.sqrt(W2.shape[0])
    return Z @ Z.T / N + (X @ X.T) * ((M * u ** 2) @ M.T) / (D * N)


def statistics(K_t, K_0, Y):
    """S_t, R_t, D_t and A_t of report Equation 6 for one pair of kernels on the test pairs."""
    r = len(K_t)
    C = np.eye(r) - 1 / r
    n_t, n_0 = np.linalg.norm(K_t), np.linalg.norm(K_0)
    Kc, Gc = C @ K_t @ C, C @ Y @ Y.T @ C
    return {'S_t': n_t / n_0,
            'R_t': 1 - np.sum(K_t * K_0) / (n_t * n_0),
            'D_t': np.linalg.norm(K_t - K_0) / n_0,
            'A_t': np.sum(Kc * Gc) / (np.linalg.norm(Kc) * np.linalg.norm(Gc))}


def run_rows(run_id):
    w = np.load(DATA / 'runs' / f'{run_id}_weights.npz')
    W1, W2 = w['W1'].astype(float), w['W2'].astype(float)
    kernels = [sum_kernel(W1[k], W2[k], X_TEST) for k in range(len(w['steps']))]
    return [{'run_id': run_id, 'step': int(s), **statistics(K, kernels[0], Y_TEST)}
            for s, K in zip(w['steps'], kernels)]


def check():
    """Closed form against autograd on one checkpoint, and the identity of report Equation 7."""
    import torch
    w = np.load(DATA / 'runs' / 'ntk_N100_a1_wd0_s0_weights.npz')
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
    rows = pd.DataFrame(run_rows('ntk_N100_a1_wd0_s0'))
    gap = rows['D_t'] ** 2 - (rows['S_t'] ** 2 - 2 * rows['S_t'] * (1 - rows['R_t']) + 1)
    assert gap.abs().max() < 1e-9, 'Equation 7 does not hold'
    print('checks passed: closed form matches autograd, Equation 7 holds')


if __name__ == '__main__':
    check()
    run_ids = pd.read_parquet(DATA / 'runs.parquet', columns=['run_id'])['run_id']
    out = pd.DataFrame([row for run_id in run_ids for row in run_rows(run_id)])
    out.to_parquet(DATA / 'report_kernel.parquet', index=False)
    print(out.shape, 'rows written to data/report_kernel.parquet')
