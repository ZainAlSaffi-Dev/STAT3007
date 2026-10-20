"""Numpy versions of the kernel quantities, for scenes that draw the kernel itself.

The animations environment has no torch, so it cannot import ntk_lib. The
functions here repeat the few definitions a scene needs.

sum_kernel and statistics copy ntk_lib.sum_kernel and ntk_lib.report_statistics,
the kernel and the statistics of the report. extract_deck_data.py checks that
the copies agree with ntk_lib, and that they give the report's S_t, R_t and A_t
from saved weights.

centre matches ntk_lib.centre_kernel. The aim is the extra number defined in
repoduced-code/term_dependence.py, for the explainer scenes. Read the docstring
there for the identity it comes from.
"""

import numpy as np


def one_hot_pairs(p):
    """Return the inputs of all p^2 pairs (a, b), and the arrays a and b.

    Row a * p + b holds the one-hot code of a followed by the one-hot code of b,
    so an input has length 2p. This is the order of make_modular_addition_dataset
    in repoduced-code/RepoducedCode.py before it splits the pairs.
    """
    a, b = np.divmod(np.arange(p * p), p)
    X = np.zeros((p * p, 2 * p))
    X[np.arange(p * p), a] = 1.0
    X[np.arange(p * p), p + b] = 1.0
    return X, a, b


def sum_kernel(W1, W2, X):
    """The tangent kernel of the summed logits over sqrt(p), in closed form.

    This copies ntk_lib.sum_kernel, the kernel of golden report Equation 5, for the
    network f = W2 relu(W1 x / sqrt(D)) / sqrt(N) without biases. The kernel of the
    scaled network alpha [f - f_0] is alpha^2 times this one, so the statistics do
    not depend on alpha.
    """
    D, N = X.shape[1], W1.shape[0]
    h = X @ W1.T / np.sqrt(D)
    Z, M = np.maximum(h, 0), (h > 0).astype(float)
    u = W2.sum(0) / np.sqrt(W2.shape[0])
    return Z @ Z.T / N + (X @ X.T) * ((M * u ** 2) @ M.T) / (D * N)


def statistics(K_t, K_0, Y=None):
    """S_t, R_t and D_t of a kernel against the kernel at step 0, and A_t if Y is given.

    This copies ntk_lib.report_statistics, golden report Equation 6 and report_v2
    Equation eq:statistics. Y holds the one-hot targets of the pairs, one row each.
    """
    n_t, n_0 = np.linalg.norm(K_t), np.linalg.norm(K_0)
    out = {"S_t": n_t / n_0,
           "R_t": 1 - np.sum(K_t * K_0) / (n_t * n_0),
           "D_t": np.linalg.norm(K_t - K_0) / n_0}
    if Y is not None:
        Kc, Gc = centre(K_t), centre(Y @ Y.T)
        out["A_t"] = np.sum(Kc * Gc) / (np.linalg.norm(Kc) * np.linalg.norm(Gc))
    return out


def centre(K):
    """Return H K H with H = I - (1/n) 1 1^T, Cortes et al. (2012), Lemma 1."""
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    return H @ K @ H


def unit(M):
    """Return M divided by its Frobenius norm."""
    return M / np.linalg.norm(M)


def target_gram(labels, p):
    """Return the centred label Gram matrix H Y Y^T H for integer labels."""
    Y = np.eye(p)[labels]
    return centre(Y @ Y.T)


def aim(R, A, A_0):
    """The aim gamma_t, from the recorded rotation R_t and alignment A_t.

    It solves A_t = A_0 (1 - R_t) + gamma_t sqrt(1 - A_0^2) sqrt(R_t (2 - R_t))
    for gamma_t. It is undefined at R_t = 0, where the kernel has not turned.
    """
    R, A = np.asarray(R, float), np.asarray(A, float)
    out = np.full_like(R, np.nan)
    moved = R > 0
    out[moved] = (A[moved] - A_0 * (1 - R[moved])) / np.sqrt(R[moved] * (2 - R[moved]) * (1 - A_0 ** 2))
    return out


class Torus:
    """Averages a probe kernel over pairs of probe points with the same offsets.

    A probe point is a pair (a, b). Two points (a, b) and (a', b') have the
    offsets (a - a') mod p and (b - b') mod p. The average over all pairs of
    points with the same offsets is a p by p image. Row i holds the offset
    a - a' = i - p // 2 and column j holds b - b' = j - p // 2, so zero offset
    sits in the centre. The label kernel is one exactly on the line where the
    two offsets add to zero mod p, since that is where a + b = a' + b'.
    """

    def __init__(self, probe_a, probe_b, p):
        da = (probe_a[:, None] - probe_a[None, :]) % p
        db = (probe_b[:, None] - probe_b[None, :]) % p
        # Shift so that offset zero lands in row and column p // 2.
        da = (da + p // 2) % p
        db = (db + p // 2) % p
        self.p = p
        self.index = (da * p + db).ravel()
        self.count = np.bincount(self.index, minlength=p * p)
        if self.count.min() == 0:
            raise ValueError("Some offset pair has no probe pairs, so its average is undefined.")

    def __call__(self, M):
        total = np.bincount(self.index, weights=np.asarray(M, float).ravel(), minlength=self.p ** 2)
        return (total / self.count).reshape(self.p, self.p)
