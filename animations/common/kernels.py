"""Numpy versions of the kernel quantities, for scenes that draw the kernel itself.

The animations environment has no torch, so it cannot import ntk_lib. The
functions here repeat the few definitions a scene needs, on the saved probe
kernels. They match ntk_lib.centre_kernel and the centred terms S_c, R_c and
A_t. The dense runs check this: the terms computed here agree with the
recorded history to about 1e-5, the precision of the saved float32 kernels.

The aim is the extra number defined in repoduced-code/term_dependence.py. Read
the docstring there for the identity it comes from.
"""

import numpy as np


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
