"""How the scale, rotation and alignment terms depend on each other.

The question came from the Tier 1 notebook, where the three terms looked as if
they might be linear combinations of one another. This script separates what
holds by definition from what holds only in the saved runs.

What holds by definition. All three terms are computed on the centred kernel.
Write k_t for the centred kernel divided by its Frobenius norm and g for the
centred target Gram matrix divided by its norm. The scale term depends only on
the norm of the centred kernel. The rotation and alignment terms depend only on
k_t. So the scale term cannot be recovered from the other two, and neither of
them can be recovered from it.

Rotation and alignment are tied by an exact identity. Split k_t into its part
along k_0 and a remainder v_t that is orthogonal to k_0. The first part has
length 1 - R_t, so v_t has length sqrt(R_t (2 - R_t)). Taking the inner product
with g gives

    A_t = A_0 (1 - R_t) + gamma_t sqrt(1 - A_0^2) sqrt(R_t (2 - R_t)),

where gamma_t is the cosine between v_t and the part of g that is orthogonal to
k_0. We call gamma_t the aim. It lies between -1 and 1. It is 1 when the
kernel turns straight towards the target and 0 when it turns in a direction
the target does not care about. The identity is our own algebra, in the same
way as Equation 1 of the proposal. It is not taken from a paper. Given A_0, the
three numbers (S_t, R_t, gamma_t) carry the same information as
(S_t, R_t, A_t), and none of the three is fixed by the other two.

What the runs show. The script prints three things.

1. A check of the identity on the dense runs from kernel_snapshots.py. The
   aim is computed once from the saved kernels and once from the recorded
   R_t and A_t through the identity.
2. For every Tier 1 grid run, the share of the variance of S_t along the run
   that a straight line in R_t explains, grouped by decay. A value near 1
   means that scale and rotation moved in lockstep in that run, so the run
   alone cannot separate them.
3. The three terms and the aim at the generalisation event of every run that
   has one. The event is test accuracy 1, the level the Tier 1 notebook uses.

ntk_lib.py is on the sam branch, as for kernel_snapshots.py.

    python term_dependence.py
"""

import json
from pathlib import Path

import numpy as np

import ntk_lib as L

HERE = Path(__file__).resolve().parent
DENSE = HERE / "results"
# The grid runs sit next to ntk_lib, which is this folder once the merge is done.
GRID = Path(L.__file__).resolve().parent / "results"
TEST_ACC_LEVEL = 1.0


def aim(R, A, A_0):
    """The aim gamma_t from the identity in the module docstring. It is undefined where R_t is zero."""
    R, A = np.asarray(R, float), np.asarray(A, float)
    out = np.full_like(R, np.nan)
    moved = R > 0
    out[moved] = (A[moved] - A_0 * (1 - R[moved])) / np.sqrt(R[moved] * (2 - R[moved]) * (1 - A_0 ** 2))
    return out


def centre(K):
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    return H @ K @ H


def unit(M):
    return M / np.linalg.norm(M)


def aim_from_kernels(name):
    """The aim at every saved checkpoint, computed directly from the saved kernels."""
    data = np.load(DENSE / f"{name}_kernels.npz")
    p = json.loads((DENSE / f"{name}.json").read_text())["config"]["p"]
    labels = (data["probe_a"] + data["probe_b"]) % p
    Y = np.eye(p)[labels]
    g = unit(centre(Y @ Y.T))
    k_0 = unit(centre(data["K"][0].astype(float)))
    g_perp = g - np.sum(g * k_0) * k_0
    out = [np.nan]
    for K in data["K"][1:]:
        k_t = unit(centre(K.astype(float)))
        v = k_t - np.sum(k_t * k_0) * k_0
        out.append(np.sum(unit(v) * unit(g_perp)))
    return data["steps"], np.array(out)


def check_identity():
    print("1. The identity on the dense runs")
    for path in sorted(DENSE.glob("dense_*_kernels.npz")):
        name = path.name[: -len("_kernels.npz")]
        h = json.loads((DENSE / f"{name}.json").read_text())["history"]
        steps, direct = aim_from_kernels(name)
        index = {s: i for i, s in enumerate(h["step"])}
        rows = [index[s] for s in steps]
        from_history = aim(np.take(h["R_c"], rows), np.take(h["A_t"], rows), h["A_t"][0])
        diff = np.nanmax(np.abs(direct - from_history))
        print(f"  {name}: largest difference {diff:.1e} over {len(steps) - 1} checkpoints")


def grid_runs():
    runs = {}
    for path in sorted(GRID.glob("ntk_N100_*.json")):
        if "probe" in path.name:
            continue
        runs[path.stem] = json.loads(path.read_text())
    return runs


def r_squared(x, y):
    """Share of the variance of y that a least-squares line in x explains."""
    X = np.c_[np.ones(len(x)), x]
    residual = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - residual.var() / y.var()


def lockstep(runs):
    print("\n2. Scale against rotation along each run: R squared of S_t on R_t")
    by_decay = {}
    for run in runs.values():
        c, h = run["config"], run["history"]
        by_decay.setdefault(c["eta_kappa"], []).append(
            (c["alpha"], c["seed"], r_squared(np.asarray(h["R_c"]), np.asarray(h["S_c"]))))
    print(f"  {'eta kappa':>9} {'runs':>4} {'min':>6} {'median':>6} {'max':>6}")
    for ek in sorted(by_decay):
        v = np.array([r for _, _, r in by_decay[ek]])
        print(f"  {ek:>9g} {len(v):>4} {v.min():6.3f} {np.median(v):6.3f} {v.max():6.3f}")


def at_event(runs):
    print(f"\n3. The terms at the generalisation event, test accuracy {TEST_ACC_LEVEL:g}")
    rows = []
    for run in runs.values():
        c, h = run["config"], run["history"]
        t = L.accuracy_crossings(run, TEST_ACC_LEVEL)["t_test"]
        if t is None:
            continue
        i = int(np.where(np.asarray(h["step"]) <= t)[0][-1])
        gamma = aim(h["R_c"], h["A_t"], h["A_t"][0])
        rows.append((c["alpha"], c["eta_kappa"], h["S_c"][i], h["R_c"][i], h["A_t"][i], gamma[i]))
    v = np.array(rows)
    print(f"  {len(v)} of {len(runs)} runs have an event.")
    print(f"  {'term':>9} {'mean':>7} {'min':>7} {'max':>7} {'max/min':>7} {'sd/mean':>7}")
    for k, name in zip(range(2, 6), ["S_t", "R_t", "A_t", "gamma_t"]):
        x = v[:, k]
        print(f"  {name:>9} {x.mean():7.4f} {x.min():7.4f} {x.max():7.4f} {x.max() / x.min():7.2f} "
              f"{x.std() / abs(x.mean()):7.3f}")
    corr = np.corrcoef(v[:, 2:].T)
    print("  Correlations over runs, in the order S_t, R_t, A_t, gamma_t:")
    for row in corr:
        print("   " + " ".join(f"{x:+.2f}" for x in row))
    print("  Mean aim at the event, by alpha and decay:")
    for a in sorted(set(v[:, 0])):
        cells = [(ek, v[(v[:, 0] == a) & (v[:, 1] == ek), 5]) for ek in sorted(set(v[v[:, 0] == a, 1]))]
        print("   alpha " + f"{a:g}: " + ", ".join(f"{ek:g} -> {g.mean():.3f} ({len(g)})" for ek, g in cells))


if __name__ == "__main__":
    check_identity()
    runs = grid_runs()
    lockstep(runs)
    at_event(runs)
