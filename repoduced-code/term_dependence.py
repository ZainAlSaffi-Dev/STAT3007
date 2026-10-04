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
   A plane fit of the alignment on scale and rotation at the event follows,
   with a bootstrap interval over runs.

Part 2 prints three shares per run, as experiment E1 of docs/zain_tierx_plan.md
asks: scale on rotation, alignment on rotation, and alignment on the square
root of rotation.

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
    """Share of the variance of y that a least-squares line in x explains.

    It is NaN when x or y is constant up to float64 rounding, because a line
    fitted to rounding noise means nothing. The synthetic paths of the plan's
    control are like that. The variance in a trained run is many orders of
    magnitude larger.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.var(x) <= 1e-24 or np.var(y) <= 1e-24:
        return float("nan")
    X = np.c_[np.ones(len(x)), x]
    residual = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - residual.var() / y.var()


SHARES = [("S_t on R_t", "S_c", "R_c"), ("A_t on R_t", "A_t", "R_c"), ("A_t on sqrt(R_t)", "A_t", "sqrt R_c")]


def along_run(runs):
    """Return one row per run with the three shares of experiment E1 in the plan.

    Each share is the R squared of a least-squares line along the run: scale
    on rotation, alignment on rotation, and alignment on the square root of
    rotation. The square root is the form the identity in the module
    docstring gives for small R_t. R_t is clipped at zero first, because the
    float32 history records values of about -3e-6 at step 0.
    """
    rows = []
    for name, run in runs.items():
        c, h = run["config"], run["history"]
        series = {"S_c": np.asarray(h["S_c"]), "R_c": np.asarray(h["R_c"]), "A_t": np.asarray(h["A_t"]),
                  "sqrt R_c": np.sqrt(np.clip(h["R_c"], 0, None))}
        row = dict(name=name, alpha=c["alpha"], eta_kappa=c["eta_kappa"], seed=c["seed"])
        for label, y, x in SHARES:
            row[label] = r_squared(series[x], series[y])
        rows.append(row)
    return rows


def lockstep(runs):
    """Print the three shares of along_run as one table per share, by decay level, and return the rows."""
    rows = along_run(runs)
    for label, _, _ in SHARES:
        print(f"\n2. Along each run: R squared of {label}")
        print(f"  {'eta kappa':>9} {'runs':>4} {'min':>6} {'median':>6} {'max':>6}")
        for ek in sorted({r["eta_kappa"] for r in rows}):
            v = np.array([r[label] for r in rows if r["eta_kappa"] == ek])
            print(f"  {ek:>9g} {len(v):>4} {v.min():6.3f} {np.median(v):6.3f} {v.max():6.3f}")
    return rows


def event_rows(runs):
    """Return one row per run that generalises, with the terms and the aim at the generalisation event.

    The event is the first checkpoint at test accuracy TEST_ACC_LEVEL, as in
    the Tier 1 notebook. A_0 is the run's own alignment at step 0.
    """
    rows = []
    for name, run in runs.items():
        c, h = run["config"], run["history"]
        t = L.accuracy_crossings(run, TEST_ACC_LEVEL)["t_test"]
        if t is None:
            continue
        i = int(np.where(np.asarray(h["step"]) <= t)[0][-1])
        gamma = aim(h["R_c"], h["A_t"], h["A_t"][0])
        rows.append(dict(name=name, alpha=c["alpha"], eta_kappa=c["eta_kappa"], seed=c["seed"], step=t,
                         S=h["S_c"][i], R=h["R_c"][i], A=h["A_t"][i], gamma=gamma[i], A_0=h["A_t"][0]))
    return rows


def plane_fit(y, X, n_boot=2000, seed=0):
    """R squared of a least-squares fit of y on the columns of X with an intercept, and a bootstrap interval.

    The bootstrap resamples runs with replacement. Returns the R squared,
    the 2.5 and 97.5 percentiles over resamples, the coefficients with the
    intercept first, and the resampled values.
    """
    y, X = np.asarray(y, float), np.c_[np.ones(len(y)), np.asarray(X, float)]

    def fit(yy, XX):
        beta = np.linalg.lstsq(XX, yy, rcond=None)[0]
        return 1 - np.var(yy - XX @ beta) / np.var(yy), beta

    r2, beta = fit(y, X)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if np.var(y[i]) > 0 and np.linalg.matrix_rank(X[i]) == X.shape[1]:
            boot.append(fit(y[i], X[i])[0])
    boot = np.array(boot)
    return dict(r2=r2, lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), coef=beta, boot=boot)


def at_event(runs):
    """Print the terms at the generalisation event, their correlations and the mean aim, and return the rows."""
    print(f"\n3. The terms at the generalisation event, test accuracy {TEST_ACC_LEVEL:g}")
    rows = event_rows(runs)
    v = np.array([[r["alpha"], r["eta_kappa"], r["S"], r["R"], r["A"], r["gamma"]] for r in rows])
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
    fit = plane_fit(v[:, 4], v[:, 2:4])
    print(f"  Plane fit of A_t on S_t and R_t with an intercept: R squared {fit['r2']:.3f}, "
          f"95 percent bootstrap interval {fit['lo']:.3f} to {fit['hi']:.3f}")
    return rows


if __name__ == "__main__":
    check_identity()
    runs = grid_runs()
    lockstep(runs)
    at_event(runs)
