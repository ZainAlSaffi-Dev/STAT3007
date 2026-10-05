"""
A small check of whether the Tier 2 results of 26 September 2026 still hold on the dataset in data/.

The dataset (see tier2_reproduction.md) gives each seed its own split, checks events at 307
checkpoints, and runs every width for 200,000 steps. The Tier 2 runs of 26 September used data
seed 42 for every seed, checked events every 500 steps, and ran 100,000 steps at N <= 100 and
300,000 steps at N >= 400. This script compares the two in four parts.

A. Seed 0 has data seed 42 and model seed 0 in both, so its training should be identical. The
   training fields are compared at the steps the two share.
B. The old definitions on the new data: memorisation and generalisation at accuracy 1.0, the gap
   t_grok = t_test - t_train, runs that never memorise left out, and the same censored log fit
   with the same cell-grouped bootstrap as tier2_width_decay.ipynb. This isolates the effect of
   the new splits, the new budget and the new checkpoint grid.
C. The definitions of the golden report of 4 October 2026: memorisation at training accuracy
   0.99, grokking at test accuracy 0.80, 0.90 and 0.95 counted from step 0, every run kept and
   censored at the budget, and a seed bootstrap that draws each seed with all its runs.
D. The kernel at the grokking event by width: R_t of the sum kernel at the event checkpoint, and
   A_t of the tangent kernel of all logits (golden report Section 3.3) at the last saved weight
   step at or before the event.

Run from repoduced-code/ with the tara-env interpreter:

    /opt/miniconda3/envs/tara-env/bin/python tier2_consistency.py

It prints every result and writes them to results/tier2_consistency.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import ntk_lib as L
from RepoducedCode import make_modular_addition_dataset

DATA = Path(__file__).resolve().parent.parent / "data"
OLD = Path(__file__).resolve().parent / "results"
OUT = OLD / "tier2_consistency.json"

P = 23
OLD_WIDTHS = [50, 100, 400, 1600]
OLD_DECAYS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4]
NEW_WIDTHS = [50, 100, 200, 400, 800, 1600]
N_BOOT = 1000
results = {}


def name(N, ek, seed):
    return L.cell_name(N, 1.0, ek, seed)


def load_new(N, ek, seed):
    return json.loads((DATA / "runs" / f"{name(N, ek, seed)}.json").read_text())


# =====================================================================
# A. Seed 0 against the old Tier 2 runs
# =====================================================================
print("A. Seed 0, old Tier 2 runs against the dataset, at the steps both checked")
fields = ["train_loss", "test_loss", "train_acc", "test_acc", "weight_norm"]
worst = {f: 0.0 for f in fields}
rows_a = []
for N in OLD_WIDTHS:
    for ek in OLD_DECAYS:
        old = json.loads((OLD / f"{name(N, ek, 0)}.json").read_text())["history"]
        new = load_new(N, ek, 0)["history"]
        o_idx = {s: i for i, s in enumerate(old["step"])}
        shared = [(o_idx[s], j) for j, s in enumerate(new["step"]) if s in o_idx]
        for f in fields:
            d = max(abs(old[f][i] - new[f][j]) / max(abs(old[f][i]), 1e-12) for i, j in shared)
            worst[f] = max(worst[f], d)
        g_old = L.accuracy_crossings({"history": old}, 1.0)
        g_new = L.accuracy_crossings({"history": new}, 1.0)
        rows_a.append(dict(N=N, ek=ek, shared_steps=len(shared), t_train_old=g_old["t_train"],
                           t_train_new=g_new["t_train"], t_test_old=g_old["t_test"], t_test_new=g_new["t_test"]))
print(f"    largest relative difference over 20 runs: " + ", ".join(f"{f} {v:.1e}" for f, v in worst.items()))
for r in rows_a:
    print(f"    N {r['N']:>4} eta lambda {r['ek']:<6g} memorised old {r['t_train_old']} new {r['t_train_new']}, "
          f"generalised old {r['t_test_old']} new {r['t_test_new']}")
results["A"] = dict(largest_relative_difference=worst, events=rows_a)


# =====================================================================
# B. The old definitions on the new data
# =====================================================================
def old_records(widths, decays, cut_narrow=False):
    """One record per run with the old events: train and test accuracy 1.0, gap from memorisation.

    With cut_narrow, runs at N <= 100 are cut at 100,000 steps, the old budget at those widths.
    """
    out = []
    for N in widths:
        for ek in decays:
            for s in range(5):
                h = load_new(N, ek, s)["history"]
                if cut_narrow and N <= 100:
                    keep = [i for i, st in enumerate(h["step"]) if st <= 100_000]
                    h = {k: [h[k][i] for i in keep] for k in ("step", "train_acc", "test_acc")}
                g = L.accuracy_crossings({"history": h}, 1.0)
                out.append(dict(N=N, ek=ek, seed=s, cell=f"{N}_{ek:g}", memorised=g["t_train"] is not None,
                                censored=g["censored"], t_train=g["t_train"], t_grok=g["t_grok"],
                                t_test=g["t_test"] if g["t_test"] is not None else g["last_step"]))
    return out


def usable(recs):
    return [r for r in recs if r["memorised"] and r["t_grok"] is not None and r["t_grok"] > 0]


def fit_log(rs, widths, response="t_grok", interaction=False, n_boot=N_BOOT):
    """The fit of tier2_width_decay.ipynb, section 5: censored log fit, bootstrap within cells."""
    rs = [r for r in rs if r["ek"] > 0]
    lnN, lnE = np.log([r["N"] for r in rs]), np.log([r["ek"] for r in rs])
    X = np.c_[np.ones(len(rs)), lnN, lnE]
    if interaction:
        X = np.c_[X, (lnN - np.mean(np.log(widths))) * (lnE - np.mean(np.log([d for d in OLD_DECAYS if d > 0])))]
    f = L.censored_log_fit(X, [r[response] for r in rs], [r["censored"] for r in rs], n_boot=n_boot, seed=0,
                           groups=[r["cell"] for r in rs])
    return dict(coef=f["coef"].tolist(), lo=f["lo"].tolist(), hi=f["hi"].tolist(), sigma=float(np.exp(f["log_sigma"])),
                n=len(rs), n_cens=int(sum(r["censored"] for r in rs)))


def ci(f, k):
    return f"{f['coef'][k]:+.3f} [{f['lo'][k]:+.3f}, {f['hi'][k]:+.3f}]"


def slopes(rs, widths):
    """Within-stratum censored slopes, as in section 5 of the old notebook."""
    out = {}
    for ek in OLD_DECAYS:
        sub = [r for r in rs if r["ek"] == ek and r["N"] in widths]
        x = np.log([r["N"] for r in sub])
        if len(sub) >= 4 and len(set(x)) >= 2 and not all(r["censored"] for r in sub):
            f = L.censored_log_fit(np.c_[np.ones(len(sub)), x], [r["t_grok"] for r in sub],
                                   [r["censored"] for r in sub], n_boot=400, seed=0, groups=[r["cell"] for r in sub])
            out[f"a at {ek:g}"] = (float(f["coef"][1]), float(f["lo"][1]), float(f["hi"][1]))
    for N in widths:
        sub = [r for r in rs if r["N"] == N and r["ek"] > 0]
        if len(sub) >= 4 and not all(r["censored"] for r in sub):
            f = L.censored_log_fit(np.c_[np.ones(len(sub)), np.log([r["ek"] for r in sub])],
                                   [r["t_grok"] for r in sub], [r["censored"] for r in sub], n_boot=400, seed=0,
                                   groups=[r["cell"] for r in sub])
            out[f"b at N {N}"] = (float(f["coef"][1]), float(f["lo"][1]), float(f["hi"][1]))
    return out


print("\nB. Old definitions on the new data (accuracy 1.0, gap from memorisation, non-memorising runs left out)")
results["B"] = {}
for label, cut in [("four old widths, budget 200,000 at every width", False),
                   ("four old widths, N <= 100 cut at 100,000 as before", True)]:
    recs = old_records(OLD_WIDTHS, OLD_DECAYS, cut_narrow=cut)
    fs = usable(recs)
    main = fit_log(fs, OLD_WIDTHS)
    inter = fit_log(fs, OLD_WIDTHS, interaction=True)
    test = fit_log(fs, OLD_WIDTHS, response="t_test")
    strata = slopes(fs, OLD_WIDTHS)
    never = [(r["N"], r["ek"]) for r in recs if not r["memorised"]]
    print(f"  {label}")
    print(f"    {len(fs)} usable runs, {len(never)} never memorised {sorted(set(never))}, "
          f"{sum(r['censored'] for r in fs)} censored")
    print(f"    main effects: a {ci(main, 1)}, b {ci(main, 2)}, sigma {main['sigma']:.3f} (runs {main['n']}, censored {main['n_cens']})")
    print(f"    interaction:  a {ci(inter, 1)}, b {ci(inter, 2)}, product {ci(inter, 3)}, sigma {inter['sigma']:.3f}")
    print(f"    on t_test:    a {ci(test, 1)}, b {ci(test, 2)}")
    print("    within strata: " + ", ".join(f"{k} {v[0]:+.2f}" for k, v in strata.items()))
    results["B"][label] = dict(main=main, interaction=inter, t_test=test, strata=strata, never_memorised=never)


# =====================================================================
# C. The golden report's definitions
# =====================================================================
runs = pd.read_csv(DATA / "report_runs.csv")
runs = runs[runs.alpha == 1].copy()


def mle(X, y, c):
    """Log-normal AFT by maximum likelihood with right censoring (golden report Eq. 10 and 11)."""
    def nll(p):
        mu, sigma = X @ p[:-1], np.exp(p[-1])
        return -(norm.logpdf(y[~c], mu[~c], sigma).sum() + norm.logsf(y[c], mu[c], sigma).sum())
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    return minimize(nll, np.r_[beta0, 0.0], method="BFGS").x


def aft_seed_boot(df, cols, level, n_boot=N_BOOT, seed=0):
    """AFT on the given covariate columns, with a percentile interval from resampling whole seeds."""
    X = np.c_[np.ones(len(df)), df[cols].to_numpy(float)]
    y = np.log(df[f"t_{level}"].to_numpy(float))
    c = df[f"censored_{level}"].to_numpy(bool)
    est = mle(X, y, c)
    rng = np.random.default_rng(seed)
    by_seed = [np.flatnonzero(df.seed.to_numpy() == s) for s in range(5)]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([by_seed[s] for s in rng.integers(0, 5, 5)])
        if np.linalg.matrix_rank(X[idx]) == X.shape[1]:
            boots.append(mle(X[idx], y[idx], c[idx]))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    return dict(names=["intercept"] + cols, coef=est[:-1].tolist(), lo=lo[:-1].tolist(), hi=hi[:-1].tolist(),
                sigma=float(np.exp(est[-1])), n=len(df), n_cens=int(c.sum()))


runs["lnN"] = np.log(runs.width)
runs["lnE"] = np.log(runs.eta_kappa.where(runs.eta_kappa > 0))
print("\nC. Golden report definitions (events from step 0, every run kept, censored at 200,000)")
results["C"] = {}
specs = {
    "crossed, 6 widths, 1e-5 to 1e-4": (runs.eta_kappa.isin([1e-5, 3e-5, 1e-4]), NEW_WIDTHS, [1e-5, 3e-5, 1e-4]),
    "crossed, 6 widths, 1e-5 to 3e-4": (runs.eta_kappa.isin([1e-5, 3e-5, 1e-4, 3e-4]), NEW_WIDTHS, [1e-5, 3e-5, 1e-4, 3e-4]),
    "old grid, 4 widths, 1e-5 to 3e-4": (runs.eta_kappa.isin([1e-5, 3e-5, 1e-4, 3e-4]) & runs.width.isin(OLD_WIDTHS),
                                         OLD_WIDTHS, [1e-5, 3e-5, 1e-4, 3e-4]),
    # The old rule left out the runs that never memorise, which are all at N = 1600 and 3e-4. This variant
    # leaves that cell out, so that the effect of the censoring rule can be read against the line above.
    "old grid, 4 widths, 1e-5 to 3e-4, without N = 1600 at 3e-4": (
        runs.eta_kappa.isin([1e-5, 3e-5, 1e-4, 3e-4]) & runs.width.isin(OLD_WIDTHS)
        & ~((runs.width == 1600) & (runs.eta_kappa == 3e-4)), OLD_WIDTHS, [1e-5, 3e-5, 1e-4, 3e-4]),
}
for label, (mask, widths, decays) in specs.items():
    df = runs[mask].copy()
    df["prod"] = (df.lnN - np.mean(np.log(widths))) * (df.lnE - np.mean(np.log(decays)))
    results["C"][label] = {}
    print(f"  {label}")
    for level in ["grok80", "grok90", "grok95", "mem"]:
        m = aft_seed_boot(df, ["lnN", "lnE"], level)
        i = aft_seed_boot(df, ["lnN", "lnE", "prod"], level)
        results["C"][label][level] = dict(main=m, interaction=i)
        print(f"    {level:>6}: a {ci(m, 1)}, b {ci(m, 2)}, sigma {m['sigma']:.2f}, censored {m['n_cens']}/{m['n']};"
              f" with product: a {ci(i, 1)}, b {ci(i, 2)}, product {ci(i, 3)}")
    # Five seeds are few clusters for a seed bootstrap, so the main fit at level 0.95 is also given the
    # interval of the old notebook, which resamples runs within each cell.
    cells = (df.width.astype(str) + "_" + df.eta_kappa.astype(str)).tolist()
    old_style = L.censored_log_fit(np.c_[np.ones(len(df)), df[["lnN", "lnE"]].to_numpy(float)],
                                   df.t_grok95.to_numpy(float), df.censored_grok95.to_numpy(bool),
                                   n_boot=N_BOOT, seed=0, groups=cells)
    old_style = dict(coef=old_style["coef"].tolist(), lo=old_style["lo"].tolist(), hi=old_style["hi"].tolist())
    results["C"][label]["grok95_cell_bootstrap"] = old_style
    print(f"    grok95 main fit with the cell bootstrap of the old notebook: a {ci(old_style, 1)}, b {ci(old_style, 2)}")

zero = runs[(runs.eta_kappa == 0) & runs.width.isin([50, 100])].copy()
zero["N50"] = (zero.width == 50).astype(float)
results["C"]["width arm at no decay, N = 50 against N = 100"] = {}
print("  width arm at no decay, N = 50 against N = 100 (the only widths with events)")
for level in ["grok80", "grok90", "grok95", "mem"]:
    f = aft_seed_boot(zero, ["N50"], level)
    results["C"]["width arm at no decay, N = 50 against N = 100"][level] = f
    print(f"    {level:>6}: N = 50 indicator {ci(f, 1)}")
wide_zero = runs[(runs.eta_kappa == 0) & (runs.width >= 200)]
print(f"  at no decay and N >= 200: {int((~wide_zero.censored_grok80).sum())} of {len(wide_zero)} runs reach level 0.80")

cv = runs[~runs.censored_grok95].groupby(["width", "eta_kappa"]).t_grok95.agg(lambda t: np.std(np.log(t)))
cv_by_width = cv[cv.index.get_level_values(1).isin([1e-5, 3e-5, 1e-4])].groupby(level=0).median()
print("  seed spread, median over 1e-5 to 1e-4 of the standard deviation of log t_grok95 within a cell: "
      + ", ".join(f"N {N} {v:.2f}" for N, v in cv_by_width.items()))
results["C"]["seed_spread_by_width"] = {str(k): float(v) for k, v in cv_by_width.items()}


# =====================================================================
# D. The kernel at the grokking event, by width
# =====================================================================
def logit_kernel(W1, W2, X):
    """Tangent kernel of all p logits on the rows of X, an (n p) x (n p) matrix, pair-major."""
    D, N = X.shape[1], W1.shape[0]; p = W2.shape[0]; n = len(X)
    h = X @ W1.T / np.sqrt(D); Z, M = np.maximum(h, 0), (h > 0).astype(float)
    G = (M[:, None, :] * W2[None, :, :]).reshape(n * p, N)
    XX = np.repeat(np.repeat(X @ X.T, p, 0), p, 1)
    return np.kron(Z @ Z.T / N, np.eye(p)) + (G @ G.T) * XX / (D * N)


def cka(K, y):
    r = len(K); C = np.eye(r) - 1 / r
    Kc, G = C @ K @ C, C @ np.outer(y, y) @ C
    return np.sum(Kc * G) / (np.linalg.norm(Kc) * np.linalg.norm(G))


print("\nD. Kernel at the grokking event (level 0.95), by width, all decays")
tests = {}
rows_d = []
for r in runs.itertuples():
    if r.data_seed not in tests:
        _, (Xv, Yv) = make_modular_addition_dataset(p=P, train_fraction=0.9, seed=int(r.data_seed))
        tests[r.data_seed] = (Xv.double().numpy(), Yv.double().numpy().reshape(-1))
    Xv, yv = tests[r.data_seed]
    w = np.load(DATA / "runs" / f"{r.run_id}_weights.npz")
    A0 = cka(logit_kernel(w["W1"][0].astype(float), w["W2"][0].astype(float), Xv), yv)
    row = dict(width=r.width, eta_kappa=r.eta_kappa, seed=r.seed, A0=A0, A_ev=np.nan, gap=np.nan,
               R_ev=r.R_t_at_grok95, S_ev=r.S_t_at_grok95)
    if not r.censored_grok95:
        k = int(np.searchsorted(w["steps"], r.t_grok95, side="right") - 1)
        row["A_ev"] = cka(logit_kernel(w["W1"][k].astype(float), w["W2"][k].astype(float), Xv), yv)
        row["gap"] = (r.t_grok95 - w["steps"][k]) / r.t_grok95
    rows_d.append(row)
d = pd.DataFrame(rows_d)
ev = d[d.A_ev.notna()]


def spread(v):
    v = np.asarray(v, float)
    return v.max() / v.min(), v.std() / v.mean()


summary = d.groupby("width").agg(A0=("A0", "mean")).join(
    ev.groupby("width").agg(A_ev=("A_ev", "mean"), R_ev=("R_ev", "mean"), S_ev=("S_ev", "mean"), runs=("A_ev", "size")))
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
for label, sub in [("all widths", ev), ("old four widths", ev[ev.width.isin(OLD_WIDTHS)])]:
    a, r_ = spread(sub.A_ev), spread(sub.R_ev)
    print(f"  {label}: A at the event max/min {a[0]:.2f}, spread {100 * a[1]:.1f}%; "
          f"R at the event max/min {r_[0]:.1f}, spread {100 * r_[1]:.1f}%")
print(f"  A is read at a saved weight step a median {100 * ev.gap.median():.0f}% "
      f"(largest {100 * ev.gap.max():.0f}%) of the event step before the event")
results["D"] = dict(by_width=summary.reset_index().to_dict("records"),
                    A_spread_all=spread(ev.A_ev), R_spread_all=spread(ev.R_ev),
                    A_spread_old_widths=spread(ev[ev.width.isin(OLD_WIDTHS)].A_ev),
                    R_spread_old_widths=spread(ev[ev.width.isin(OLD_WIDTHS)].R_ev),
                    weight_step_gap_median=float(ev.gap.median()), weight_step_gap_max=float(ev.gap.max()))

OUT.write_text(json.dumps(results, indent=1, default=float))
print(f"\nwritten to {OUT}")
