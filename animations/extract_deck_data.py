"""Write the data files that the deck clips read, into animations/data.

The animations environment has only Manim and numpy, so it cannot read parquet
or import torch. This script runs once with the environment of the repository
root and writes three numpy files that git keeps.

- n100_checkpoints.npz: the 120 runs at width 100 of the report extract,
  report_v2/code/data/checkpoints_slim.parquet, at all 307 checkpoints. It
  holds training and test accuracy and the report's statistics S_sum, R_sum
  and A_sum.
- split_seed42.npz: which of the 529 pairs are test pairs under data seed 42,
  the split of every seed 0 run.
- ntk_N100_a1_wd0_s0_weights.npz: the weights of the baseline run with seed 0
  at its 48 saved steps, copied unchanged from the full sweep of 3 October 2026.
  The full sweep is not in git. It is read from model_fitting/data, or from the
  folder that STAT3007_DATA names.

The checks at the end are what the clips rely on. The numpy kernel of
common/kernels.py must equal ntk_lib.sum_kernel. The copied weights must give
the report's S_t, R_t and A_t on the 53 test pairs at all 48 steps. The
committed seed 0 curves must equal the report extract.

Run from animations/:  ../.venv/bin/python extract_deck_data.py
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FULL = Path(os.environ.get("STAT3007_DATA", REPO / "model_fitting" / "data"))
SLIM = REPO / "report_v2" / "code" / "data"
CURVES = REPO / "model_fitting" / "data" / "curves"
OUT = REPO / "animations" / "data"

sys.path.insert(0, str(REPO / "repoduced-code"))
sys.path.insert(0, str(REPO / "animations"))
from RepoducedCode import make_modular_addition_dataset  # noqa: E402
from ntk_lib import report_statistics, sum_kernel  # noqa: E402
from common import kernels  # noqa: E402

P = 23
BASELINE = "ntk_N100_a1_wd0_s0"
COLUMNS = ["train_acc", "test_acc", "S_sum", "R_sum", "A_sum"]


def write_n100(runs, ckpt):
    """The checkpoint series of the runs at width 100, one row per run."""
    n100 = runs[runs.width == 100].sort_values(["alpha", "eta_kappa", "seed"])
    sub = ckpt[ckpt.run_id.isin(n100.run_id)]
    wide = {c: sub.pivot(index="run_id", columns="step", values=c).loc[n100.run_id] for c in COLUMNS}
    steps = wide["train_acc"].columns.to_numpy()
    assert len(n100) == 120 and len(steps) == 307, "expected 120 runs and 307 checkpoints"
    assert all(np.isfinite(w.to_numpy()).all() for w in wide.values()), "a value is missing"
    np.savez_compressed(OUT / "n100_checkpoints.npz",
                        run_id=n100.run_id.to_numpy().astype(str), alpha=n100.alpha.to_numpy(),
                        eta_kappa=n100.eta_kappa.to_numpy(), seed=n100.seed.to_numpy(), step=steps,
                        **{c: w.to_numpy(dtype=float) for c, w in wide.items()})


def write_split(runs):
    """The test mask of data seed 42 over the pairs in the order of kernels.one_hot_pairs."""
    assert (runs.loc[runs.seed == 0, "data_seed"] == 42).all(), "a seed 0 run does not use data seed 42"
    _, (X_test, _) = make_modular_addition_dataset(p=P, train_fraction=0.9, seed=42)
    X_test = X_test.numpy()
    _, a, b = kernels.one_hot_pairs(P)
    is_test = np.zeros(P * P, dtype=bool)
    is_test[X_test[:, :P].argmax(1) * P + X_test[:, P:].argmax(1)] = True
    assert is_test.sum() == 53 and (~is_test).sum() == 476, "expected 476 training and 53 test pairs"
    np.savez_compressed(OUT / "split_seed42.npz", a=a, b=b, is_test=is_test)
    return is_test


def copy_weights():
    source = FULL / "runs" / f"{BASELINE}_weights.npz"
    shutil.copyfile(source, OUT / source.name)


def check_kernel(is_test):
    """The numpy kernel against ntk_lib, and the report's statistics from the copied weights."""
    with np.load(OUT / f"{BASELINE}_weights.npz") as w:
        steps, W1, W2 = w["steps"], w["W1"].astype(float), w["W2"].astype(float)
    X, a, b = kernels.one_hot_pairs(P)
    for k in (0, len(steps) // 2, len(steps) - 1):
        ours, theirs = kernels.sum_kernel(W1[k], W2[k], X), sum_kernel(W1[k], W2[k], X)
        assert np.abs(ours - theirs).max() <= 1e-12 * np.abs(theirs).max(), "the numpy kernel differs from ntk_lib"

    X_test, Y_test = X[is_test], np.eye(P)[(a[is_test] + b[is_test]) % P]
    report = pd.read_parquet(SLIM / "report_kernel.parquet")
    report = report[report.run_id == BASELINE].set_index("step")
    assert list(report.index) == list(steps), "the saved steps differ from report_kernel.parquet"
    K_0 = kernels.sum_kernel(W1[0], W2[0], X_test)
    worst = 0.0
    for k, step in enumerate(steps):
        ours = kernels.statistics(kernels.sum_kernel(W1[k], W2[k], X_test), K_0, Y_test)
        theirs = report_statistics(sum_kernel(W1[k], W2[k], X_test), K_0, Y_test)
        for s in ("S_t", "R_t", "A_t"):
            worst = max(worst, abs(ours[s] - report.loc[step, s]), abs(ours[s] - theirs[s]))
    assert worst < 1e-9, f"the statistics differ from the report by {worst:.2e}"
    return worst


def check_curves(ckpt):
    """The committed seed 0 curves, which some clips load, against the report extract."""
    worst = 0.0
    names = sorted(p.stem for p in CURVES.glob("*.npz"))
    for name in names:
        rows = ckpt[ckpt.run_id == name].sort_values("step")
        with np.load(CURVES / f"{name}.npz") as curve:
            assert np.array_equal(curve["step"], rows.step.to_numpy()), f"{name}: the steps differ"
            for c in COLUMNS:
                worst = max(worst, float(np.abs(curve[c] - rows[c].to_numpy()).max()))
    assert len(names) == 144 and worst < 1e-6, f"the curves differ from the report extract by {worst:.2e}"
    return len(names), worst


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    runs = pd.read_csv(SLIM / "report_runs.csv")
    ckpt = pd.read_parquet(SLIM / "checkpoints_slim.parquet")
    write_n100(runs, ckpt)
    is_test = write_split(runs)
    copy_weights()
    worst_kernel = check_kernel(is_test)
    n_curves, worst_curve = check_curves(ckpt)
    for path in sorted(OUT.glob("*.npz")):
        print(f"wrote {path.relative_to(REPO)}  {path.stat().st_size / 1e6:.2f} MB")
    print(f"checks passed: the numpy kernel matches ntk_lib, the copied weights give the report's statistics "
          f"(largest difference {worst_kernel:.1e}), and the {n_curves} seed 0 curves match the report extract "
          f"(largest difference {worst_curve:.1e})")
