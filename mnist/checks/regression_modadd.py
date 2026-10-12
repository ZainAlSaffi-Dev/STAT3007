"""
Regression check of section 2.4 of mnist/exploration/PLAN_mnist_closed_form.md.

It trains the same 1500-step modular addition run under two copies of
ntk_lib: the working copy in repoduced-code and the committed copy, which is
`git show HEAD:repoduced-code/ntk_lib.py` saved to a scratch folder with
RepoducedCode.py beside it. It then prints the largest absolute difference
over HISTORY_KEYS. The check passes only if that difference is 0.0.

    uv run python regression_modadd.py SCRATCH_DIR

SCRATCH_DIR is created if needed. The committed copy is written there by
`git show`, which reads the repository and changes no file in it.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE.parents[1] / "repoduced-code"
CONFIG = dict(probe="mixed", alpha=2.0, eta_kappa=1e-4, kernel_method="closed_form", seed=1, steps=1500)

RUNNER = """
import sys, json
sys.path.insert(0, {libdir!r})
import torch
torch.set_num_threads(1)
import ntk_lib as L
L.train_run("regression", verbose=False, results_dir={outdir!r}, **json.loads({cfg!r}))
"""


def run_with(libdir, outdir):
    code = RUNNER.format(libdir=str(libdir), outdir=str(outdir), cfg=json.dumps(CONFIG))
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(libdir))
    return json.loads((Path(outdir) / "regression.json").read_text())


def main(scratch):
    scratch = Path(scratch)
    head = scratch / "head_lib"
    head.mkdir(parents=True, exist_ok=True)
    for f in ("ntk_lib.py", "RepoducedCode.py"):
        text = subprocess.run(["git", "show", f"HEAD:repoduced-code/{f}"], cwd=str(HERE), check=True,
                              capture_output=True, text=True).stdout
        (head / f).write_text(text)
    new = run_with(LIB, scratch / "out_new")
    old = run_with(head, scratch / "out_head")
    sys.path.insert(0, str(LIB))
    import ntk_lib as L
    worst = 0.0
    for k in L.HISTORY_KEYS:
        a, b = np.asarray(new["history"][k], float), np.asarray(old["history"][k], float)
        if a.shape != b.shape:
            print(f"{k}: shapes differ, {a.shape} against {b.shape}")
            worst = float("inf")
            continue
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        worst = max(worst, float(np.nanmax(d)) if d.size else 0.0)
        if np.isnan(d).any():
            worst = float("inf")
    extra = sorted(set(new["history"]) - set(L.HISTORY_KEYS))
    print(f"history keys present only in the new run: {extra if extra else 'none'}")
    print(f"maximum absolute difference over HISTORY_KEYS: {worst!r}")
    print("REGRESSION PASS" if worst == 0.0 and not extra else "REGRESSION FAIL")
    return worst == 0.0 and not extra


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
