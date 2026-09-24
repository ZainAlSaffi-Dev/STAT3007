"""Loader for the saved runs, used by every scene that shows a measurement.

A scene must not carry a curve in its source. It loads the run that produced
the curve, so that the deck and the notebooks cannot drift apart. The runs
live in repoduced-code/results as one JSON per run, written by
ntk_lib.train_run.

Two kinds of run are saved there. The Tier 0 and Tier 1 runs come from the
notebooks and are committed. The dense runs come from
repoduced-code/kernel_snapshots.py. Their JSON histories are committed, but
their probe kernels are not, so a scene that draws the kernel itself needs that
script to have been run once on the machine. See animations/README.md.

The JSON holds the configuration and the history. The history keys are the
ones ntk_lib.HISTORY_KEYS lists: step, train_loss, test_loss, train_acc,
test_acc, S_t, R_t, S_c, R_c, A_t, param_dist, weight_norm, yKy and K_norm.
The centred terms are S_c and R_c. The report computes all three timing
variables on the centred kernel, so a scene wants S_c and R_c, not S_t and
R_t, unless it is showing the raw comparison on purpose.

The probe kernels themselves are saved as name_kernels.npz alongside the
JSON. Those files are in .gitignore, so they exist only on the machine that
ran the sweep. A scene that animates the kernel matrix or its eigenvectors
has to regenerate them by re-running the cell that produced the run.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REPRO = REPO / "repoduced-code"
RESULTS = REPRO / "results"

_MISSING = (
    "Call available() for the runs that exist. A dense run is written by "
    "repoduced-code/kernel_snapshots.py. Read animations/README.md."
)

# The three dense runs, with the value of eta lambda each one was trained with.
# They share alpha 1, width 100, NTK parameterisation and seed 0, and save the
# probe kernel every 250 steps up to step 30,000. Scenes that compare decay
# rates take their list from here, so that they all show the same cells.
DENSE_RUNS = [
    ("dense_N100_a1_wd0_s0", "0"),
    ("dense_N100_a1_wd0.0003_s0", "0.0003"),
    ("dense_N100_a1_wd0.001_s0", "0.001"),
]


def available():
    """Return the names of the runs that can be loaded, sorted."""
    if not RESULTS.is_dir():
        return []
    return sorted(p.stem for p in RESULTS.glob("*.json"))


def load_run(name):
    """Return one saved run as the dictionary ntk_lib.train_run wrote."""
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. {_MISSING}")
    return json.loads(path.read_text())


def history(run, *keys):
    """Return the named history series of a run as float arrays.

    Called with one key it returns one array. Called with several it returns
    one array per key, in the order given.
    """
    out = []
    for key in keys:
        if key not in run["history"]:
            have = ", ".join(sorted(run["history"]))
            raise KeyError(f"{key} is not in this run. It holds: {have}")
        out.append(np.asarray(run["history"][key], dtype=float))
    return out[0] if len(out) == 1 else tuple(out)


def load_kernels(name):
    """Return the saved probe kernels of a run as the arrays (steps, K).

    K has shape (checkpoints, n, n) with n the probe size, 256 by default.
    """
    path = RESULTS / f"{name}_kernels.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. These files are in .gitignore, so they "
            f"are only on the machine that ran the sweep. Re-run the cell "
            f"that produced the run {name} to write them again. For a dense run "
            f"that is repoduced-code/kernel_snapshots.py."
        )
    data = np.load(path)
    return data["steps"], data["K"]


def load_probe(name):
    """Return the probe pairs of a run as the integer arrays (a, b).

    Only the dense runs from repoduced-code/kernel_snapshots.py record them.
    The label of probe point i is (a[i] + b[i]) mod p.
    """
    path = RESULTS / f"{name}_kernels.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run repoduced-code/kernel_snapshots.py first.")
    data = np.load(path)
    if "probe_a" not in data:
        raise KeyError(f"{path.name} has no probe pairs. Only the dense runs from kernel_snapshots.py record them.")
    return data["probe_a"], data["probe_b"]


def grokking_time(run, tau_train, tau_test):
    """Return the two crossings and the grokking time, through ntk_lib.

    The definition is pre-registered and lives in one place. This function
    delegates to ntk_lib.grokking_time rather than repeating the thresholds,
    so a scene cannot label a gap that the notebooks would measure
    differently. ntk_lib imports torch, so this call needs the environment of
    repoduced-code rather than the one in this folder.
    """
    if str(REPRO) not in sys.path:
        sys.path.insert(0, str(REPRO))
    try:
        import ntk_lib
    except ImportError as exc:
        raise ImportError(
            f"ntk_lib could not be imported: {exc}. It needs torch, which the "
            f"animations environment does not install. Run the scene with the "
            f"repository root environment, or compute the times in a notebook."
        ) from exc
    return ntk_lib.grokking_time(run, tau_train, tau_test)
