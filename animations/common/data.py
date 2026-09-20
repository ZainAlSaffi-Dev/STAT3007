"""Loader for the saved runs, used by every scene that shows a measurement.

A scene must not carry a curve in its source. It loads the run that produced
the curve, so that the deck and the notebooks cannot drift apart. The runs
live in repoduced-code/results as one JSON per run, written by
ntk_lib.train_run.

State of the data on this branch. The runs and ntk_lib.py are on origin/sam
and have not been merged yet, so every function here raises until that merge
lands. The error message says what is missing. See animations/README.md.

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
    "The saved runs are not on this branch yet. They are on origin/sam, "
    "together with ntk_lib.py, and the merge is waiting on the helper "
    "functions that the tier notebooks call. Read animations/README.md."
)


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
            f"that produced the run {name} to write them again."
        )
    data = np.load(path)
    return data["steps"], data["K"]


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
            f"animations environment does not install. {_MISSING}"
        ) from exc
    return ntk_lib.grokking_time(run, tau_train, tau_test)
