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
Runs made from 24 September 2026 on also hold S_c, R_c, A_t and A_u on the
training part and the test part of the probe, as S_c_train, S_c_test and so
on. The grid runs use the mixed probe of the report, 203 training pairs and
53 test pairs. The dense runs keep the probe of the first 256 training pairs.
The centred terms S_c and R_c of these runs belong to the explainer scenes. The
deck uses the statistics of the report, described below.

The probe kernels themselves are saved as name_kernels.npz alongside the
JSON. Those files are in .gitignore, so they exist only on the machine that
ran the sweep. A scene that animates the kernel matrix or its eigenvectors
has to regenerate them by re-running the cell that produced the run.

The feature dataset of repoduced-code/dataset_sweep.py is a third source. It
covers alpha {0.5, 1, 2}, width {50, 100, 200, 400, 800, 1600} and eta lambda
{0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}, with five seeds, for 200,000
steps. It lives in model_fitting/data. Two parts of it are committed. The
runs table, runs.csv, has one row per run with the grokking times and the
values at the events. load_runs_table reads it. The curves of the seed 0
runs, curves/<run_id>.npz, hold every checkpoint column of the dataset.
load_curves reads one of them in the same form as load_run, so history works
on it. model_fitting/data/data_dictionary.md defines every column. The long
table of all seeds is a parquet file that git ignores, and this environment
cannot read parquet in any case.

The deck clips follow the restructured report, report_v2. Its statistics are
measured on the tangent kernel of the summed logits over sqrt(p), on the 53
test pairs of each run. They are the columns S_sum, R_sum and A_sum of the
curves above and of the report's own extract in report_v2/code/data. Four
sources serve the clips.

- load_report_runs reads report_runs.csv of that extract: one row per run of
  the 720, with the events at each level and the statistics at each event.
- report_numbers reads the results of one report item from
  report_v2/code/out/numbers, and deck_number reads the string that the slides
  print for a number, from presentation/_variables.yml.
- load_n100, load_split and load_weights read the files in animations/data,
  which extract_deck_data.py writes from the report extract and from the full
  sweep. Git keeps them, so the clips render on any machine.
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REPRO = REPO / "repoduced-code"
RESULTS = REPRO / "results"
DATASET = REPO / "model_fitting" / "data"
CURVES = DATASET / "curves"
REPORT_CODE = REPO / "report_v2" / "code"
DECK = REPO / "presentation"
DECK_DATA = Path(__file__).resolve().parents[1] / "data"

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


def dataset_run_id(width, alpha, eta_kappa, seed=0):
    """Return the name of a dataset run, the same name ntk_lib.cell_name gives it."""
    return f"ntk_N{width}_a{alpha:g}_wd{eta_kappa:g}_s{seed}"


def curve_runs():
    """Return the names of the dataset runs whose curves are committed, sorted."""
    if not CURVES.is_dir():
        return []
    return sorted(p.stem for p in CURVES.glob("*.npz"))


def load_curves(name):
    """Return the curves of one dataset run as a dictionary with a config and a history.

    The history maps every checkpoint column of the dataset to an array in
    checkpoint order, so history(run, "step", "S_c") works as it does for a
    saved run. The config holds alpha, width, eta_kappa and seed.
    """
    path = CURVES / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Only the seed 0 runs of the dataset have committed curves. "
            f"curve_runs() lists them, and dataset_run_id() builds a name.")
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    config = {key: arrays.pop(key).item() for key in ("alpha", "width", "eta_kappa", "seed")}
    return dict(config=config, history=arrays)


def _parse(value):
    """Turn one cell of runs.csv into a float, a bool, or the string itself. An empty cell is NaN."""
    if value == "":
        return np.nan
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def load_runs_table(path=DATASET / "runs.csv"):
    """Return a runs table as a dictionary of column name to array, one entry per run.

    The default is runs.csv of the dataset. Numeric columns are float arrays,
    with NaN where the value is empty, for example the grokking time of a run
    that never memorised. The flag columns are bool arrays, unless a run left
    one empty, and the text columns are object arrays.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. repoduced-code/dataset_sweep.py assemble writes it.")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    table = {}
    for key in rows[0]:
        values = [_parse(r[key]) for r in rows]
        kinds = {type(v) for v in values if not (isinstance(v, float) and np.isnan(v))}
        if kinds <= {float}:
            table[key] = np.asarray(values, dtype=float)
        elif kinds == {bool} and all(isinstance(v, bool) for v in values):
            table[key] = np.asarray(values, dtype=bool)
        else:
            table[key] = np.asarray(values, dtype=object)
    return table


def load_report_runs():
    """Return report_runs.csv of the report extract, in the form load_runs_table gives.

    Its events follow report_v2 Section 3.4. t_mem is the first checkpoint at
    which training accuracy reaches 0.99, and t_grok95 the first at which test
    accuracy reaches 0.95. A run that misses a level has censored_<level> True
    and its time set to the budget of 200,000 steps.
    """
    return load_runs_table(REPORT_CODE / "data" / "report_runs.csv")


def report_numbers(item):
    """Return the results of one report item, such as "E03", from report_v2/code/out/numbers."""
    return json.loads((REPORT_CODE / "out" / "numbers" / f"{item}.json").read_text())["results"]


def deck_number(name):
    """Return the string that the slides print for a number, such as deck_number("EthreeNhundredRRatio").

    It reads presentation/_variables.yml with the pattern that filters/g2.lua uses,
    so a clip prints exactly what the slide prints. The strings may hold the
    minus sign U+2212, which MathTex needs written as "-".
    """
    text = (DECK / "_variables.yml").read_text()
    match = re.search(rf'^{re.escape(name)}:\s*"(.*?)"', text, flags=re.MULTILINE)
    if match is None:
        raise KeyError(f"{name} is not in presentation/_variables.yml. make_numbers.py writes that file.")
    return match.group(1)


def _deck_file(name):
    path = DECK_DATA / name
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. animations/extract_deck_data.py writes it.")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_n100():
    """Return the 120 runs at width 100 of the report extract, at all 307 checkpoints.

    The dictionary holds run_id, alpha, eta_kappa and seed with one entry per
    run, step with one entry per checkpoint, and train_acc, test_acc, S_sum,
    R_sum and A_sum with one row per run and one column per checkpoint.
    """
    return _deck_file("n100_checkpoints.npz")


def load_split():
    """Return the arrays a, b and is_test of the 529 pairs under data seed 42, the split of every seed 0 run.

    Pair i is (a[i], b[i]), in the order of kernels.one_hot_pairs. 476 pairs train
    and 53 test.
    """
    return _deck_file("split_seed42.npz")


def load_weights(name):
    """Return the saved steps and the weights W1 and W2 of a run whose weights are in animations/data.

    W1 has shape (steps, N, 2p) and W2 has shape (steps, p, N). Only the
    baseline run ntk_N100_a1_wd0_s0 is there for now.
    """
    weights = _deck_file(f"{name}_weights.npz")
    return weights["steps"], weights["W1"].astype(float), weights["W2"].astype(float)


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
