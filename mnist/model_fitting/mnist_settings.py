"""Settings and loading shared by the MNIST model-fitting notebooks.

The four notebooks in this folder are MNIST copies of Zain's notebooks in
model_fitting/ on the zain branch. They read the tables that
mnist_sweep.py assemble writes. The settings that differ from modular
addition live here, so the notebooks agree on them.

- The knob that sets how lazy the network starts is the initial scale s,
  column init_scale, where the modular addition sweep has alpha.
- MNIST does not reach test accuracy 0.95 reliably, so the grokking step T is
  the first step with test accuracy at least LEVEL = 0.80. The notebooks also
  use the levels in LEVELS.
- The step budget is 500000, read from the data as last_step.
- report_kernel.parquet holds the kernel statistics at every checkpoint.
  The modular addition notebooks had them at 48 log-spaced steps, and their
  time-varying fits were built on those. kernel_steps picks the same number
  of log-spaced checkpoints, and load keeps only those.
- LANDMARK, the step t* at which the kernel is read for prediction, must come
  before any run reaches LEVEL. It is the kernel step nearest LANDMARK_TARGET.
  Check it against the data once the sweep has run.

Set the environment variable MNIST_SWEEP_DATA to read the tables from
another folder, for example a test fixture.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("MNIST_SWEEP_DATA", Path(__file__).resolve().parent.parent / "sweep"))

KNOB, KNOB_LABEL = "init_scale", "s"
MEM = 0.99
LEVEL = 0.80
LEVELS = (0.70, 0.80, 0.85)
N_KERNEL_STEPS = 48
LANDMARK_TARGET = 5_000

# The two experiments of notebook 1. The modular addition notebook compares weight decay 3e-5 with no decay;
# on MNIST the level at which the exploration found strong grokking is 3e-6.
WD_LEVEL = 3e-6
# The pilot cell, the setup of the Option B report.
PILOT = dict(width=200, init_scale=7.0)


def kernel_steps(steps, n=N_KERNEL_STEPS):
    """n checkpoint steps spaced evenly in log step, with step 0, chosen from the available steps."""
    steps = np.unique(np.asarray(steps))
    positive = steps[steps > 0]
    targets = np.logspace(np.log10(positive.min()), np.log10(positive.max()), n - 1)
    picked = {int(positive[np.argmin(np.abs(np.log(positive) - np.log(t)))]) for t in targets}
    return sorted(picked | {0})


def load(run_columns=("run_id", KNOB, "width", "eta_kappa", "seed", "last_step", "test_acc_final"),
         acc_columns=("run_id", "step", "train_acc", "test_acc")):
    """Return (runs, acc, kern, LANDMARK) from DATA. kern is cut to the kernel steps of kernel_steps."""
    runs = pd.read_parquet(DATA / "runs.parquet", columns=list(run_columns))
    acc = pd.read_parquet(DATA / "checkpoints.parquet", columns=list(acc_columns))
    kern = pd.read_parquet(DATA / "report_kernel.parquet")
    keep = kernel_steps(kern["step"])
    kern = kern[kern["step"].isin(keep)].reset_index(drop=True)
    landmark = int(min(keep, key=lambda s: abs(s - LANDMARK_TARGET)))
    return runs, acc, kern, landmark


def cell_name(row):
    """The label of a grid cell, as the modular addition notebooks build it from alpha, width and decay."""
    return f"{row[KNOB]:g}_{row['width']}_{row['eta_kappa']:g}"
