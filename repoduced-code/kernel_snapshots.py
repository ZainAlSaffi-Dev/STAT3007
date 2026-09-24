"""Retrain three Tier 1 cells and save the probe kernel at every checkpoint.

The Tier 1 grid runs in results/ keep the probe kernel at the first and the
last step only, so they cannot show how the kernel changes during training.
This script retrains three cells of that grid for the first 30,000 steps and
saves the kernel at every checkpoint, which is every 250 steps. The animations
load the files through animations/common/data.py.

The configuration is the one ntk_lib.load_cell uses for the grid. It is NTK
parameterisation, width 100, base rate 100, seed 0 and data seed 42. The run
length, the checkpoint interval and the kernel interval differ. So does the
probe. The dense runs keep the probe of the first 256 training pairs, which
every run used before 24 September 2026, because the scenes were built on
them. The grid now uses the mixed probe of the report. The weights follow
the same path in both, so the losses at a shared checkpoint should match the
saved grid run exactly, and the script prints the largest difference so that
this can be checked. The kernel fields are compared only when the probes
match.

The three cells all have alpha 1. The first has no weight decay. Its kernel
grows and turns, and it generalises. The second has eta kappa 0.0003, the
strongest decay in the Tier 1 grid. It generalises sooner with a smaller
kernel. The third has eta kappa 0.001, which Tier 1 Section 5 shows collapsing
the weights within about two thousand steps.

Each kernel file also records the probe pairs (a, b). A scene can then sort
the probe points by their label (a + b) mod p without importing torch.

Run it from repoduced-code with the environment at the repository root. Each
cell takes about 20 seconds on a laptop.

    uv run python kernel_snapshots.py            # all three cells
    uv run python kernel_snapshots.py 0.0003     # one cell, by its eta kappa
"""

import json
import sys
from pathlib import Path

import numpy as np

import ntk_lib as L

HERE = Path(__file__).resolve().parent
# ntk_lib writes next to itself. After the merge that is this folder anyway.
# The line below makes it true when ntk_lib is imported from elsewhere.
L.RESULTS = HERE / "results"
L.RESULTS.mkdir(exist_ok=True)

WIDTH, ALPHA, SEED = 100, 1.0, 0
DECAYS = [0.0, 3e-4, 1e-3]
STEPS, INTERVAL = 30000, 250


def run_name(eta_kappa):
    """Name of a dense run. It is the grid cell name with a different prefix."""
    return "dense" + L.cell_name(WIDTH, ALPHA, eta_kappa, SEED)[len("ntk"):]


def probe_pairs(cfg):
    """Return the probe pairs (a, b) that ntk_lib.train_run used for this configuration."""
    a, b, _ = L.probe_pairs({**L.LEGACY_VALUES, **cfg})
    return a, b


def add_probe_to_kernel_file(name, cfg):
    """Rewrite the kernel file of a run with the probe pairs added."""
    path = L.RESULTS / f"{name}_kernels.npz"
    data = dict(np.load(path))
    if "probe_a" in data:
        return
    data["probe_a"], data["probe_b"] = probe_pairs(cfg)
    np.savez_compressed(path, **data)


def compare_with_grid(run, eta_kappa):
    """Print the largest difference from the saved grid run at the checkpoints both share."""
    # The grid runs sit next to ntk_lib, which is this folder once the merge is done.
    grid_path = Path(L.__file__).resolve().parent / "results" / f"{L.cell_name(WIDTH, ALPHA, eta_kappa, SEED)}.json"
    if not grid_path.exists():
        print(f"{grid_path.name} is not in results, so there is nothing to compare against.")
        return
    saved = json.loads(grid_path.read_text())
    grid, mine = saved["history"], run["history"]
    index = {s: i for i, s in enumerate(grid["step"])}
    shared = [(i, index[s]) for i, s in enumerate(mine["step"]) if s in index]
    same_probe = {**L.LEGACY_VALUES, **saved["config"]}["probe"] == run["config"]["probe"]
    keys = ["train_loss", "test_loss"] + (["S_c", "R_c", "A_t"] if same_probe else [])
    if not same_probe:
        print("  the grid run uses another probe, so only the losses are compared")
    for key in keys:
        diff = max(abs(mine[key][i] - grid[key][j]) for i, j in shared)
        print(f"  {key:>10}: largest difference {diff:.2e} over {len(shared)} shared checkpoints")


def main(decays):
    for eta_kappa in decays:
        name = run_name(eta_kappa)
        run = L.run_or_load(name, parameterisation="ntk", hidden_dim=WIDTH, alpha=ALPHA,
                            eta_0=100.0, eta_kappa=eta_kappa, seed=SEED, steps=STEPS,
                            eval_interval=INTERVAL, kernel_save_interval=INTERVAL)
        add_probe_to_kernel_file(name, run["config"])
        print(f"{name}: compared with the saved grid run")
        compare_with_grid(run, eta_kappa)


if __name__ == "__main__":
    main([float(v) for v in sys.argv[1:]] or DECAYS)
