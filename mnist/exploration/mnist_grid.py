"""
A crossed grid on MNIST in the style of repoduced-code/dataset_sweep.py (Zain's
modular addition sweep), for the bias-free three layer network of Option B.

Every run uses the best Option B setup (results/mnist_optionB/REPORT.md):
DeepBiasFreeMLP with standard parameterisation and depth 3, uncentred, plain
SGD with batch 200, eta_0 = 1 / lam_0 and the curvature learning rate rule.
The grid varies width, init_scale and eta_kappa. Each seed draws its own
training subset from data seed 42 + seed, as dataset_sweep.py does.

The pilot checks that widths above 200 still grok at the best cell
(init_scale 7, eta_kappa 3e-6) and measures their run time.

    uv run python mnist_grid.py run pilot
    uv run python mnist_grid.py summary pilot
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

import mnist_optionB as B

OUT = Path(__file__).resolve().parent / "results" / "mnist_grid"
STEPS = 500_000
DATA_SEED_BASE = 42


def grid_config(width, scale, eta_kappa, seed):
    """Return (name, config, lam_0) for one run of the grid."""
    ratio, extra = B.RULES["curv"]
    cfg = dict(B.base_config(STEPS), hidden_dim=width, init_scale=scale, eta_kappa=eta_kappa, seed=seed,
               data_seed=DATA_SEED_BASE + seed, **extra)
    return B.with_rate(f"grid_w{width}_s{scale:g}_ek{B.ek_tag(eta_kappa)}_seed{seed}", cfg, ratio)


def pilot_configs():
    return [grid_config(w, 7.0, 3e-6, seed) for w in (800, 400) for seed in (0, 1)]


BATCHES = dict(pilot=pilot_configs)


def _train(item):
    torch.set_num_threads(int(os.environ.get("GRID_THREADS", "1")))
    name, cfg, _ = item
    return B.L.run_or_load(name, verbose=False, results_dir=OUT, **cfg)["seconds"]


def summary(batch):
    rows = {}
    for name, cfg, lam in BATCHES[batch]():
        path = OUT / f"{name}.json"
        if path.exists():
            run = json.loads(path.read_text())
            r = B.classify(run)
            r["lam_0"], r["eta"], r["seconds"] = lam, cfg["eta_0"], run["seconds"]
            rows[name] = r
    (OUT / f"summary_{batch}.json").write_text(json.dumps(rows, indent=1, default=str))
    B.print_table(rows)
    for name, r in rows.items():
        print(f"{name}: {r['seconds']:.0f} s, lam_0 {r['lam_0']:.4g}")
    return rows


def main(cmd, batch):
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "run":
        items = BATCHES[batch]()
        for name, cfg, lam in items:
            print(f"{name}: lam_0 {lam:.4g}, eta {cfg['eta_0']:.4g}", flush=True)
        with ProcessPoolExecutor(max_workers=min(len(items), 12)) as pool:
            for (name, _, _), sec in zip(items, pool.map(_train, items)):
                print(f"{name}: {sec:.0f} s", flush=True)
        summary(batch)
    elif cmd == "summary":
        summary(batch)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "pilot")
