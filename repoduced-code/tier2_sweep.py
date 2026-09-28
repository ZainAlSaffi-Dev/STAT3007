"""
This script holds the Tier 2 grid and trains the runs it needs. The grid
crosses width N with weight decay eta*lambda at alpha 1, in NTK
parameterisation, with five model seeds per cell. The notebook
tier2_width_decay.ipynb imports the constants below, so the grid is defined
in one place.

Runs are cached by ntk_lib.run_or_load, so the script can be stopped and
started again, and a run that is already saved with the same configuration is
loaded rather than trained. The N = 100 cells reuse the Tier 1 runs, which
have the same configuration.

The decisions behind the grid, and the reasons for them, are recorded in
tier_2.md. In short, the decay axis is the Tier 1 axis because a decay of
1e-3 stops the N = 100 network from memorising, and the wider networks get a
longer budget because under H they are expected to grok later.

Usage, from this folder, with the tara-env interpreter:

    /opt/miniconda3/envs/tara-env/bin/python tier2_sweep.py --width 1600 --threads 3

Seed 0 of every requested cell is trained before any other seed, so a sweep
that is stopped early still covers the whole grid.
"""

import argparse
import sys
import time

import torch

import ntk_lib as L

WIDTHS = [50, 100, 400, 1600]
DECAYS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4]
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 1.0
EVAL_INTERVAL = 500


def steps_for(N):
    """Step budget of a width: 100,000 steps up to N = 100, as in Tier 1, and 300,000 above it."""
    return 100_000 if N <= 100 else 300_000


def load(N, ek, seed, verbose=False):
    """Load one Tier 2 run from results/, training it first if it is missing."""
    return L.load_cell(N, ALPHA, ek, seed, steps_for(N), eval_interval=EVAL_INTERVAL, verbose=verbose)


def is_saved(N, ek, seed):
    """True if a run of this cell and seed has a saved file. The file may still hold a different configuration."""
    return (L.RESULTS / (L.cell_name(N, ALPHA, ek, seed) + ".json")).exists()


def main():
    parser = argparse.ArgumentParser(description="Train the Tier 2 width by weight-decay grid.")
    parser.add_argument("--width", type=int, nargs="+", default=WIDTHS, help="widths to run")
    parser.add_argument("--decay", type=float, nargs="+", default=DECAYS, help="values of eta*lambda to run")
    parser.add_argument("--seed", type=int, nargs="+", default=SEEDS, help="model seeds to run")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads for this process")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    print(f"interpreter {sys.executable}")
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    print(f"widths {args.width}, decays {args.decay}, seeds {args.seed}")

    # Seed 0 of every cell first, then the remaining seeds in order.
    order = [(N, ek, s) for s in args.seed for N in args.width for ek in args.decay]
    timings = []
    for N, ek, s in order:
        name = L.cell_name(N, ALPHA, ek, s)
        t0 = time.time()
        run = load(N, ek, s, verbose=True)
        seconds = time.time() - t0
        g = L.accuracy_crossings(run, 1.0)
        print(f"{name}: memorised at {g['t_train']}, generalised at {g['t_test']}, "
              f"grokking time {L.format_time(g)}, {seconds:.0f} s in this call", flush=True)
        timings.append((name, seconds))

    print("\nwall-clock seconds per run in this call (near zero means loaded from results/)")
    for name, seconds in timings:
        print(f"  {name:>32} {seconds:8.0f}")
    print(f"total {sum(s for _, s in timings) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
