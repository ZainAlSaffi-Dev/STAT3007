# STAT3007 Group G2: what sets the grokking time on modular addition

The project asks whether the grokking time is set by rotation of the empirical
neural tangent kernel, by the scale of the kernel or the weights, or by the
weight-decay rate itself. The proposal is `docs/G2_proposal.pdf`. The rules
for anyone working here, including an assistant, are in `CLAUDE.md`. Read that
file before changing anything.

## What is where

- `repoduced-code/` holds the experiments. `ntk_lib.py` is the shared module
  that trains a run and records the kernel terms. The notebooks run Tier 0,
  the baseline from Kumar et al. (2024), Appendix 8.3, and Tier 1.
- `repoduced-code/results/` holds one JSON file per saved run. The probe
  kernels, `*_kernels.npz`, are too large for git and are written again on each
  machine.
- `animations/` holds the Manim scenes for the presentation and for explaining
  the maths. It has its own environment and its own `README.md`.
- `docs/` holds the proposal and the report source, `report.tex`.
- `ref.bib` holds the bibliography.

## Setting up

The experiments use the environment at the repository root.

```
uv sync
cd repoduced-code
uv run jupyter lab                   # the notebooks
uv run python kernel_snapshots.py    # the dense kernel runs the animations use
```

## The feature dataset

`repoduced-code/dataset_sweep.py` builds a dataset for exploratory analysis
and model fitting. It trains the crossed design of alpha {0.5, 1, 2}, width
{50, 100, 200, 400, 800, 1600}, eta lambda {0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
1e-3, 3e-3} and five seeds, which is 720 runs of 200,000 steps each. It
records about 140 features at about 300 checkpoints per run. The pilot
projected about two hours on the M4 Pro with 12 worker processes.

```
cd repoduced-code
uv run python dataset_sweep.py check      # the kernel and logger checks
uv run python dataset_sweep.py run        # the sweep; run it again to resume
uv run python dataset_sweep.py status     # how many runs have finished
uv run python dataset_sweep.py assemble   # the tables, dictionary, manifest and validation
```

The output is in `repoduced-code/results/dataset/`.

- `checkpoints.parquet` has one row per run and checkpoint.
- `runs.parquet` and `runs.csv` have one row per run. They hold the
  configuration, the grokking times under several definitions with their
  censoring flags, and the values at the events.
- `data_dictionary.md` and `data_dictionary.csv` give the meaning, formula and
  source of every column.
- `manifest.json` records the design, the git commit, the counts and the
  validation results.
- `runs/` holds the per-run files. The weights saved at about 50 checkpoints
  per run let any kernel be rebuilt with `ntk_lib.entk_closed_form`.

`checkpoints.parquet` and `runs/` are too large for git and are written
again by `run` and `assemble`. Load the tables with pandas:

```python
import pandas as pd
ckpt = pd.read_parquet("repoduced-code/results/dataset/checkpoints.parquet")
runs = pd.read_parquet("repoduced-code/results/dataset/runs.parquet")
data = ckpt.merge(runs, on=["run_id", "alpha", "width", "eta_kappa", "seed"])
```

The animations use a separate environment in `animations/`, because Manim
needs system libraries that the experiments do not. Follow
`animations/README.md` for that one.

The report builds with Tectonic from `docs/`:

```
cd docs
tectonic report.tex
```
