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

The animations use a separate environment in `animations/`, because Manim
needs system libraries that the experiments do not. Follow
`animations/README.md` for that one.

The report builds with Tectonic from `docs/`:

```
cd docs
tectonic report.tex
```
