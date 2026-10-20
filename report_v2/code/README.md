# Code for the report_v2 figures and numbers

This folder rebuilds every figure, table and number of the G2 report (`report_v2/overleaf/main.tex`) from the training sweep of 3 October 2026. That sweep has 720 runs: α ∈ {0.5, 1, 2}, N ∈ {50, …, 1600}, ηλ ∈ {0, 3e-6, …, 3e-3}, 5 seeds, and 200,000 steps. No script trains a network. Each one reads a small frozen extract of the sweep in `data/`.

## Rebuild everything

```
python make_all.py
```

Run this from this folder, with the interpreter of an environment that has the packages below. Name the interpreter explicitly, for example `/path/to/env/bin/python make_all.py`. A shell profile that edits `PATH` can otherwise send `conda run` to a different environment. On macOS the script starts `caffeinate` so that the computer stays awake. It takes about 15 minutes on one Apple silicon Mac, and most of that is the bootstrap of E06.

`make_all.py` does five things:

1. It checks `data/` against `data/SHA256SUMS.json`.
2. It runs every item script.
3. It writes `overleaf/numbers.tex` with `make_numbers.py`.
4. It copies the figures and tables into the Overleaf folder.
5. It writes `environment.txt` and a log for each script in `out/logs/`.

**Packages.** Use Python 3.11 or later. The run behind the report used numpy, pandas, pyarrow, scipy, matplotlib, scikit-learn 1.9.1, lifelines 0.30.0, xgboost 3.2.0 and torch. `environment.txt` gives the exact versions. On macOS, xgboost also needs the OpenMP runtime, for example `conda install llvm-openmp`.

**The full data.** `extract.py` writes `data/` from the full sweep, which is about 11 GB and is not in git. It reads the repository's `data/` folder, or the folder that `STAT3007_DATA` names. Run it only when you have the full data. `repoduced-code/dataset_sweep.py` regenerates the full sweep. It took 2.7 h on 12 workers, or 18.9 h of summed run time.

## Where the outputs go

| Output | Written to | Copied to |
|---|---|---|
| Figures (PDF, with a PNG preview) | `out/figures/E<k>_<name>.pdf` | `overleaf/figures/` (PDF only) |
| LaTeX tables | `out/tables/<name>.tex` | `overleaf/tables/` |
| Numbers, with seeds and versions | `out/numbers/<item>.json` | `overleaf/numbers.tex`, as macros |
| Logs and run times | `out/logs/` | |

`main.tex` reads only the Overleaf folder, so that folder can be uploaded to Overleaf as it is.

## How each script relates to the report

Each test in the main text answers one column of Table 1 of the report (`tab:predictions`).

| Script | Place in the report | What it gives | Inputs | Time |
|---|---|---|---|---|
| `N_numbers.py` | Table 2; §3.4 | N1 compute; N3 checkpoint resolution; N4 test-loss floor | manifest, sweep timing, runs, checkpoints | 1 s |
| `E01_baseline.py` | §4.1, Fig. 1 | Baseline dynamics; N5 event steps by α; N2 share of D² carried by scale | runs, checkpoints | 1 s |
| `E03_placebo.py` | §4.2 (Tier 1), Table 3 | SD at grokking, at memorisation and placebo; tightening ratio; permutation p; N6, N7, N8 | runs, checkpoints | 1 s |
| `E02_thresholds.py` | §4.2, Fig. 2 | S, R and A at the grokking step | runs, E03 | 1 s |
| `E04_map.py` | §4.3 (Tier 2), Fig. 3 | Time and outcome map at α = 1 | runs, checkpoints | 1 s |
| `E06_aft.py` | §4.3, Table 4 | Censored log-normal AFT fits of a, b and γ with cell bootstrap | runs | 9 min |
| `E05_scaling.py` | §4.3, commented out on 17 Oct 2026 (kept for the seminar) | T against ηλ by width, with the fitted plane | runs, E06 | 1 s |
| `E07_mechanism.py` | §4.3, Fig. 4 | R and S at t* over N × ηλ; N9 variance shares | runs, kernel | 1 s |
| `E08_ladder.py` | §4.4 (Tier 3), Fig. 5 | Model ladder on a held-out α, with the settings as a reference and each setting alone; the figure shows five of its rungs | runs, kernel | 1 min |
| `E09_surface.py` | §4.4, commented out on 17 Oct 2026 | Kernel ridge surface over S × R; product coefficient γ_SR | runs, kernel | 4 min |
| `E10_subsets.py` | §4.4, Table 5 | Kernel ridge on subsets of the statistics; variance split | runs, kernel | 1 min |
| `E12_kumar.py` | Appendix B | The baseline against Kumar et al. (2024), arXiv v3 | E01 | 1 s |
| `E13_grid.py` | Appendix C | Outcomes over the whole grid | runs, checkpoints | 1 s |
| `E15_settings.py` | Appendix D | Settings of the prediction models | E08, E09 | 1 s |
| `E16_levels.py` | Appendix E | E03 and E06 at grokking levels 0.80 and 0.90 | runs, checkpoints, E03, E06 | 1 s |
| `E19_mnist.py` | §4.5 (Tier 4), Fig. 6; Appendix F | The MNIST tier: the fit of E06 on the MNIST sweep, the weight norm at grokking and the law of Liu et al. (2023), the placebo test of E03 on S, R and A, and the statistics at grokking by width | `mnist/sweep/runs.parquet` and `report_kernel.parquet` in the repository, not `data/` | 15 min |

Items that are not built:

- E11, E17 and E18 belong to the Tier 4' theory section. That section is commented out in `main.tex`, so they are not built.
- E14 needs the databases of the second pipeline (release grid-v0.1.0, branch `feat/grid`), so it is not built. Its appendix was dropped on 17 Oct 2026.

`out/logs/times.json` holds the run time of each script from the last full run.

## Shared code

- `common.py` holds the paths, the loaders, the event definitions, the phase rule, the colours and the output writers.
- `models.py` holds the prediction models of Tier 3 and their settings.

The events follow §3.4 of the report:

- Memorisation is the first checkpoint with training accuracy ≥ 0.99.
- Grokking at level ℓ is the first checkpoint with test accuracy ≥ ℓ, counted from step 0.
- A run that misses a level is censored at 200,000 steps.

The kernel statistics S, R and A are the columns `S_sum`, `R_sum` and `A_sum` of the checkpoints. They are measured on the tangent kernel of the summed logits over √p, on the 53 test pairs of each run.

## Reproducibility

- Every random seed is fixed at the top of its script and written into its JSON file.
- Two full runs give the same numbers.
- The PDF figures are written without a creation date, so they can be compared by hash.
