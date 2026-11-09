# Code for the G2 report

This folder holds the scripts behind the figures, tables and numbers in our report. They work from the training sweep we ran on 3 October 2026. The sweep has 720 runs, covering α ∈ {0.5, 1, 2}, N ∈ {50, …, 1600} and ηλ ∈ {0, 3e-6, …, 3e-3}, with 5 seeds each and 200,000 steps per run. None of the scripts trains a network. They all read a small extract of the sweep that is saved in `data/`, so this folder is all you need to rebuild the results.

## Running it

From this folder, run

```
python make_all.py
```

with Python 3.11 or later. You need numpy, pandas, pyarrow, scipy, matplotlib, scikit-learn, lifelines, xgboost and torch. The exact versions we used are in `environment.txt`. On macOS, xgboost also needs the OpenMP runtime, which you can get with `conda install llvm-openmp`. If you use conda, call the environment's interpreter directly (for example `/path/to/env/bin/python make_all.py`). We found that `conda run` can pick up the wrong environment when the shell profile changes `PATH`.

A full run took about 15 minutes on an Apple silicon Mac. Most of that time is the bootstrap in `E06_aft.py`. On macOS the script calls `caffeinate` so the machine does not sleep partway through.

`make_all.py` first checks the files in `data/` against the hashes in `data/SHA256SUMS.json`. It then runs each script in turn and writes a log for each one to `out/logs/`. Figures go to `out/figures/`, LaTeX tables to `out/tables/`, and the numbers to `out/numbers/` as one JSON file per script. Each JSON file also records the seeds and package versions used. At the end, `make_numbers.py` turns the JSON files into LaTeX macros in `out/numbers.tex`. The script also copies that file, the figures and the tables into `../overleaf/`, which is our report's folder. If you have only this folder, it creates `../overleaf/` next to it. Our report reads its numbers from those macros, so no number in the text was typed by hand.

## The MNIST section

`E19_mnist.py` covers the MNIST section of the report (§4.5 and Appendix F). It reads its own sweep of 560 runs from 19 October 2026, which is saved in `data/` as `mnist_runs.parquet`, `mnist_report_kernel.parquet` and `mnist_manifest.json`. These are copies of the tables in `mnist/sweep/` of our repository, and their hashes are checked along with the rest of `data/`.

## The full sweep

`extract.py` builds `data/` from the full sweep and copies in the MNIST tables. The full sweep is about 11 GB, so it is not included here. You only need `extract.py` if you want to rebuild the extract yourself. It looks for the sweep in the repository's `data/` folder, or in the folder named by the environment variable `STAT3007_DATA`. The sweep itself was produced by `repoduced-code/dataset_sweep.py` in our repository. It took 2.7 hours on 12 workers, which is 18.9 hours of total run time.

## Which script makes what

Each test in the main text answers one column of Table 1 in the report (`tab:predictions`). The times are from our last full run and are also in `out/logs/times.json`.

| Script | Report section | Output | Reads | Time |
|---|---|---|---|---|
| `N_numbers.py` | Table 2, §3.4 | Compute used (N1), checkpoint spacing (N3), test-loss floor (N4) | manifest, sweep timing, runs, checkpoints | 1 s |
| `E01_baseline.py` | §4.1, Fig. 1 | Baseline training curves, event steps by α (N5), share of D² due to scale (N2) | runs, checkpoints | 1 s |
| `E03_placebo.py` | §4.2, Table 3 | Spread of each statistic at grokking, at memorisation and at a placebo step; permutation p-values (N6 to N8) | runs, checkpoints | 1 s |
| `E02_thresholds.py` | §4.2, Fig. 2 | S, R and A at the grokking step | runs, E03 | 1 s |
| `E04_map.py` | §4.3, Fig. 3 | Grokking time and outcome over the grid at α = 1 | runs, checkpoints | 1 s |
| `E06_aft.py` | §4.3, Table 4 | Censored log-normal AFT fit of a, b and γ, with a cell bootstrap | runs | 9 min |
| `E05_scaling.py` | Not in the report (cut on 17 Oct; kept for the seminar) | T against ηλ for each width, with the fitted plane | runs, E06 | 1 s |
| `E07_mechanism.py` | §4.3, Fig. 4 | R and S at t* over N × ηλ; variance shares (N9) | runs, kernel | 1 s |
| `E08_ladder.py` | §4.4, Fig. 5 | Prediction models on a held-out α, compared with the settings alone; the figure shows five of them | runs, kernel | 1 min |
| `E09_surface.py` | Not in the report (cut on 17 Oct) | Kernel ridge surface over S × R and the product term γ_SR | runs, kernel | 4 min |
| `E10_subsets.py` | §4.4, Table 5 | Kernel ridge on subsets of the statistics, and the variance split | runs, kernel | 1 min |
| `E12_kumar.py` | Appendix B | Our baseline next to Kumar et al. (2024), arXiv v3 | E01 | 1 s |
| `E13_grid.py` | Appendix C | Outcomes over the whole grid | runs, checkpoints | 1 s |
| `E15_settings.py` | Appendix D | Settings of the prediction models | E08, E09 | 1 s |
| `E16_levels.py` | Appendix E | E03 and E06 repeated at grokking levels 0.80 and 0.90 | runs, checkpoints, E03, E06 | 1 s |
| `E19_mnist.py` | §4.5, Fig. 6, Appendix F | The MNIST version of E06 and E03, the weight norm at grokking against the law of Liu et al. (2023), and the statistics at grokking by width | the `mnist_` files in `data/` | 15 min |

There are no scripts for E11, E14, E17 or E18. E11, E17 and E18 were for a theory section that we took out of the report. E14 needed data from a second pipeline, and its appendix was dropped on 17 October.

`common.py` holds the shared paths, data loaders, event definitions, phase rule, colours and output helpers. `models.py` holds the prediction models used in §4.4 and their settings.

## Definitions

The events follow §3.4 of the report. A run memorises at the first checkpoint where its training accuracy is at least 0.99. It groks at level ℓ at the first checkpoint where its test accuracy is at least ℓ, counting from step 0. A run that never reaches a level is censored at 200,000 steps.

S, R and A are the columns `S_sum`, `R_sum` and `A_sum` in the checkpoint table. They are measured on the tangent kernel of the summed logits divided by √p, over the 53 test pairs of each run.

## Reproducibility

Every random seed is set at the top of its script and saved in that script's JSON output. Running `make_all.py` twice gives the same numbers. The PDF figures are saved without a creation date, so two runs can be compared by file hash. The MNIST figure is the exception. It was last made with matplotlib 3.11.2, and the other outputs with 3.10.8. With 3.10.8 its PDF differs slightly in the fonts, but it shows the same data and its numbers are identical.
