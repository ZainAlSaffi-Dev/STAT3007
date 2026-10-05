# Experiments TODO for report_v2

This file lists every figure, table and number that `overleaf/main.tex` needs. Each item has an ID, and the same ID appears in red in the report next to the placeholder or the planning value it replaces. The file has two uses. First, it is the work list for making the figures and numbers. Second, it sets out how the code that makes them will later become a clean repository for readers.

Written on 5 October 2026. The planning values below come from read-only checks on `data/` made that day. Treat each one as a target to recompute, not as a result.

## Ground rules

- Every item reads from `data/` only. That is the sweep of 3 October 2026 with a separate split for each seed. Its design is α ∈ {0.5, 1, 2}, N ∈ {50, 100, 200, 400, 800, 1600} and ηλ ∈ {0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}, with seeds 0 to 4 and a budget of 200,000 steps. It holds 720 runs, and all finished with status `ok`.
- Do not use `data/runs_seed42/` or `data/seed42_tables/`. They hold the old runs, in which every seed shared one split.
- The events follow the golden report §3.5 and CLAUDE.md. Memorisation is the first checkpoint with training accuracy ≥ 0.99. Grokking at level ℓ is the first checkpoint with test accuracy ≥ ℓ, with ℓ = 0.95 in the main text and 0.80 and 0.90 in Appendix F. Both are counted from step 0. A run that misses a level is censored at 200,000.
- The kernel statistics are S, R and A on the kernel of the summed logits over √p, on the 53 test pairs of each run. In `checkpoints.parquet` they are the columns `S_sum`, `R_sum` and `A_sum`. In `report_runs.csv` they are `S_t_at_*`, `R_t_at_*` and `A_t_at_*`. In `report_kernel.parquet` they are `S_t`, `R_t` and `A_t`. S is a plain ratio, so 1 means no change. R is uncentred. A is the centred alignment with Y Yᵀ.
- The column `eta_kappa` is ηλ.
- Use Python 3.11 or later with the packages in `pyproject.toml` at the repository root. The fits need `lifelines` and `xgboost`. Record the interpreter and package versions with each output.
- Fix every random seed and write it next to the output.

## Where the data are

| File | Rows | What it holds |
|---|---|---|
| `data/report_runs.csv` | 1 per run (720) | Settings; `t_mem`, `t_grok80`, `t_grok90` and `t_grok95` with their `censored_*` flags and `*_prev` checkpoints; S, R, D and A at each event and at the end. |
| `data/checkpoints.parquet` | 307 per run | `step`, the accuracies and losses, and `S_sum`, `R_sum`, `D_sum`, `A_sum` at every checkpoint. It has 146 columns, so read only the ones needed. |
| `data/report_kernel.parquet` | 48 per run | `S_t`, `R_t`, `D_t` and `A_t` at the steps whose weights were saved, including t* = 2,430. |
| `data/runs/<run_id>_weights.npz` | 48 steps | `W1` and `W2`, from which `ntk_lib.sum_kernel` rebuilds any kernel. |
| `data/curves/<run_id>.npz` | seed 0 only | Every checkpoint column, already committed. |
| `data/manifest.json` | | Design, fixed settings, versions and commit. |
| `model_fitting/data/` | | The same tables without the weights. Notebooks 1 to 4 read from here. |

## Figures and tables

Status words: **carry over** means the number exists and only needs a clean script; **port** means the method exists in an old notebook and must be rerun on `data/`; **new** means nothing exists yet.

### E1. Baseline dynamics (Fig. 1, §4.1)
- **Question.** Does the baseline grok, does a lazier network grok later, and does the kernel grow more than it rotates?
- **Setup.** N = 100, ηλ = 0, α ∈ {0.5, 1, 2}, 5 seeds each.
- **Data.** `checkpoints.parquet`: `step`, `train_acc`, `test_acc`, `S_sum`, `R_sum`, `A_sum`. Events come from `report_runs.csv`.
- **Method.** There are four panels against log step: accuracies, S, R and A. Each shows the median over seeds with the range shaded. Mark memorisation with a hollow marker and grokking at 0.95 with a filled one.
- **Port from.** `repoduced-code/tier0_v2.ipynb` §3 (old data, layout only).
- **Output.** `overleaf/figures/E1_baseline.pdf`. **Status:** port.

### E2. Statistics at the grokking step (Fig. 2, §4.2)
- **Setup.** All runs that grok at 0.95. Left block: N = 100, grouped by α and coloured by ηλ. Right block: α = 1, grouped by N.
- **Data.** `report_runs.csv`: `S_t_at_grok95`, `R_t_at_grok95`, `A_t_at_grok95`.
- **Method.** One panel per statistic, labelled with the CV and the tightening ratio from E3.
- **Output.** `E2_thresholds.pdf`. **Status:** new.

### E3. Threshold table with placebo (Table 3, §4.2)
- **Setup.** Two groups of runs: N = 100 across α × ηλ (78 grokked runs), and α = 1 across N × ηλ (130 grokked runs).
- **Data.** `report_runs.csv` for the events. `checkpoints.parquet` to read each run at another run's grokking step: take the value at that step, or the last checkpoint before it.
- **Method.**
  - Compute the CV at grokking, at memorisation and at the placebo step.
  - The placebo uses 2,000 random permutations with no fixed points.
  - Report the tightening ratio (median placebo CV over the true CV) and the permutation p (the share of permutations at least as tight).
  - Also run the test across all grokked runs.
- **Planning values** (100 permutations):

  | Group | S: CV, placebo, ratio | R: CV, placebo, ratio | A: CV, placebo, ratio |
  |---|---|---|---|
  | N = 100 | 0.48, 0.62, 1.3× | 0.20, 0.37, 1.9× | 0.08, 0.09, 1.1× |
  | α = 1 | 0.75, 0.78, 1.0× | 0.75, 0.77, 1.0× | 0.13, 0.13, 1.0× |

- **Port from.** `repoduced-code/tier1_v2.ipynb` §7 (placebo code, on old data).
- **Output.** A table, and numbers in `overleaf/numbers.tex`. **Status:** port.

### E4. Time and outcome map (Fig. 3, §4.3)
- **Setup.** α = 1, all N × ηλ.
- **Data.** `report_runs.csv`: `t_grok95` and `censored_grok95`, plus the final accuracies for the outcome letters.
- **Method.**
  - Colour is the median log T95 over the seeds that grok, shown only when at least 3 of 5 grok.
  - Each cell is labelled with the seeds that grok out of 5.
  - Cells with no grokking seed get a letter: M memorises only, F forgets (training accuracy rose above 0.9 and ended below it), N never fits. The letters use the phase rule of the golden report at threshold 0.9.
- **Check against.** The table in `repoduced-code/tier2_reproduction.md` §7.
- **Output.** `E4_map.pdf`. **Status:** carry over (the table exists; the figure is new).

### E5. Scaling with weight decay at each width (Fig. 4, §4.3)
- **Setup.** α = 1, ηλ > 0, all widths, 5 seeds.
- **Method.**
  - Plot log T95 against log ηλ, one line per N, with seeds as points and censored runs as open markers at the budget.
  - Overlay the fitted plane of E6 for the main range.
  - Draw grey reference slopes of 0 and −1.
- **Output.** `E5_scaling.pdf`. **Status:** new.

### E6. Scaling fits (Table 4, §4.3)
- **Setup.** α = 1.
  - Main range: ηλ ∈ {1e-5, 3e-5, 1e-4} at all six widths (90 runs, none censored).
  - Checks: adding 3e-4; levels 0.80 and 0.90; the memorisation step.
- **Method.**
  - Fit the log-normal AFT `log T = c + a log N + b log ηλ + γ log N log ηλ`, with both logarithms centred at the middle of the grid. Use lifelines `LogNormalAFTFitter` with censoring.
  - Compute intervals by bootstrap that resamples whole cells and seeds within cells. The older intervals resampled seeds only, which leaves out the misfit of the cells about the plane.
  - Report σ, and the spread between seeds within cells beside it.
- **Planning values** (`repoduced-code/results/tier2_consistency.json`, part C, seed bootstrap):
  - Main range: a = +0.24 [+0.20, +0.28], b = −0.63 [−0.66, −0.60], γ = −0.16, σ = 0.38.
  - With 3e-4: a = +0.43, b = −0.21.
  - Level 0.80: a = +0.30, b = −0.55. Level 0.90: a = +0.27, b = −0.60.
  - Memorisation: a = −1.53.
- **Port from.** `repoduced-code/tier2_consistency.py` part C.
- **Output.** A table, plus macros. **Status:** port.

### E7. Width sets rotation, weight decay sets scale (Fig. 5, §4.3)
- **Setup.** α = 1, all N × ηλ, read at t* = 2,430.
- **Data.** `report_kernel.parquet` at `step == 2430`.
- **Method.** Two heatmaps: the median log R and the median S per cell. Also compute N9.
- **Port from.** `model_fitting/4_nonlinear_exploration.ipynb` Part 2 (the setting maps).
- **Output.** `E7_mechanism.pdf`. It may be merged into E4. **Status:** port.

### E8. Model ladder on a held-out α (Fig. 6, §4.4)
- **Setup.** All 630 runs with ηλ ≤ 1e-3. The statistics are at t* = 2,430, with log R. Hold out one α at a time.
- **Method.**
  - The ladder: each statistic alone in a linear AFT; the linear AFT; the AFT with interactions; kernel ridge; XGBoost with `survival:aft`; and a GRU on the 25 saved steps up to t*.
  - The reference line is XGBoost on N and ηλ, labelled as the model that is given the settings.
  - Score each fold by concordance. Show the three folds as points and the mean as a bar. For XGBoost and the GRU, add the range over seeds.
- **Planning values** (notebook 3, Parts 3 and 5):
  - Single statistics: A 0.519, S 0.541, R 0.549.
  - Linear 0.585; with interactions 0.618.
  - Kernel ridge 0.568; XGBoost 0.708; GRU 0.768.
  - Settings reference 0.780.
- **Port from.** `model_fitting/3_kernel_only.ipynb`.
- **Output.** `E8_ladder.pdf`. **Status:** carry over.

### E9. The S × log R surface (Fig. 7, §4.4)
- **Setup.** Grokked runs (or all runs with XGBoost). A is held at its median.
- **Method.**
  - Draw the kernel ridge prediction over S and log R, with the runs overlaid and the empty region shaded.
  - Also report γ_SR from the AFT with interactions and its cell-bootstrap interval. Planning value: −0.62 [−1.41, −0.12], from notebook 3 Part 2.
- **Port from.** Notebook 3 Part 4 and notebook 4 Part 5 (maps).
- **Output.** `E9_surface.pdf`. **Status:** carry over.

### E10. Subsets and variance split (Table 5, §4.4)
- **Setup.** The 355 runs that grok at 0.95 among those with ηλ ≤ 1e-3. The statistics are at t*.
- **Method.**
  - Kernel ridge (RBF) on each subset of {S, log R, A}: the three singles, the three pairs, and all three. Score by leave-one-cell-out R² and concordance.
  - Choose the penalty and width inside each fold by grouped cross-validation. The planning run fixed them at 0.1 and 0.1, the values notebook 4 chose.
  - Variance split: the share of the spread of log T, S, log R and A that lies between cells.
- **Planning values.**
  - Subsets, as R² and concordance:
    - A: 0.02, 0.44
    - S: 0.27, 0.64
    - log R: 0.37, 0.71
    - S + log R: 0.52, 0.76
    - S + A: 0.26, 0.65
    - log R + A: 0.39, 0.73
    - all three: 0.57, 0.78
  - Share between cells: log T 0.964, S 0.987, log R 0.993, A 0.563 (notebook 4 Part 2).
- **Output.** A table. **Status:** new (subsets) and carry over (split).

### E11. Rate-addition model (Fig. 8, §4.5, option A; on hold)
- **On hold.** Tier 4' is undecided. §4.5 of the report lists options A to E. Run this item only if the group chooses option A, and run E17 first, because this model assumes what E17 tests.
- **Setup.** α = 1, all widths, ηλ from 0 to 3e-4. Leave out the cells above the edge where no seed memorises or every seed forgets.
- **Method.**
  - Fit `1/T = r0 N^(−q) + c ηλ` by maximum likelihood with a noise model on log T. State the noise model as a choice.
  - Censored runs enter as T > 200,000.
  - Compare with E6 by AIC on the same runs and by leave-one-cell-out error.
  - Report q, c and the share of the rate from weight decay at each width.
  - Check whether the model reproduces the sign of γ.
- **Port from.** The reciprocal form in `repoduced-code/tier2_width_decay.ipynb` §7 (old data and definitions; the method only).
- **Output.** `E11_rates.pdf`. **Status:** new.

### E12. Baseline against Kumar et al. (Appendix B)
- **Task.** Open Kumar et al. (2024), arXiv:2310.06110v3, Appendix 8.3. Compare every setting with Table 2 of the report: task, encoding, training fraction, width, parameterisation, loss, optimiser, learning rate, α values and step budget. Record any difference.
- **Also.** Tabulate the memorisation and grokking steps by α at N = 100 and ηλ = 0, beside what Kumar et al. report or show. α = 1.5 is not in `data/`; say so.
- **Status:** new (reading) and port (`tier0_v2.ipynb` §2).

### E13. Grid outcome tables (Appendix C)
- For each α, a table over N × ηλ giving the seeds that grok out of 5 and the median T95. **Status:** carry over (extend the §7 table of `tier2_reproduction.md` to α = 0.5 and 2).

### E14. Cross-check from the second pipeline (Appendix D)
- **Source.** The golden report §4.2.2 on main (commit 7da1996), built from release grid-v0.1.0 on branch `feat/grid`. It has 12 seeds per cell, a budget of 100,000 steps and the decay edge near ηλ ≈ 7e-4.
- **Task.**
  - Export the broken-line figure as a PDF from that pipeline. Overleaf and Tectonic cannot run PythonTeX.
  - Quote the break ψ = 7.0e-4 [6.8e-4, 7.3e-4] and the outcomes at the four design levels, after rechecking them against the databases.
- **Status:** carry over.

### E15. Prediction model settings (Appendix E)
- Write down the grouped cross-validation, the search grids (XGBoost: depth 1 to 4, learning rate 0.03 or 0.1, minimum child weight 1, 5 or 10, early stopping after 50 rounds), the kernel ridge selection, the GRU (16 hidden units, learning rate 0.01, weight decay 1e-4, early stopping up to 800 epochs) and all seeds. Copy them from notebooks 3 and 4. **Status:** carry over.

### E16. Sensitivity to the grokking level (Appendix F)
- Repeat E3 and E6 at levels 0.80 and 0.90. **Status:** new.

### E17. Test the premise of the rotation clock (§4.5, option B)
- **Question.** Does the early rate of rotation fall with width as 1/N, as the order of T_0 in Lewkowycz and Gur-Ari suggests? How much of the change of scale does weight decay alone account for?
- **Setup.** All α and N. For the first part, ηλ = 0; for the second, every ηλ > 0. All 5 seeds.
- **Data.** `checkpoints.parquet`: `step`, `S_sum`, `R_sum`. Use `report_runs.csv` for `t_mem`.
- **Method.**
  - Rotation rate. Take the slope of log R_t against log t over a window before memorisation, or R_t at a fixed early step. Fit log rate = c − q log N for each α and compare q with 1. Wide networks memorise within the first hundred steps, so choose the window before looking at the result and say how it was chosen.
  - Scale under weight decay alone. The network is 2-homogeneous, so scaling every weight by c scales its tangent kernel by c². Weight decay with no gradient would therefore give S_t = (1 − ηλ)^(2t) and R_t = 0. This is the group's own derivation; it matches the term −2(k − 1)λΘ_t of the kernel-flow equation with k = 2. Plot the measured S_t against this curve, and report the ratio at memorisation and at grokking.
- **Output.** `E17_premise.pdf`. **Status:** new. Run it first if the group keeps any theory section.

### E18. Can the frozen kernel alone generalise? (§4.5, option C)
- **Question.** Kernel ridge regression with a ridge proportional to the kernel's trace gives the same predictions when the kernel is rescaled, so what a frozen kernel can learn depends only on its shape. Does the frozen tangent kernel start to generalise near the step at which the network groks?
- **Setup.** Start with α = 1, all N, ηλ ∈ {0, 1e-5, 1e-4}, and 5 seeds. Extend to all cells if it works.
- **Data.** `runs/<run_id>_weights.npz` at the 48 saved steps. The splits come from data seed 42 + seed, through `RepoducedCode.make_modular_addition_dataset`.
- **Method.**
  - Rebuild the sum-of-logits kernel on all p² pairs at each saved step with `ntk_lib.sum_kernel`.
  - Predict the test pairs with K_te,tr (K_tr,tr + r I)^(−1) Y_tr, with r equal to 1e-3 times the mean diagonal of K_tr,tr. Use one kernel for every output column, as the pseudo-NTK of Mohamadi, Bae and Sutherland does. Score by test accuracy, taking the largest output.
  - First check the kernel at step 0. If it already generalises, report that before anything else, since it bears on the lazy-to-rich reading.
  - Find the first saved step at which the frozen kernel reaches test accuracy 0.95. Compare it with the network's grokking step, and with R_t and S_t at that step.
- **Caution.**
  - The saved steps are up to 20,000 apart late in training, so the frozen-kernel event is coarse.
  - The column `krr_full_test_acc` in `checkpoints.parquet` does a similar regression with the older kernel of the feature logger. It can serve as a first look, but this item must use the report's kernel.
- **Output.** `E18_frozen_kernel.pdf`. **Status:** new. Run it only if the group chooses option C.

## Numbers quoted in the text

| ID | Where | What | Planning value or source |
|---|---|---|---|
| N1 | Table 2 | Total compute, and whether it is wall-clock or CPU time, and on which machine. | `manifest.json` gives `total_seconds` = 68,056. |
| N2 | §1, §4.1 | Share of D² carried by scale at memorisation, at grokking and at the end, as medians over seeds in the baseline cells. | Golden §4.1 gives 70%, 92% and 94% on the other pipeline. Recompute on `data/` from S and R with Eq. 4 of the report. |
| N3 | §3.4 | Gap between an event and the checkpoint before it, as a share of the event step. | At α = 1: at most 9.7% of the grokking step and 10.9% of the memorisation step (`tier2_reproduction.md`). Extend to all α. |
| N4 | §3.4 | Range of the test loss after the test accuracy is perfect, in the baseline cell. | Golden §4.2.1 gives 0.017 to 0.029 on the other pipeline. Recompute. |
| N5 | §4.1 | Median and range of the memorisation and grokking steps by α at N = 100 and ηλ = 0, and the seeds that grok. | At α = 1 the median T95 is 23,000 with 5 of 5 grokking (`tier2_reproduction.md` §7). |
| N6 | §4.2 | Median A at step 0 and at grokking over grokked runs. | 0.398 and 0.406. |
| N7 | §4.2 | Median R and S at grokking by width. | R: 0.108, 0.063, 0.042, 0.027, 0.022, 0.019. S: 4.11, 2.52, 1.53, 1.06, 0.87, 0.58. Values are for N = 50 to 1600, over all α and ηλ. |
| N8 | §4.2 | Cells whose R or A enters the band of the grokking runs without grokking. | On old data: N = 1600 at 3e-4 and α = 2 at 3e-4. Recheck. |
| N9 | §4.3 | Share of the spread of log R and of S explained by each setting alone. | Notebook 4 Part 2: width explains 0.655 of log R; S is shared (α 0.28, N 0.22, ηλ 0.26). |

## Open checks before any number is final

1. Recompute the placebo test (E3) with 2,000 permutations at full checkpoint resolution, and add the memorisation column.
2. Redo the kernel-at-event table by width with A_sum. Part D of `tier2_consistency.py` used the full-logit A.
3. Refit E6 with intervals that resample cells.
4. Decide whether E7 is its own figure or part of E4.
5. Read Kumar et al. Appendix 8.3 for E12.
6. Confirm what `total_seconds` in the manifest measures (N1).
7. Decide the direction of Tier 4' once E3, E6 and E7 are final. If any theory section stays, run E17 first.

## Approach for the clean repository (later, not built yet)

The aim is a repository a reader can clone to rebuild every figure and number in the report from data, with no notebooks in the path.

1. **One script per item.** `report_v2/code/E01_baseline.py` to `E16_levels.py` each read from `data/`, or from the slim extract in step 4. Each writes:
   - its figure to `report_v2/overleaf/figures/`;
   - its numbers to a JSON file in `report_v2/code/out/`.

   A script takes no arguments and fixes its seeds at the top.
2. **Numbers as macros.** `report_v2/code/make_numbers.py` collects the JSON files into `report_v2/overleaf/numbers.tex`. That file holds `\newcommand` macros such as `\newcommand{\RcvNhundred}{0.20}`. `main.tex` then uses `\input{numbers}` and the macros in place of `\pending{...}`. No number is typed by hand, and the Overleaf folder stays self-contained.
3. **Shared code.** `report_v2/code/common.py` holds:
   - the loaders;
   - the event definitions;
   - the column names;
   - the cell grouping;
   - one colour per statistic (S, R, A) and one per width, used by every figure.
4. **Slim data.** `report_v2/code/extract.py` writes a small frozen extract: `report_runs.csv`, plus the checkpoint columns `run_id`, `step`, `train_acc`, `test_acc`, `test_loss`, `S_sum`, `R_sum` and `A_sum`, and `report_kernel.parquet`. The extract is small enough to publish, while the full `data/` is 11 GB. The README says that `repoduced-code/dataset_sweep.py` regenerates the full data, and how long it takes.
5. **One entry point.** `report_v2/code/make_all.py` runs the extract check, every script, and `make_numbers.py`, in that order. An environment file pins the package versions that were used.
6. **README.** It gives a table of item, figure or table in the report, script, inputs, and run time. The table can start as a copy of the list above.
7. **Publication.** When the report is final, copy into a new public repository:
   - `report_v2/code/`;
   - the slim data;
   - the parts of `repoduced-code/ntk_lib.py` and `RepoducedCode.py` that training and the kernel need;
   - `dataset_sweep.py`;
   - the README and a licence.

   Run `make_all.py` there from a clean environment, and confirm that the figures match the report.
