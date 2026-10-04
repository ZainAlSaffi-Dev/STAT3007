# Changelog: the golden report against the older documents and data

This file records every difference between the golden report (`docs/report_golden.pdf`) and the older material. The older material is the proposal (`docs/DeepLearningProposal.pdf`, checked on 31 August 2026), the report skeleton (`docs/report.tex`), and the feature dataset built on 29 September 2026 (commit afc2ab2). The golden report wins wherever they differ, as `CLAUDE.md` states.

The golden report is not finished. Its abstract, grid section (§4.2), conclusion and contribution statement are still placeholders. Where it says nothing about a setting, this file says so instead of guessing.

The last column of each table gives the state of the regenerated dataset of 3 October 2026. That dataset is described at the end.

## 1. Task, network and training

| Item | Proposal and `report.tex` | Old dataset (29 Sep) | Golden report | New dataset (3 Oct) |
|---|---|---|---|---|
| Task | Modular addition, p = 23, 90% of the p² pairs for training. | Same. | Same (§3.1, §4.1). | Same. |
| Train and test split | Not stated. `report.tex` fixes the probe "across all runs and seeds". | Data seed 42 for every run, so all seeds share one split. | Each seed draws its own split and its own initial weights (§3.1). | Data seed 42 + seed. Each seed has its own split, and seed 0 keeps the old split. |
| Network | One hidden layer, no biases, ReLU, weights from N(0, 1), NTK parameterisation with 1/√N. | Same. | Same (Eq. 2). The report also names the mean-field factor 1/N of Gromov (2023), but the pilot uses the NTK form. | Same. |
| Centred predictor | f̃ = α[f(θ) − f(θ₀)] with η = η₀/α². | Same. | Same (Eq. 3). | Same. |
| Loss | Mean squared error. | `nn.MSELoss`, which averages over the p outputs and the training pairs. | Squared error averaged over the p outputs and the training pairs, written out in §3.1. | Same. The code already matched. |
| Optimiser | Full-batch gradient descent, η₀ = 100. | Same. | Same. | Same. |
| Weight decay | Swept as ηλ, with coupled L2. | `torch.optim.SGD` with weight_decay = ηλ/η. This gives θ ← (1 − ηλ)θ − η∇L. | Written as decoupled weight decay, θ ← (1 − ηλ)θ − η∇L (Eq. 4). The report notes that coupled and decoupled decay coincide for plain gradient descent. | Same update. The code already matched. |
| Step budget | Not stated. | 200,000 steps for every run. | 100,000 steps for the pilot cell (§4.1). The grid budget is not stated. | 200,000 steps. Cut at 100,000 to compare with the pilot. |
| Seeds | 5 per cell, 10 on the clamp arm. | 5 per cell. | 5 for the pilot cell. | 5 per cell. |

## 2. The kernel

| Item | Proposal and `report.tex` | Old dataset (29 Sep) | Golden report | New dataset (3 Oct) |
|---|---|---|---|---|
| Scalar output | `report.tex` uses the trace kernel, the sum over c of ⟨∇f_c, ∇f_c⟩, which has no cross terms between outputs. The proposal says "sum-over-logits". | The trace kernel. | The kernel of g = (Σ_c f_c)/√p, the pseudo-NTK of Mohamadi, Bae and Sutherland (2022). It has the cross terms (Eq. 5). The first logit is a second output. | Columns `S_sum`, `R_sum`, `D_sum` and `A_sum` use the kernel of g. The old trace-kernel columns are still there under their old names. |
| Where it is evaluated | A fixed probe of 256 pairs: all 53 test pairs and 203 training pairs. | The same mixed probe, plus all p² pairs for some columns. | All p² pairs. S, R, D and A use the submatrix on the 53 test pairs of each seed (§3.3). Spectral measures use all p² pairs (§3.4). | S, R, D and A are on each run's own test pairs. |
| When it is evaluated | On a log-spaced grid of steps. | At step 0, every 1,000 steps, and 120 log-spaced steps, about 307 checkpoints. | At initialisation, on a log-spaced grid, and at every event step (§3.2). | At every checkpoint. So the statistics are available at each event step, to the resolution of the checkpoint grid. |

## 3. Kernel statistics

| Statistic | Proposal and `report.tex` | Old dataset (29 Sep) | Golden report (Eq. 6) | New dataset (3 Oct) |
|---|---|---|---|---|
| Scale S_t | log(‖K̃_t‖/‖K̃_0‖) on the centred kernel. | `S_c`, the log ratio on the centred probe kernel. | ‖K_t‖/‖K_0‖ on the uncentred test-pair kernel. It is a plain ratio, so 1 means no change. | `S_sum`, the report form. |
| Shape change R_t | 1 − cosine of the centred, normalised kernels. | `R_c`, on the centred probe kernel. | 1 − uncentred cosine of K_t and K_0. This is the kernel distance of Fort et al. (2020). | `R_sum`, the report form. |
| Variation D_t | Named only as the "standard statistic" ‖K_t − K_0‖/‖K_0‖. | `D`, on the centred probe kernel. | ‖K_t − K_0‖/‖K_0‖, after Geiger et al. (2020). It is a named statistic. | `D_sum`, the report form. |
| Identity | ‖K_t − K_0‖²/‖K_0‖² = e^{2S} + 1 − 2e^{S}(1 − R), with S as a log. | Checked in that log form. | D² = S² − 2S(1 − R) + 1, with S as a ratio (Eq. 7, proof in §A.1). | `report_kernel.py` checks Eq. 7. |
| Alignment A_t | Centred alignment of the probe kernel with YYᵀ, after Cortes et al. (2012). The uncentred form is also reported, for comparison with Kumar et al. | `A_t`, on the mixed probe. A smoke test found that it leaks, because the probe mixes training and test pairs. | CKA of the test-pair kernel with YYᵀ, after Cortes, Mohri and Rostamizadeh (2024). The report gives no uncentred variant. | `A_sum`, the report form. |
| Spectral measures | Not in the proposal. | `eig_*` columns on the centred probe kernel, with the group's own definitions. | Effective rank of Roy and Vetterli (2007), trace ratios T_{t,k}, and the target power along each eigenvector. All are on the centred kernel of all p² pairs (Eq. 8). | **Not computed in the report's form.** See section 8. |
| y⊤K⁻¹y | Logged at each checkpoint. | `yKy`. | Not in the report. | Still logged as `yKy`. |

## 4. Event times

| Item | Proposal and `report.tex` | Old dataset (29 Sep) | Golden report (§3.5) | New dataset (3 Oct) |
|---|---|---|---|---|
| What is timed | t_grok, the step gap from the training loss crossing τ_train to the test loss crossing τ_test. The thresholds are pre-registered. | Loss gaps at τ ∈ {3e-2, 2e-2, 1e-2, 3e-3, 1e-3}, and accuracy gaps at test levels {0.95, 0.99, 1.0}, measured from memorisation. | Separate event times, all counted from step 0. | `report_runs.csv` has the report's events. `runs.csv` keeps the old ones. |
| Memorisation | Not defined. | First step with training accuracy 1.0. | First step with training accuracy 0.99, after Khanh et al. (2026). | `t_mem` at 0.99. |
| Grokking | Defined by the test loss threshold. | Test accuracy 0.95, 0.99 or 1.0. | First step with test accuracy at 0.80, 0.90 or 0.95, after Khanh et al. (2026). | `t_grok80`, `t_grok90` and `t_grok95`. |
| Censoring | A run that has not grokked is right censored, not dropped. | Censored at the last step. A run that never memorised had no grokking time. | Censored at the budget. A run keeps its event times whatever its phase. | Censored at the budget. Every run has every event. |
| Phases | Not defined. | Not recorded. | Grokking, memorisation, forgetting or no fitting at the last step, after Pracher et al. (2026). The threshold is not stated. | **Not added**, because the threshold is not stated. |
| Kernel at the event | Not defined. | Values at the memorisation step and at the generalisation step (test accuracy 1.0). | Kernel evaluated at every event step. | `S_t_at_*`, `R_t_at_*`, `D_t_at_*` and `A_t_at_*` for each event. |

## 5. Statistical analysis

| Item | Proposal and `report.tex` | Golden report |
|---|---|---|
| Head claim | Fit log t_grok = c + a log N + b log(ηλ) on a crossed grid of N and ηλ, with bootstrap intervals and an interaction term. `report.tex` adds 1/t_grok = c₀ + c₁ηλ + c₂ log N, so that the cells with ηλ = 0 can enter. | The question is which of scale and shape moves before and while the network groks, and whether the event times follow either (§1). No regression on width appears in the report. |
| Threshold hypothesis | Generalisation starts when A_t crosses a threshold A*, which is shared across cells. t_grok is independent of S_t given A_t. | Not in the report. |
| Survival models | A Tobit model, or a Cox model on log t_grok. | A Kaplan–Meier estimate per cell, with Greenwood intervals and the median event time (Eq. 9). Then three models, fitted with lifelines. |
| Model 1 | Not present. | A log-normal accelerated failure time model (Eq. 10–11). It is fitted twice: to the ηλ = 0 cells with indicators of α, and to the ηλ > 0 cells at the pilot α with the covariate log(ηλ). |
| Model 2 | Not present. | Aalen's additive model on all cells, with indicators of α and ηλ itself, and BCa bootstrap intervals (Eq. 12). |
| Model 3 | Not present. | A time-varying Cox model with Z(t) = (S_t, R_t, A_t), held from each checkpoint to the next, with Efron ties (Eq. 13–14). The covariates are internal, so the model is descriptive. |
| Choice of levels | A list of grids. | An optimal design (§3.6). It uses Berger and Wong (2009) for the straight line and the quadratic, Nelson (1990) for censored information, and segmented regression of Muggeo (2003, 2008) with a break ψ. Park (1978) gives the design {−1, ψ, (1 + ψ)/2, 1} on the rescaled ηλ axis. The guess of ψ comes from Pracher et al. (2026), with a second stage at the estimated break. |

## 6. Experimental design

| Arm | Proposal | `report.tex` | Old and new dataset | Golden report |
|---|---|---|---|---|
| Baseline | N = 100, α = 1, ηλ = 0. | The same, as Tier 0. | The cell N = 100, α = 1, ηλ = 0 is in the grid. | The pilot cell: N = 100, α = 1, ηλ = 0, 100,000 steps, 5 seeds (§4.1, Tables 1 and 2, Figure 1). |
| α levels | 6 levels in an α × ηλ grid at the baseline width. | {0.1, 0.3, 1, 3, 10, 30}. | {0.5, 1, 2}. | Indicators of α for the ηλ = 0 cells. The levels are not stated. |
| ηλ levels | {0, 1e-3, 1e-2, 1e-1} on the width arm. | Tier 1: {0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}. Tier 2: {0, 1e-3, 1e-2, 1e-1}. | {0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}. Tier 1 found that ηλ ≥ 1e-3 collapses the weights at N = 100. | Set by the optimal design of §3.6 around the break ψ. The ηλ > 0 cells use the pilot α only. The levels are not yet stated. |
| Widths | {50, 100, 400, 1600}. | The same. | {50, 100, 200, 400, 800, 1600}. | Not stated. The report has no width arm. |
| Other arms | A clamp arm with ρ ∈ {1, 1.25, 1.5, 1.75, 2}. p ∈ {23, 29, 31} and training fraction ∈ {50, 70, 90}%. A cross-entropy arm. | The same. | None. | None. |
| Size | About 350 runs. | About 380 runs. | 720 runs. | Not stated. |

## 7. References

These entries changed between the older documents and the golden report. The golden report's form is the verified one.

- Truong (2026) and Truong et al. (2026) are now Khanh (2026), arXiv:2606.18465, and Khanh et al. (2026), arXiv:2606.13753.
- Cortes, Mohri and Rostamizadeh (2012), JMLR 13, is now cited as the 2024 arXiv version, arXiv:1203.0550.
- Jacot, Gabriel and Hongler (2018) is now dated 2020, arXiv:1806.07572.
- Lewkowycz and Gur-Ari (2020) is now dated 2021, arXiv:2006.08643.
- Chizat, Oyallon and Bach (2019) is now dated 2020, arXiv:1812.07956.
- The proposal's draft notes cite Kumar et al. Appendix 14 for the weight-decay argument. `CLAUDE.md` records that the arXiv v3 numbers it 13.
- Kornblith et al. (2019) and Tian (2025), which `report.tex` cites, are not in the golden report.
- The golden report adds Baratin et al. (2021), Fort et al. (2020), Geiger et al. (2020), Shan and Bordelon (2022), Mohamadi, Bae and Sutherland (2022), Pracher et al. (2026), Xu, Vardi and Safran (2026), and the statistics references Aalen (1989), Kalbfleisch and Prentice (2002), Nelson (1990), Berger and Wong (2009), Muggeo (2003, 2008, 2017), Park (1978), Efron and Tibshirani (1994) and Davidson-Pilon (2019).

## 8. Open differences

These differences remain after the regeneration of 3 October 2026.

1. **The grid is not in the report.** `CLAUDE.md` gives the head claim as a regression of log grokking time on log width and log(ηλ). The golden report fits ηλ only at the pilot α, and it has no width covariate. The group should decide which one holds before §4.2 is written.
2. **The ηλ levels differ.** The dataset levels were not chosen by the design of §3.6.
3. **Spectral measures are missing.** The effective rank, the trace ratios and the target power of Eq. 8 are not computed in the report's form, which uses the sum kernel on all p² pairs. The `eig_*` columns use the old probe kernel.
4. **No first-logit kernel in the report's form.** The `*_first_logit` columns use the old centred probe-kernel statistics.
5. **Phases are not recorded,** because §3.5 does not give the threshold.
6. **Events have checkpoint resolution.** Between 1,000 and 200,000 steps, checkpoints are about 1,000 steps apart. Each event therefore lies between `t_*_prev` and `t_*`. Table 1 of the report reports memorisation at 2,500 to 2,800, which suggests a finer grid there.

## 9. Data files

The regeneration of 3 October 2026 changed these files.

- `repoduced-code/dataset_sweep.py` gives each seed the data seed 42 + seed, and records `data_seed` in `runs.csv`. Its reproduction check against the saved Tier 0 runs now compares only runs on the same split, which is seed 0.
- `repoduced-code/dataset_features.py` adds `S_sum`, `R_sum`, `D_sum` and `A_sum` at every checkpoint, and `FEATURE_VERSION` is now 2.
- `repoduced-code/ntk_lib.py` holds `sum_kernel` and `report_statistics`. `report_kernel.py` and the logger share them.
- `model_fitting/report_kernel.py` uses each run's own test pairs.
- `model_fitting/report_runs.py` writes `data/report_runs.csv` (one row per run, report events and kernel statistics) and `data/report_kernel.csv` (the 48 saved steps of each run).
- The old dataset is kept. The old per-run files are in `model_fitting/data/runs_seed42/`, and the old tables are in `model_fitting/data/seed42_tables/`.

## 10. Findings that changed with the regenerated data

The sweep was rerun on 3 October 2026 with a separate split for each seed. All 720 runs finished, and every validation check passed. The pilot cell still agrees with Table 2 of the golden report. Its median A_t at step 0 is 0.39 against 0.40, and its median S_t at step 100,000 is 4.07 against 4. Notebooks 1 to 4 were rerun, and their text was rewritten from the new outputs. These findings changed.

- **One subset of notebook 1 has no finite Cox estimate.** Notebook 1 fits the report's time-varying Cox model (Eq. 13 and 14) separately to the cells without weight decay. On that subset alone the fit does not converge whenever R_t is in it. The golden report does not fit Eq. 13 to that subset, and on all runs pooled (notebook 2) the model converges. Notebook 1 notes this next to its table and adds a labelled check with a ridge penalty of 0.01.
- **The kernel no longer predicts within a cell.** In notebook 2, the stratified Cox model gave S_t a clear effect inside a cell (p = 0.001). With a split for each seed, no statistic has an effect inside a cell (joint p = 0.97).
- **The pooled AFT gains nothing from the kernel.** The likelihood ratio test of the kernel on top of the settings went from p = 0.02 to p = 0.34.
- **Interactions matter less among grokked runs.** In notebook 4, the gain in held-out R² from interaction terms fell from 0.08 to 0.02, and kernel ridge fell from 0.69 to 0.55. Interactions still help to predict whether a run groks at all.
- **A_t has much more seed noise.** The within-cell share of the spread of A_t rose from 11% to 44%. A_t is measured on the test pairs, and these now differ between seeds.
- **The history of the kernel helps more.** In notebook 3, a GRU on the history scores 0.768 against 0.705 for XGBoost at the prediction time, a gain of 0.06 (before, 0.02). The settings alone still score 0.780.
