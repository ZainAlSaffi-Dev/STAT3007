# Reproducing Tier 2 from `data/` under the golden report of 4 October 2026

This file says how to rebuild Tier 2, the width arm, from the dataset in `data/` with the definitions of the
golden report of 4 October 2026 (`docs/report_golden.pdf`, 21 pages). It replaces the
recipe of `tier2_sweep.py` and `tier2_width_decay.ipynb`, whose runs and definitions are superseded (see the
entry of 4 October in `tier_2.md`). The text for the report is drafted in `docs/tier2_additions.tex`, and the
numbers produced here fill its placeholders.

Nothing in this file has been fitted yet. The checks in section 4 were run on 4 October 2026 and their
results are given.

## 1. What `data/` holds

`data/` is the full output of `dataset_sweep.py` from the rerun of 3 October 2026, with a separate split for
each seed. Its tables are byte for byte the same as those in `model_fitting/data/`. It adds the files that
`model_fitting/data/` leaves out, which are the weights, the kernels and the long checkpoint table.

| Path | Contents |
|---|---|
| `manifest.json` | The design, the fixed settings, the versions and the commit (0c2c9fa, dirty). |
| `validation.json` | The checks of `dataset_sweep.py validate`. All passed. |
| `runs.parquet`, `runs.csv` | One row per run under the older definitions. |
| `report_runs.csv` | One row per run with the report's events (§3.5) and S, R, D at each event. |
| `checkpoints.parquet` | One row per run and checkpoint, 307 checkpoints per run, 146 columns. |
| `report_kernel.parquet` | S_t, R_t, D_t and the older A_t at the 48 steps whose weights were saved. |
| `runs/<run_id>_weights.npz` | `steps` (48), `W1` (48, N, 46) and `W2` (48, 23, N) of each run. |
| `runs/<run_id>.json`, `_meta.json`, `.parquet`, `_kernels.npz` | The history, the configuration and the probe kernels of each run. |
| `curves/` | Every checkpoint column of the seed 0 runs. |
| `runs_seed42/`, `seed42_tables/` | The old runs with one split for every seed. Do not use them. |
| `pilot/`, `check/`, `logs/` | The timing pilot, the logger check and the sweep logs. |

The design is α in {0.5, 1, 2}, N in {50, 100, 200, 400, 800, 1600}, ηλ in {0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
1e-3, 3e-3}, and seeds 0 to 4. That makes 720 runs, each of 200,000 steps. **Tier 2 uses the α = 1 slice:
240 runs, 6 widths, 8 decays and 5 seeds.** Every one of them finished with status `ok`.

`data/` is 11 GB and is not listed in `.gitignore`. It must not be committed.

## 2. How the data compare with the golden report

### Settings that match

The data and the report agree on the task (p = 23, training fraction 0.9) and the network (Eq. 2, no biases,
ReLU, NTK factor N^{-1/2}). They also agree on the centred predictor with η = η0/α² and η0 = 100 (Eq. 3), the
loss averaged over the outputs and the training pairs, the decoupled update θ ← (1 − ηλ)θ − η∇L (Eq. 4), and
each seed drawing its own split. Memorisation is the first step at training accuracy 0.99. Grokking is the
first step at test accuracy 0.80, 0.90 or 0.95, counted from step 0 and censored at the budget (§3.5).

### Differences, and what to do about each

| Item | Golden report | `data/` | What to do |
|---|---|---|---|
| Seeds | 12 per cell (§4.1, §4.2). | 5 per cell. | State it in the report. More seeds would need new runs. |
| Budget | 100,000 steps. | 200,000 steps. | Use 200,000 (decision by Jasper Chong). Compare a width cell with the other arms by treating events after 100,000 as censored. |
| Split and initial weights | `grid/src/deeplearning/training.py` on `feat/grid`. The split is `torch.randperm(p*p)` from a generator seeded with the seed, and the weights come from a second generator. | `RepoducedCode.make_modular_addition_dataset` with `torch.manual_seed(42 + seed)`, and the weights from `torch.manual_seed(seed)`. | Seed k here is a different draw from seed k in the report. Compare cells by their distributions, not seed by seed. |
| Event resolution | Checked at every step. | Checked at the 307 checkpoints: the log grid and then every 1,000 steps. Each event lies in `(t_*_prev, t_*]`. | Report the resolution. At α = 1 the gap before an observed event is at most 9.7% of its grokking step and 10.9% of its memorisation step. |
| S_t, R_t, D_t | Sum kernel on the test pairs (Eq. 5, 6). | `S_sum`, `R_sum`, `D_sum` at every checkpoint, and in `report_kernel.parquet` at 48 steps. | Use them as they are. |
| A_t | CKA of the tangent kernel of **all p logits** on the test pairs, an np × np matrix, against yyᵀ with y the stacked one-hot targets (§3.3). | **Missing.** `A_sum` is the older report's A_t, on the sum kernel. `A_full` is a centred alignment on all p² pairs. Neither is the report's A_t. | Compute it from the weights (section 3.1). |
| Spectral measures | erank, T_{t,2(p−1)} and the target power, on the centred sum kernel of all p² pairs (Eq. 8). | **Missing in the report's form.** The `eig_*` columns use the old probe kernel. | Compute them from the weights (section 3.2). |
| Kernel at the event | Evaluated at every event step (§3.2). | S, R and D at the event checkpoint (`S_t_at_*` and the others in `report_runs.csv`). Weights only at 48 steps. | Take A at the last saved weight step at or before the event, and state it. Section 6 gives the exact alternative. |
| Phases | At the last step, against the threshold 0.9 (§3.5, §4.2.2). | Not recorded. | Compute them from `checkpoints.parquet` (section 3.3). |

## 3. Quantities to compute from `data/`

The two kernel functions below were tested on 4 October 2026 (section 4). They work on float64 numpy arrays
from `runs/<run_id>_weights.npz`. They belong next to `sum_kernel` and `report_statistics` in `ntk_lib.py`.

### 3.1 The report's A_t

The tangent kernel of the logits f_c = W2[c] · relu(W1 x / √D) / √N has a closed form. For pairs i, j and
logits c, c', with Z = relu(h) and M = 1[h > 0], it is

Θ[(i,c),(j,c')] = δ_cc' Z_i·Z_j / N + Σ_k W2[c,k] W2[c',k] M_ik M_jk (x_i·x_j) / (D N).

The centred predictor scales every gradient by α. CKA does not change under that scaling, so the kernel of f
gives the same A_t.

```python
def logit_kernel(W1, W2, X):
    """Tangent kernel of all p logits on the rows of X, an (n p) x (n p) matrix, pair-major."""
    D, N = X.shape[1], W1.shape[0]; p = W2.shape[0]; n = len(X)
    h = X @ W1.T / np.sqrt(D); Z, M = np.maximum(h, 0), (h > 0).astype(float)
    G = (M[:, None, :] * W2[None, :, :]).reshape(n * p, N)
    XX = np.repeat(np.repeat(X @ X.T, p, 0), p, 1)
    return np.kron(Z @ Z.T / N, np.eye(p)) + (G @ G.T) * XX / (D * N)

def cka(K, y):
    """Centred kernel alignment of K with y y^T (golden report Eq. 6)."""
    r = len(K); C = np.eye(r) - 1 / r
    Kc, G = C @ K @ C, C @ np.outer(y, y) @ C
    return np.sum(Kc * G) / (np.linalg.norm(Kc) * np.linalg.norm(G))

# For a run with data seed 42 + seed:
_, (X_test, Y_test) = make_modular_addition_dataset(p=23, train_fraction=0.9, seed=42 + seed)
A_t = cka(logit_kernel(W1[k], W2[k], X_test), Y_test.reshape(-1))   # pair-major, as logit_kernel
```

With 53 test pairs the kernel is 1,219 × 1,219. At N = 1600 one evaluation is a single matrix product of
size 1,219 × 1,600, so all 240 runs at 48 steps take a few minutes.

### 3.2 The spectral measures of Eq. 8

These use the sum kernel `ntk_lib.sum_kernel` on all p² = 529 pairs, centred with C = I − 11ᵀ/n.

```python
X_all, Y_all = all 529 pairs in the order (a, b) for a in range(p) for b in range(p)
w, V = np.linalg.eigh(C @ sum_kernel(W1[k], W2[k], X_all) @ C); w, V = w[::-1], V[:, ::-1]
s = np.clip(w, 0, None); mu = s[s > 0] / s.sum()
erank = np.exp(-(mu * np.log(mu)).sum())          # Roy and Vetterli (2007), 0 log 0 = 0
T = s[:2*(p-1)].sum() / s.sum()                   # trace ratio T_{t, 2(p-1)}
P = ((V.T @ C @ Y_all) ** 2).sum(1)               # target power P_{t,i}, summed over outputs
share = P[:2*(p-1)].sum() / P.sum()               # share on the 2(p-1) leading eigenvectors
```

The centred kernel is positive semi-definite, so its eigenvalues are its singular values, and `s` serves for
both (clipped at 0 against rounding).

### 3.3 Phases

These follow `grid/src/deeplearning/training.py` on `feat/grid` with the threshold 0.9. Take the last
checkpoint of each run. The phase is **grokking** when the final training and test accuracies are both above
0.9. It is **memorisation** when only the final training accuracy is above 0.9. It is **forgetting** when the
final training accuracy is at or below 0.9 but the largest training accuracy over all checkpoints was above
0.9. Every other run is **no fitting**. Use `train_acc` and `test_acc` from `checkpoints.parquet`.

## 4. Checks run on 4 October 2026

- **Closed form against autograd.** `logit_kernel` matches the kernel from `torch.autograd` on six test pairs
  of `ntk_N100_a1_wd0_s0` at weight step 20, with a largest relative difference of 5.7e-16.
- **A_t on the cell of the pilot settings.** This is N = 100, α = 1, ηλ = 0, with 5 seeds. A_0 is 0.0133 to
  0.0145, against 0.013 to 0.016 in Table 2 of the report. At step 100,000, A_t is 0.0156 to 0.0166, against
  0.016 to 0.018.
- **Spectral measures on the same cell, from step 0 to step 100,000.** erank goes from 116 to 124 at
  step 0 and 49 to 54 at step 100,000, against 110 to 130 and 49 to 54 in Table 2. T_{t,2(p−1)} goes from
  0.76 to 0.78 and then 0.92 to 0.93, against 0.76 to 0.78 and
  0.92 to 0.94. The target share goes from 0.0016 to 0.0020 and then 0.0177 to 0.0274, against 0.0016 to 0.0021 and
  0.019 to 0.026. Two seeds fall just outside the report's range at step 100,000. With 5 seeds on other splits,
  that is expected.
- **Events on the same cell.** Memorisation comes at 2,692 to 2,983 steps, which is checkpoint resolution,
  against 1,900 to 2,800 in Table 1. Grokking at level 0.95 has a median of 23,000 (15,395 to 34,000),
  against 31,000 (18,000 to 57,000) over 12 seeds in Table 1.
- **The rest of the data.** `validation.json` passed: the closed-form kernel, the identities, and an exact
  reproduction of 16 saved runs in their training fields.

Rerun these checks first whenever the code or the data change.

`tier2_consistency.py` compares the data with the Tier 2 runs of 26 September 2026. It takes about 5 minutes
and writes `results/tier2_consistency.json`. On 4 October it found the following, and `tier_2.md` has the
details.

- Seed 0 trains identically in both sets.
- The old definitions reproduce the old head fit on the new data: a = +0.28 and b = -0.50, against +0.31 and
  -0.49 before.
- Under the report's definitions, the exponents depend on whether the cells at the decay edge enter the fit.

## 5. Steps

All Python runs in the conda environment tara-env (`/opt/miniconda3/envs/tara-env/bin/python`). It has
pandas 2.3.3, pyarrow 23.0.1 and scipy 1.17.1, but **not lifelines**. Install lifelines first with
`/opt/miniconda3/envs/tara-env/bin/pip install lifelines`. The fits below follow
`grid/workflow/scripts/fits.py` on `feat/grid`, which is the code behind the report's tables. Reuse its
choices so that the width arm sits beside the other arms.

1. **Build the Tier 2 tables.** This is a new script, `tier2_tables.py`, which writes to `data/tier2/`.
   - `runs`: the α = 1 rows of `report_runs.csv`, plus the phase (section 3.3) and A_t at the last saved
     weight step at or before each event (section 3.1).
   - `kernel`: one row per run and saved weight step, with `S_t`, `R_t`, `D_t` from `report_kernel.parquet`,
     the report's `A_t`, and the spectral measures.
   - `intervals`: the long table for the Cox model, one row per run and interval `(start, stop]` between
     saved weight steps, with the covariates held from `start`, and `occurred` set on the interval that holds
     the event.
2. **Run the checks of section 4.** Also check Eq. 7, D² = S² − 2S(1 − R) + 1, on every row of `kernel`.
3. **Kaplan-Meier per cell (Eq. 9).** Use `KaplanMeierFitter` for every cell, event and level, and take the
   median with its interval from `median_survival_times`, as `fits.py` does.
4. **AFT width arm at ηλ = 0 (Eq. 10).** Follow the report's α arm: use indicators of N against N = 100,
   fitted to the widths with at least one event at that level. In these data, N ≥ 200 has no event at any
   grokking level within 200,000 steps, so the arm compares only N = 50 with N = 100. The widths without
   events are reported through their Kaplan-Meier estimates and their phases. A slope in ln N would be
   misleading here, because the grokking time falls from N = 50 to N = 100 and then no wider cell groks.
5. **AFT crossed arm (Eq. 10).** Use the cells with ηλ in {1e-5, 3e-5, 1e-4}, where every seed at every
   width groks at level 0.95. The covariates are ln N, ln ηλ and their product, with both logarithms centred
   at the middle of the grid. Add a check that includes 3e-4. I chose this range after seeing the table in
   section 7, and the report must say so.
6. **The lower break at each width.** At N ≥ 200 the time without decay is censored, while at 1e-5 it is
   about 100,000 steps. This is the break of §3.6, where the time returns to its value without decay. Fit
   the broken line of Muggeo (2003) per width with `grid/workflow/scripts/broken_line.R`, which needs R with
   `segmented` and `censReg`. This step is optional, and it waits on a group decision.
7. **Aalen (Eq. 12).** Use every α = 1 cell, ηλ = 0 included, with the covariates ηλ and ln N. Port
   `cumulative_regression` from `grid/src/deeplearning/survival.py` on `feat/grid`, and take the BCa interval
   from `scipy.stats.bootstrap` over seeds, each seed drawn with all its runs, as `fits.py` does. The
   estimator stops where the design loses rank.
8. **Time-varying Cox (Eq. 13, 14).** Use `CoxTimeVaryingFitter` on `intervals` with (S_t, R_t, A_t), and a
   second time with ln N added. Ties use Efron's method, which is the lifelines default.
9. **Write it up.** Put the analysis in a notebook, `tier2_v2.ipynb`. Fill the placeholders in
   `docs/tier2_additions.tex` and rebuild it with `tectonic tier2_additions.tex` from `docs/`. Execute the
   notebook twice and confirm that every number is the same.

## 6. Known limits

- **A_t at the event is approximate.** The weights are saved only at 48 steps. After step 9,000 they are 2,000
  to 20,000 steps apart, so A_t at an event is taken up to 20,000 steps early, and the Cox covariates are held
  over the same gaps. Training is deterministic: `validation.json` reproduced 16 saved runs exactly. The exact
  fix is therefore to retrain the α = 1 runs with weights saved at every checkpoint, through
  `dataset_sweep.py` with `ntk_lib.train_run(on_checkpoint=...)`. That costs about 30 to 40 minutes per run
  at N = 1600. It should not be started without the group's agreement.
- **5 seeds, not 12.** The Kaplan-Meier intervals and the seed bootstrap are wider than those of the other
  arms.
- **The width levels and the decay levels were not chosen by the design of §3.6.**
- **The bootstrap intervals of the AFT condition on the cells.** Resampling seeds or resampling runs within
  cells keeps every cell in every resample. Both therefore leave out the misfit of the cells about the fitted
  plane. In the consistency check, the residual sigma of the crossed fit was 0.38, while the spread between
  seeds within a cell was 0.02 to 0.19, and the two bootstraps gave almost the same narrow intervals. State
  this with every interval, or add an interval that allows for misfit at the cell level.
- **The seed draws differ from the report's pipeline** (section 2), so the cell with the pilot settings agrees
  with the report's pilot cell in distribution only.

## 7. First look at the α = 1 slice

Each entry gives the seeds that reach test accuracy 0.95 out of 5, the median grokking step at level 0.95
when at least 3 seeds reach it, and the phases of the 5 seeds at threshold 0.9. The phase letters are G for
grokking, M for memorisation, F for forgetting and N for no fitting. These come straight from `data/`
under the report's definitions of events and phases. They involve no kernel statistic.

| N | 0 | 3e-6 | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|---|---|---|---|
| 50 | 5/5, 33,000, GGGGG | 5/5, 30,000, GGGGG | 5/5, 27,000, GGGGG | 5/5, 24,000, GGGGG | 5/5, 18,900, GGGGG | 5/5, 15,000, GGGGG | 0/5, MMMMM | 0/5, NNNNN |
| 100 | 5/5, 23,000, GGGGG | 5/5, 23,000, GGGGG | 5/5, 20,941, GGGGG | 5/5, 17,000, GGGGG | 5/5, 10,000, GGGGG | 5/5, 9,218, GGGGG | 0/5, MMMMM | 0/5, NNNNN |
| 200 | 0/5, MMMMM | 0/5, MMMMM | 5/5, 106,000, GGGGG | 5/5, 27,000, GGGGG | 5/5, 11,000, GGGGG | 5/5, 10,000, GGGGG | 0/5, NNNNN | 0/5, NNNNN |
| 400 | 0/5, MMMMM | 0/5, MMMMM | 5/5, 82,000, GGGGG | 5/5, 27,000, GGGGG | 5/5, 12,000, GGGGG | 5/5, 13,894, GGGGG | 0/5, NNNNN | 0/5, NNNNN |
| 800 | 0/5, MMMMM | 0/5, MMMMM | 5/5, 88,000, GGGGG | 5/5, 30,000, GGGGG | 5/5, 15,000, GGGGG | 0/5, GGGGM | 0/5, NNNNN | 0/5, NNNNN |
| 1600 | 0/5, MMMMM | 0/5, MMMMM | 5/5, 99,000, GGGGG | 5/5, 37,000, GGGGG | 5/5, 20,000, GGGGG | 0/5, FFFFF | 0/5, NNNNN | 0/5, NNNNN |

Three patterns stand out, and the fits have to confirm them.

- Without decay, the grokking time falls from N = 50 to N = 100. No wider network reaches level 0.80 within
  200,000 steps.
- Inside the range of the crossed arm, the time falls with ηλ at every width, and it falls faster at the wide
  widths.
- The edge of the decay axis moves down as N grows. At 3e-4, N = 800 ends above 0.9 test accuracy on 4 seeds
  without reaching 0.95, and N = 1600 forgets. At 1e-3 every width fails.
