# Tier 2 experimental log

This file records what was decided, checked, run and noticed during Tier 2, in the order it happened. It is
the reference for writing the Tier 2 part of `report.tex`. The analysis itself is in
`tier2_width_decay.ipynb`, and the grid constants and the run driver are in `tier2_sweep.py`.

Tier 2 is the head claim. It fits log t_grok = c + a log N + b log(eta*lambda) on a grid of width N and
weight decay eta*lambda. Under H the width exponent a is positive and b is about zero. Under a weight-decay
clock a is about zero and b is about -1. The report also asks for the interaction term, the residuals, and
the reciprocal form 1/t_grok = c0 + c1 eta*lambda + c2 log N.

Every run in Tier 2 uses the NTK parameterisation. Every width result below should be read with that in mind.

## 4 October 2026: the results below are superseded

The golden report of 4 October 2026 (`report_golden.pdf` at the repository root, 21 pages) changed the
definitions that Tier 2 was run under. No number in the Results and Observations sections below holds under
the new definitions. They are kept as a record of what was run. The reasons are as follows.

- **Split.** Each seed now draws its own split of the pairs (report §3.1). The Tier 2 runs use data seed 42
  for every seed, so only seed 0 still matches.
- **Kernel.** S_t, R_t and D_t are now taken on the kernel of the summed logits divided by √p, on the test
  pairs of each seed (§3.2, §3.3). Tier 2 used the trace kernel on a probe of 256 pairs. S_t is now a plain
  ratio and R_t is uncentred.
- **Alignment.** A_t is now the CKA of the tangent kernel of all p logits on the test pairs, with the stacked
  one-hot targets (§3.3). Its value at initialisation in the pilot cell is 0.015.
- **Events.** Memorisation is now training accuracy 0.99, and grokking is test accuracy 0.80, 0.90 or 0.95,
  counted from step 0 (§3.5). Tier 2 used accuracy 1.0 for both and timed the gap from memorisation.
- **Runs that never memorise.** These are now kept and censored at the budget (§3.5). Tier 2 dropped them
  (open question 7).
- **Models.** The report fits a Kaplan-Meier estimate per cell, a log-normal AFT (Eq. 10), Aalen's additive
  model (Eq. 12) and a time-varying Cox model (Eq. 13). The reciprocal form, the A* placebo test and the
  early-rate test of the Tier 2 notebook have no counterpart in the report.

The width arm is redrafted against the new report in `docs/tier2_additions.tex`. Its numbers are left as
placeholders until the new data run has been checked. The budget for the width arm is 200,000 steps at every
width, which is the budget of that data run (decision by Jasper Chong).

The new report also changes which open questions below still apply. Question 3 (the reciprocal intercept) and
question 4 (its noise model) lapse, because the reciprocal form is replaced by the Aalen model. Question 7 is
settled by the censoring rule above. Question 12 lapses, because the report no longer states that the slower
clock sets the time. Question 5 (the source for a = 1) still stands. The golden report attributes the
order 1/n size of T_0 to Lewkowycz and Gur-Ari (2021), but a = 1 has not been checked against that paper.

## 4 October 2026: consistency check on the new data

**What was run.** `tier2_consistency.py` compares the Tier 2 results of 26 September with the dataset in
`data/` (see `tier2_reproduction.md`). It uses the α = 1 slice: 240 runs, widths {50, 100, 200, 400, 800,
1600}, decays {0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}, 5 seeds, and 200,000 steps. Seeds 1 to 4 have
their own splits. The script ran in tara-env in about 5 minutes and wrote `results/tier2_consistency.json`.
It has four parts.

- **A.** Seed 0 is compared directly with the old runs.
- **B.** The old definitions are applied to the new data.
- **C.** The definitions of the golden report are applied.
- **D.** The kernel at the event is measured by width.

**A. Seed 0 is identical.** Seed 0 has data seed 42 and model seed 0 in both sets. At every step both sets
checked, the training loss, test loss, training accuracy, test accuracy and weight norm agree exactly, with a
relative difference of 0 over the 20 shared cells. The event steps differ only by checkpoint resolution. For
example, N = 1600 at 1e-5 generalises at 127,500 in the old runs and at 128,000 here. The 300,000-step budget
of the old wide runs had no effect at seed 0, because the old no-decay wide runs never generalised.

**B. The old definitions on the new data reproduce the old head result.** This part uses memorisation and
generalisation at accuracy 1.0, the gap from memorisation, runs that never memorise left out, the four old
widths, the old nonzero decays, and the fit and cell bootstrap of `tier2_width_decay.ipynb`.

| Quantity | Old runs (26 Sep) | New data |
|---|---|---|
| Main effects, a | +0.307 [+0.228, +0.382] | +0.284 [+0.216, +0.354] |
| Main effects, b | -0.491 [-0.558, -0.421] | -0.497 [-0.565, -0.424] |
| Residual sigma | 0.603 | 0.553 |
| Interaction term | -0.210 [-0.271, -0.154] | -0.191 [-0.243, -0.140] |
| a and b with the interaction | +0.241, -0.554 | +0.225, -0.555 |
| On t_test, a and b | +0.152, -0.377 | +0.135, -0.398 |
| a within decay 0, 1e-5, 3e-5, 1e-4, 3e-4 | +1.76, +0.61, +0.32, +0.16, -0.21 | +1.71, +0.58, +0.24, +0.21, -0.21 |
| b within N = 50, 100, 400, 1600 | -0.15, -0.38, -0.82, -0.71 | -0.15, -0.46, -0.79, -0.70 |
| Runs that never memorise | 5, all at N = 1600 and 3e-4 | 5, all at N = 1600 and 3e-4 |

Cutting the runs at N ≤ 100 to the old budget of 100,000 steps changes a and b by less than 0.003. The new
splits, the new budget and the coarser checkpoint grid therefore leave the old result in place.

**C. Under the report's definitions, the exponents depend on which cells enter.** The events are now counted
from step 0, every run is kept and censored at 200,000, and the intervals resample seeds. The main fits at
level 0.95 are as follows.

| Cells | a | b | Interaction | Censored |
|---|---|---|---|---|
| 6 widths, 1e-5 to 1e-4 | +0.237 [+0.202, +0.276] | -0.633 [-0.659, -0.604] | -0.162 | 0 of 90 |
| 6 widths, 1e-5 to 3e-4 | +0.432 [+0.400, +0.468] | -0.209 [-0.231, -0.190] | +0.141 | 10 of 120 |
| Old 4 widths, 1e-5 to 3e-4 | +0.381 [+0.353, +0.419] | -0.196 [-0.226, -0.173] | +0.076 | 5 of 80 |
| Old 4 widths, 1e-5 to 3e-4, without N = 1600 at 3e-4 | +0.200 [+0.168, +0.242] | -0.372 [-0.404, -0.345] | -0.171 | 0 of 75 |

At levels 0.80 and 0.90 the same pattern holds. In the range from 1e-5 to 1e-4, a is +0.30 and +0.27, and b is
-0.55 and -0.60.

- **The new censoring rule causes most of the change.** The last row leaves out the cell that never memorises,
  as the old rule did. That gives a = +0.20 and b = -0.37, close to the old t_test fit (a = +0.15,
  b = -0.38), which also counts from step 0. Keeping those 5 runs as censored at 200,000 raises a to +0.38,
  moves b to -0.20, and turns the interaction positive. Counting from step 0, in place of the gap from
  memorisation, accounts for the rest of the difference from the old head result.
- **The 3e-4 column lies at the edge.** Including it makes the time rise again at the wide widths, so a single
  plane in ln N and ln ηλ no longer fits. The residual sigma rises from 0.38 to 0.82. The range from 1e-5 to
  1e-4 is the one where every seed at every width groks. It gives a plane with a residual sigma of 0.38 and a
  negative interaction, as the old fit did.
- **Memorisation moves the other way.** In the range from 1e-5 to 1e-4, a for memorisation is -1.53: wider
  networks memorise much sooner.
- **Width without decay.** Only N = 50 and N = 100 reach any grokking level. N = 50 groks later than N = 100
  by a factor of e^0.39 = 1.5 at level 0.95 (indicator +0.39 [+0.14, +0.72]). None of the 20 runs at N ≥ 200
  reaches level 0.80 within 200,000 steps. The old within-stratum a of +1.76 without decay rested on the same
  censored wide cells. Both say that, without decay, the width effect is not monotone: time falls from 50 to
  100 and then rises past the budget.
- **The wide cells vary less across seeds.** This repeats an old observation. The standard deviation of
  ln t_grok95 within a cell, as a median over 1e-5 to 1e-4, is 0.14, 0.19, 0.05, 0.14, 0.11 and 0.02 from
  N = 50 to N = 1600.

**The intervals are too narrow to read as uncertainty about the law.** The old cell bootstrap and the seed
bootstrap give almost the same intervals here: for b at 0.95 in the first row, [-0.664, -0.602] against
[-0.659, -0.604]. Both keep every cell in every resample, so both measure only the noise between seeds inside
a cell. The residual sigma of the plane, 0.38, is well above that noise, which ranges from 0.02 to 0.19.
Most of the residual is therefore misfit between the cells and the plane, and neither bootstrap includes it.
The same holds for the old intervals. The intervals should be described as conditional on the cells, or
replaced by intervals that allow for misfit at the cell level.

**D. The kernel at the event: the old pattern holds with the new statistics.** R_t is now the uncentred
R of the sum kernel at the event checkpoint. A_t is the CKA of the tangent kernel of all logits on the test
pairs, read at the last saved weight step before the event. That step is a median of 10% of the event time
early, and at most 27%. The table gives means over the runs that grokked at level 0.95, across all decays.

| N | 50 | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|---|
| A_0 | 0.0132 | 0.0139 | 0.0145 | 0.0146 | 0.0148 | 0.0149 |
| A at the event | 0.0151 | 0.0169 | 0.0180 | 0.0181 | 0.0180 | 0.0182 |
| R at the event | 0.116 | 0.061 | 0.038 | 0.022 | 0.017 | 0.017 |
| S at the event | 3.44 | 2.27 | 1.28 | 0.84 | 0.69 | 0.45 |
| Runs | 30 | 30 | 20 | 20 | 15 | 15 |

- **A_0 still rises with width.** The new values go from 0.0132 to 0.0149. The old ones, on the older A, went
  from 0.0845 to 0.0957 at seed 0.
- **A at the event barely changes with width, while R falls.** A at the event has a max/min ratio of 1.42 and
  a spread of 7.9%. R at the event falls 7-fold, from 0.116 to 0.017, with a max/min ratio of 13.7 and a
  spread of 75%. The old values were 1.75 and 11.7% for A and a 27-fold fall for R. The direction is the same,
  and the contrast is smaller because R is now uncentred.
- **S at the event falls with width as well,** from 3.44 to 0.45. At the wide widths the kernel at the event
  is smaller than at initialisation. This fits the old observation that the weight norm falls with width.

**Summary.** The old Tier 2 results reproduce on the new data under the old definitions. Under the report's
definitions, the qualitative picture is the same:

- a is positive, and b is negative and neither 0 nor -1;
- without decay the width effect is not monotone;
- the decay edge falls with width;
- A at the event is nearly flat across width while R falls.

The values of a and b now depend on two choices that the group has to make: whether the cells at the decay
edge enter the fit, and how the intervals treat misfit at the cell level. These results are a check on the
data and are not yet the Tier 2 analysis of `tier2_reproduction.md`. That analysis still needs the
Kaplan-Meier estimates, the Aalen model and the Cox model, and it needs A_t at the exact event steps.

## Design decisions

**26 September 2026. The decay axis is the Tier 1 axis.** The grid uses eta*lambda in {0, 1e-5, 3e-5, 1e-4,
3e-4}. The report's Tier 2 axis is {0, 1e-3, 1e-2, 1e-1}, but at N = 100 and alpha 1 a decay of 1e-3 stops
the network from memorising. The saved run `ntk_N100_a1_wd0.001_s0` ends with train accuracy 0.941 and a
weight norm at 0.71 of its starting value. The Tier 1 axis also lets the 17 saved N = 100, alpha 1 runs serve
as the N = 100 column. The decision was taken by Jasper Chong while planning.

**26 September 2026. The widths are {50, 100, 400, 1600}.** This is the report's list. The report's fix (i)
would drop one width to pay for more decay levels, but compute turned out not to be the limit, so all four
widths are kept. The spacing is uneven, a factor of 2 and then two factors of 4.

**26 September 2026. The step budget depends on width.** Runs at N = 50 and N = 100 last 100,000 steps, as
in Tier 1. Runs at N = 400 and N = 1600 last 300,000 steps. The reason is that under H with a near 1, the
N = 1600 cells without decay would grok near 200,000 steps, so a 100,000-step budget would leave the width
exponent resting mostly on lower bounds. The rule was fixed before any wide run was made. Checkpoints are
every 500 steps at every width.

**26 September 2026. Other fixed settings.** Alpha is 1, the base learning rate is 100, p is 23, the
training fraction is 0.9, the data seed is 42, and the model seeds are 0 to 4. The probe set is the first 256
training pairs. The generalisation event is the first step at test accuracy 1.0, matching `tier1_v2.ipynb`,
and the memorisation event is the first step at train accuracy 1.

**26 September 2026. The environment is the conda environment tara-env.** Every run and every notebook
execution uses `/opt/miniconda3/envs/tara-env/bin/python`. When checked, it had Python 3.11.4, torch 2.10.0,
scipy 1.17.1 and numpy 2.4.6, together with matplotlib, plotly, ipykernel and nbconvert. CLAUDE.md still says
to use uv, and it has not been changed.

**26 September 2026. Pilot seed 0 at the wide widths before the other seeds.** The smoke test showed that
N = 1600 at decay 3e-4 never memorises. Before spending about 13 CPU-hours on the other N = 1600 seeds, seed 0
of every decay level at N = 400 and N = 1600 is run first to find where memorisation fails. The decision on
seeds 1 to 4 at those widths waits for that pattern. The step budget and the decay axis are unchanged for the
pilot. The decision was taken by Jasper Chong.

**26 September 2026. Run all of seeds 1 to 4 at the wide widths, including N = 1600 at 3e-4.** After the
pilot, all 40 remaining runs are run so that the crossed design stays intact. Running the failing cell on four
more seeds means the report can say whether the cell fails, rather than that one seed failed. The decision was
taken by Jasper Chong.

**26 September 2026. The head numbers come from the main-effects fit.** The report states its predictions
for the main-effects exponents a and b, and it also asks for the interaction term. The notebook reports the
main-effects fit as the head result and the interaction fit as a check on it. The group has not yet
confirmed this choice.

## Deviations from the proposal and the report

| Item | Proposal or report | What Tier 2 does |
|---|---|---|
| Decay axis | Report: eta*lambda in {0, 1e-3, 1e-2, 1e-1}, with fix (i) refining the nonzero part to {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}. Proposal: from 0 to 1e-1. | {0, 1e-5, 3e-5, 1e-4, 3e-4}, for the reason given above. The four nonzero levels span 1.5 decades, not 2. |
| Widths | Report fix (i): drop one of {50, 100, 400, 1600}. | All four kept. |
| Step budget | Not stated for Tier 2. | 100,000 steps for N of 50 and 100, and 300,000 steps for N of 400 and 1600. |
| Event definition | Proposal: loss thresholds. | Accuracy events, as in Tiers 0 and 1. The loss definition is kept as a sensitivity check. |
| Seeds | Five per cell. | Five per cell. The N = 100 cells with nonzero decay had three saved seeds, so seeds 3 and 4 are new. |

## Source checks

### Kim (2026), arXiv:2607.23967v1, checked 26 September 2026

The PDF of v1 was read in full. The title is "Grokking on the Weight-Decay Clock: A Rate Hierarchy from
Softly Broken Symmetries". The sole author is Taeyoung Kim, and v1 was posted on 27 July 2026. These match the
entry in `ref.bib`. Page numbers below refer to the v1 PDF.

- **The law.** Corollary 5 (page 8) defines k_grok(eps) := min{k : ||Pi_g xi_k|| <= eps}. It states that "in
  the weak-regularization regime eta*lambda << (1 - beta)^2 this gives the headline law"
  k_grok(eps) ≈ (1 - beta)/(eta*lambda) · log(||Pi_g xi_0|| / eps), which is Equation 20.
- **What k_grok counts.** The paper says that "k_grok is defined here as a parameter-space relaxation count of
  the population-active null component, not through test accuracy". It is counted from iteration 0.
- **The regime.** Assumption 18 (page 14) is eta*lambda < (1 - sqrt(beta))^2, together with a condition on the
  transverse eigenvalues. At beta = 0 the first bound is 1, and the largest decay in the grid, 3e-4, is far
  inside it.
- **Beta equal to 0 is covered.** Section 2.2 (page 3) analyses the heavy-ball scheme "with learning rate
  eta > 0 and momentum beta in [0, 1)". At beta = 0 the law is k_grok ≈ (1/(eta*lambda)) · log(...).
- **Coupled and decoupled decay.** Remark 2 (page 4) says, "For plain gradient descent the two coincide." This
  confirms the report's statement that Kim's coupled/decoupled distinction does not arise in our regime.
- **The modular-addition check.** Section 7.5 (page 23) uses p = 17, a training fraction of 0.55, a two-layer
  MLP of width 128 with quadratic activation, squared loss, full-batch heavy ball with coupled L2, and a
  5-times initialisation. The sweep covers eta in {0.2, 0.5, 1.0}, lambda in {1, 2, 3} × 1e-4 and beta in
  {0.8, 0.9, 0.95}, with 27 configurations on one seed. It measures "the iteration at which [test accuracy
  0.9] is first reached", counted from step 0. That iteration "scales with (1 - beta)/(eta*lambda), with a
  fitted log-log slope of 1.01". Beta = 0 is inside the theory but was not part of this sweep.
- **Scope.** The Scope paragraph (page 3) lists "grokking without explicit weight decay" among the things the
  mechanism does not explain. It also lists "the creation of new population-visible directions by feature
  learning".
- **Width.** In Kim, N is the number of training examples (Section 2.1, "a dataset {(x_i, y_i)}, i = 1 to N").
  The "N-sweep" in Section 7.6 (page 23) is a sweep over the training-set size, with the rate predicted to be
  independent of it. The paper makes no claim about network width. The only route by which width could enter
  the law is the initial amplitude ||Pi_g xi_0|| inside the logarithm.

**What follows for Tier 2.** Kim's quantity is counted from step 0, so the notebook also fits the
generalisation step t_test next to the gap t_grok. The t_test fit is the like-for-like comparison with Kim.
A weight-decay clock in Kim's form predicts b = -1 and at most a logarithmic dependence on width. It does not
predict exactly a = 0. A finite grokking time at eta*lambda = 0 is outside Kim's stated scope, so it shows a
clock that Kim's mechanism does not cover, but it does not contradict Kim.

### Dyer and Gur-Ari, the source for "a = 1": not yet checked

The report says a = 1 would follow if the finite-width correction scales as O(1/N), and it attributes this to
Dyer and Gur-Ari. There is no entry for this paper in `ref.bib`, and nobody has checked it. Until someone
does, the notebook treats a > 0 as H's prediction and draws a = 1 only as a grey reference marked
[UNVERIFIED].

## Run log

**26 September 2026, before the sweep.** A timing test ran under `/opt/miniconda3/envs/eht_env/bin/python`, not
tara-env, and saved nothing. One kernel evaluation took 0.65 s at N = 50, 0.04 s at N = 100, 0.19 s at
N = 400, and 0.86 s at N = 1600. The N = 50 figure includes the first-call start-up cost. One full-batch
training step took 0.15, 0.29, 0.49 and 1.74 ms at the same widths. From these, a 300,000-step run at
N = 1600 takes about 20 minutes, and the whole grid takes about 12 CPU-hours. The smoke test will re-time this
under tara-env.

**26 September 2026, library checks.** `censored_log_fit` was rewritten as a wrapper around the new
`censored_linear_fit`. On the inputs of `tier1_v2.ipynb` section 8 (48 runs, alignment level 0.105, 200
bootstrap draws), the coefficients, intervals, log sigma and bootstrap count before and after the change are
identical, with a largest absolute difference of 0.0. On simulated data from the grid design, with
log t = 2 + 0.8 log N - 0.3 log(eta*lambda), noise of 0.3, and 18 of 80 runs censored, the fit returned
a = +0.779 [+0.739, +0.816] and b = -0.299 [-0.339, -0.256]. Both intervals cover the true values. Least
squares on the same censored values gave a = 0.646 and b = -0.226, which shows the bias that the censored fit
removes. The reciprocal form with left censoring (6 of 80 censored) recovered all three coefficients inside
their intervals.

**26 September 2026, cache check.** All 17 saved N = 100, alpha 1 runs match the Tier 2 configuration exactly,
so they are loaded and not retrained. The other 83 runs of the grid were not yet saved.

**26 September 2026, sweep started.** The smoke test (N = 1600, decay 3e-4, seed 0) was started under tara-env
with 4 torch threads. At the same time, a second process with 2 threads started on the N = 50 and N = 100
cells. The two processes share the machine, so the smoke-test timing is an upper bound. The machine has 12
logical cores and 24 GB of memory. An N = 50 run takes about 25 seconds.

**26 September 2026, N = 50 and N = 100 columns finished.** The second process trained 33 new runs in 0.49
hours of wall-clock time: 25 at N = 50 and the 8 missing seeds at N = 100. The other 17 N = 100 runs were
loaded. An N = 50 run took 22 to 32 seconds on its own and up to about 140 seconds while sharing the machine.
An N = 100 run took about 60 seconds. Every run memorised. Two N = 50 runs never reached test accuracy 1.0
within 100,000 steps and are censored: seed 1 with no decay (memorised at step 17,000) and seed 1 at decay
1e-4 (memorised at step 11,500).

**26 September 2026, smoke test finished: N = 1600 at decay 3e-4 never memorises.** The run took 1,846
seconds, about 31 minutes, for 300,000 steps on 4 threads while sharing the machine. That is half again the
planning estimate of 20 minutes. Train accuracy peaked at 0.916 at step 500 and then fell to 0.735 by the end.
Test accuracy stayed at 0.0755, against a chance level of 1/23 = 0.043. The weight norm fell to 0.31 of its
initial value. The centred scale term ended at -1.94, so the centred kernel norm fell to about 0.14 of its
initial value. The train loss flattened at 0.034 from about step 100,000. The run has no grokking time and
does not enter any fit. It was not stopped early, and it is saved as `ntk_N1600_a1_wd0.0003_s0`.

**26 September 2026, pilot finished.** Seed 0 of every decay level at N = 400 and N = 1600 was run in three
parallel processes of 3 threads each, taking 1.11 hours of wall-clock time. An N = 400 run took 518 to 794
seconds and an N = 1600 run 1,891 to 2,098 seconds. The seed 0 results across the whole grid follow. Each cell
gives the memorisation step, the grokking time, and the final weight norm as a multiple of the initial one.

| N | no decay | 1e-5 | 3e-5 | 1e-4 | 3e-4 |
|---|---|---|---|---|---|
| 50 | 16,000; 12,000; 2.81 | 15,500; 7,500; 2.54 | 13,000; 7,500; 2.23 | 10,500; 5,500; 1.91 | 8,000; 6,500; 1.60 |
| 100 | 4,500; 12,500; 2.16 | 4,500; 11,500; 1.93 | 4,500; 10,000; 1.73 | 4,000; 6,000; 1.49 | 4,500; 4,000; 1.28 |
| 400 | 500; >299,500; 1.65 | 500; 81,000; 1.18 | 500; 27,000; 1.07 | 500; 12,500; 0.95 | 7,500; 7,000; 0.75 |
| 1600 | 500; >299,500; 1.29 | 500; 127,000; 0.80 | 500; 50,500; 0.71 | 500; 24,000; 0.60 | never memorises; none; 0.31 |

Only one cell fails to memorise on seed 0, N = 1600 at 3e-4. Every other cell at the wide widths memorises
and, apart from the no-decay cells, generalises within the 300,000-step budget. So the decay axis works at the
wide widths except in that one corner of the grid.

**26 September 2026, 16:01, seeds 1 to 4 started.** Six tara-env processes of 2 threads each were started:
one per seed at N = 1600, covering all five decays, and two at N = 400, one for seeds 1 and 2 and one for
seeds 3 and 4. The logs are kept outside the repository, in the session scratchpad.

**26 September 2026, about 18:15, N = 400 column finished.** The two N = 400 processes finished after 2.21
and 2.23 hours. While all six processes shared the machine, with a load average of about 9 to 10, an N = 400
run took from about 530 seconds to about 2,070 seconds, and the first N = 1600 runs took about 3,900
seconds. Six processes of 2 threads got less total work done than the pilot's three processes of 3 threads,
so four or fewer processes would be the better setting for a future sweep. The configuration was not changed
mid-sweep, because runs do not save part-way and restarting would have discarded about 3 CPU-hours of
partial runs. At N = 400 all 25 runs memorised. All five no-decay runs are censored at more than 299,500
steps, and every run at nonzero decay generalised.

**26 September 2026, 19:30, the machine had been sleeping.** The power log shows the Mac entering idle sleep
repeatedly from 16:13 onward whenever it was left alone, with the runs advancing only in short wake windows.
This, rather than competition between processes, explains most of the slowdown and the erratic per-run times.
It also means that the `seconds` field saved with each run of this sweep is wall-clock time including sleep.
That field is not a reliable measure of compute cost for any run after 16:13, so the per-run times quoted
above for the six-process sweep are upper bounds. No run was affected in any other way, because a paused
process resumes exactly where it stopped. At 19:31, `caffeinate -is -w <pid>` was attached to each of the
four N = 1600 processes. It prevents idle and system sleep on AC power while they run. Future sweeps on this
machine should be started under `caffeinate`.

**26 September 2026, remaining cost (superseded).** Before the pilot, the estimate was about 31 minutes per
N = 1600 run and 8 to 12 minutes per N = 400 run. On that basis the remaining runs would have taken about 17
hours in one process, or 6 to 8 hours in four parallel processes.

**26 September 2026, 20:40, sweep complete.** All 100 runs are saved and match the Tier 2 configuration. The
last four N = 1600 processes finished after 4.62 to 4.65 hours of wall-clock time, including the sleep before
19:31. With the Mac awake, an N = 1600 run took about 1,890 to 1,940 seconds on 2 threads with four processes
sharing the machine. All five seeds at N = 1600 with decay 3e-4 failed to memorise. Train accuracy peaked
between 0.90 and 0.93 and ended between 0.68 and 0.75, test accuracy peaked between 0.075 and 0.170, and the
weight norm ended at 0.30 to 0.31 of its initial value.

**26 September 2026, notebook executed.** `tier2_width_decay.ipynb` was executed top to bottom in place with
`/opt/miniconda3/envs/tara-env/bin/jupyter nbconvert --execute`. It raised no errors and produced 11 figures.
It was executed a second time after the findings cell was written, and every reported number was identical,
because all bootstrap and permutation seeds are fixed.

## Results (superseded on 4 October 2026)

All numbers are from NTK parameterisation at alpha 1, with the generalisation event at test accuracy 1.0.
Notebook sections are given in brackets.

- **Usable runs.** 95 of the 100 runs memorised. The 5 that did not are all at N = 1600 with decay 3e-4. Of
  the 95, 12 are censored: the 10 no-decay runs at N = 400 and N = 1600, and 2 at N = 50 (section 1).
- **Main effects on t_grok.** Over the 75 nonzero-decay runs, 1 of them censored, a = +0.307 [+0.228, +0.382]
  and b = -0.491 [-0.558, -0.421]. The residual sigma is 0.603 (section 5).
- **Interaction.** The centred log N times log(eta*lambda) term is -0.210 [-0.271, -0.154]. With it,
  a = +0.241 and b = -0.554, and sigma falls to 0.520 (section 5).
- **Main effects on t_test, counted from step 0 as Kim measures.** a = +0.152 [+0.100, +0.200] and
  b = -0.377 [-0.421, -0.330] (section 5).
- **Width exponent within each decay level.** It is +1.76 with no decay, where the wide cells are all
  censored so this is a rough lower-bound fit. It is +0.61 at 1e-5, +0.32 at 3e-5, +0.16 [-0.07, +0.35] at
  1e-4, and -0.21 at 3e-4, where N = 1600 is missing (section 5).
- **Decay exponent within each width.** It is -0.15 [-0.37, +0.06] at N = 50, -0.38 at N = 100, -0.82 at
  N = 400, and -0.71 at N = 1600 (section 5).
- **Residuals.** Four cells have a mean residual above twice its standard error: N = 50 at 1e-4 (+0.73) and
  3e-4 (+0.79), N = 100 at 1e-5 (-0.65), and N = 400 at 1e-5 (+0.74) (section 6).
- **Reciprocal form, 95 runs of which 12 censored.** c0 = +1.17 [+0.85, +1.45] at N = 1, and
  c0' = +0.435 [+0.334, +0.530] at N = 100. The decay coefficient is c1 = +0.435 [+0.375, +0.500] per
  0.0001 of decay, and the width coefficient is c2 = -0.159 [-0.203, -0.113], all per 10,000 steps. The joint
  test of c0 = c2 = 0 gives Wald 89.8 and p = 3.2e-20. The no-decay rates are 0.545, 0.435, 0.215 and
  -0.005 [-0.049, +0.040] at N = 50, 100, 400 and 1600. The share of the fitted rate at N = 100 that comes
  from decay is 9, 23, 50 and 75 percent at 1e-5, 3e-5, 1e-4 and 3e-4 (section 7).
- **Terms at the event, over the 83 runs that generalised.** The alignment has a max/min ratio of 1.75 and a
  spread of 11.7 percent, and its mean by width is 0.107, 0.121, 0.132 and 0.131. The shape term R has a
  max/min of 37.8 and a spread of 87.1 percent, and its mean by width is 0.158, 0.068, 0.018 and 0.006. The
  kernel norm ratio exp(S) has a max/min of 29.3 and a spread of 79.2 percent (section 8).
- **Placebo test.** For the alignment the spread is 11.7 percent at the true events against a median of 13.5
  percent at swapped events. That is a tightening of 1.1-fold, with p < 0.0005 (0 of 2,000 permutations as
  tight). R shows no tightening (p = 0.92) and neither does exp(S) (p = 0.80) (section 8).
- **Early rate of the alignment over steps 0 to 2,000.** The median by width is 0.0094, 0.0078, 0.0035 and
  0.0012 per 1,000 steps. The censored slope of log t_grok on the log rate is -0.89 [-1.02, -0.78]. With
  log N added, the slope is -1.12 [-2.13, -0.20] and the log N coefficient is -0.14 [-0.77, +0.44]. In 40 of
  the 95 runs memorisation falls inside the window (section 9).
- **Sensitivity.** At test accuracy 1.0, 0.95 and 0.9, a is +0.31, +0.41 and +0.50 and b is -0.49, -0.49 and
  -0.48. Under the loss definition 53 of 90 runs are censored, so that fit is not interpretable (section 10).

## Observations (superseded on 4 October 2026)

**26 September 2026. A_0 rises with width.** At seed 0 the centred alignment of the initial kernel is 0.0845
at N = 50, 0.0903 at N = 100, 0.0951 at N = 400, and 0.0957 at N = 1600. In Tier 1 the alignment at the
generalisation event was about 0.12. If that threshold holds at every width, the distance from A_0 to the
threshold shrinks by about a third between N = 50 and N = 1600. That would shorten the grokking time at large
N and work against a positive a. The effect is examined in notebook section 8.

**26 September 2026. The Tier 1 runs already give a decay slope at N = 100.** At alpha 1 the grokking time is
11,500 to 24,500 steps at decay 1e-5 and 4,000 to 5,500 steps at 3e-4. Across 1.5 decades that is a log slope
of about -0.35, which is neither 0 nor -1. The two-clock case is therefore likely, at least at N = 100.

**26 September 2026. In the fast cells the grokking time is about as long as the memorisation time.** At
N = 100 memorisation happens between steps 4,000 and 5,500 in every cell. In the 3e-4 cells the gap t_grok is
also about 4,000 steps, so subtracting the memorisation step roughly halves the time counted from step 0.
In the slow cells it barely changes it. The decay slope of t_grok will therefore be steeper than the decay
slope of t_test, and both are reported.

**26 September 2026. Memorisation moves with width.** At N = 50 memorisation happens between steps 8,000
and 19,500, against 4,000 to 5,500 at N = 100. A width effect on the grokking time therefore has to be read
beside the width effect on the memorisation step, which is why t_test is fitted as well as t_grok.

**26 September 2026. At N = 1600 a decay of 3e-4 stops memorisation, and the alignment still passes the
Tier 1 threshold.** In the smoke test, weight decay shrinks the weights faster than the gradient can grow
them. A likely reason is that in NTK parameterisation a wider network is lazier, so its weights move less per
step while decay shrinks every weight at the same rate. This has not been tested. At N = 100 the same decay
lets the weights grow to 1.28 times their initial norm. The run's centred alignment reached 0.126, above the
alignment of about 0.12 at which Tier 1 runs generalised, yet test accuracy stayed near chance. This is a
second case, after the alpha 2, decay 3e-4 runs of Tier 1, where crossing the alignment band is not
sufficient for generalisation. It bears on H1 and belongs in the report. It also means that the decay axis
may leave some nonzero-decay cells at the widest width with no usable runs, which would unbalance the
(a, b) fit.

**26 September 2026. On seed 0 the grokking time rises steeply with width above N = 100, and memorisation
becomes almost immediate.** At N = 400 and N = 1600 the network memorises by step 500, the first checkpoint,
in every cell except the strongest decay. The grokking time at seed 0 then grows with width at every nonzero
decay below 3e-4. At 1e-5 it is 7,500, 11,500, 81,000 and 127,000 steps from N = 50 to N = 1600, and at 3e-5 it
is 7,500, 10,000, 27,000 and 50,500. With no decay, both wide widths are censored at 299,500 steps. Between
N = 50 and N = 100 the grokking time barely changes, but memorisation moves from about 16,000 to about 4,500
steps. The width effect is therefore not a single power law across the grid, and the residuals in section 6
of the notebook should show it. This is one seed and must not be read as a result until seeds 1 to 4 are in.

**26 September 2026. The final weight norm falls with width at every decay.** At no decay it is 2.81 times
the initial norm at N = 50, 2.16 at N = 100, 1.65 at N = 400, and 1.29 at N = 1600. With decay it drops below
1 at the wide widths. Scale and width therefore move together, which is the difficulty the reading guide of
section 3 warns about: a width effect on timing could run through scale as well as through shape.

**26 September 2026. Seeds vary by up to about fourfold.** At N = 100 with no decay, the five seeds grok after
12,500, 47,000, 30,000, 18,500 and 51,500 steps. Five seeds per cell is the minimum for a stable median.

**26 September 2026. The wide cells are far more consistent across seeds than the narrow ones.** At N = 1600
the five seeds at 1e-5 grok in 108,000 to 127,000 steps and the five at 1e-4 in 20,500 to 24,000 steps. At
N = 50 and N = 100 the seeds of one cell can differ fourfold. This fits a wide network behaving close to its
deterministic kernel limit.

**26 September 2026. The data suggest two routes that run side by side, rather than two clocks where the
slower one sets the time.** At narrow widths decay barely changes the grokking time (b = -0.15 at N = 50). At
wide widths the no-decay route has stalled within the budget, and the grokking time falls close to
1/(eta*lambda) (b = -0.82 and -0.71 at N = 400 and N = 1600). That is what one would expect if the faster of
two routes sets the time. The report says the slower one does. This is a hypothesis drawn from the pattern.
No model of the form 1/t = r(N) + s(eta*lambda), or of the minimum of two times, has been fitted and
compared, so it must not be written up as a result.

**26 September 2026. Across width the alignment is a better threshold variable than R, which reverses
Tier 1.** In Tier 1, across alpha and decay at N = 100, R was locked to the event more tightly than A. Across
width, R at the event falls 27-fold from N = 50 to N = 1600, because a wider network turns its kernel less.
The alignment at the event rises only from 0.107 to 0.131. Width therefore separates the task-blind shape
term from the task-referenced alignment in the way H needs, which the alpha by decay grid could not do.

**26 September 2026. The early alignment rate is almost a function of width alone.** Within one width its
range is narrow, for example 0.0009 to 0.0013 per 1,000 steps at N = 1600, while decay moves the grokking time
tenfold. The early rate therefore carries the width effect and none of the decay effect. Its slope of about
-0.9 against the grokking time should not be read as confirming "distance over rate" across the whole grid.

## Open questions for the report

1. **The decay axis is short (resolved on 26 September 2026).** The worry was that four nonzero levels over
   1.5 decades might not separate b = 0 from b = -1. They did: the interval on b is [-0.558, -0.421] and
   excludes both. The interval should still be stated beside b in the report.
2. **The budget differs by width (open).** The censoring bound is 100,000 or 300,000 steps depending on N. The
   Tobit likelihood handles unequal bounds, but the rule has to be stated in the report.
3. **The reciprocal intercept (open).** In 1/t = c0 + c1 eta*lambda + c2 log N, c0 is the rate at N = 1, which
   is outside the grid. A pure weight-decay clock needs c0 = c2 = 0, not c0 = 0 alone as the report says. The
   notebook reports c0, the intercept at the baseline width c0' = c0 + c2 log 100, and the joint test.
4. **The noise model of the reciprocal fit (open).** The report does not give one. The notebook uses normal
   noise on 1/t, with a censored run entering as left-censored because 1/t is at most one over the bound. This
   is a choice and has to be stated as one.
5. **The source for a = 1 (open).** See the source-check section above.
6. **Main effects against interaction (open).** The group should confirm the choice recorded under design
   decisions.
7. **Runs that never memorise (open, now quantified).** These have no grokking time and are left out of the
   fits rather than censored, because the pre-registered censoring rule covers only a missing generalisation
   event. Five runs fall in this group, all at N = 1600 with decay 3e-4, so that cell is missing from the
   log fit. The report has to say so. It also has to say that the within-decay width exponent at 3e-4 comes
   from three widths only.
11. **The early-rate window contains memorisation at the wide widths (open).** The window of steps 0 to 2,000
    was fixed in advance. At N of 400 and more, memorisation happens at step 500, inside it, in 40 of the 95
    usable runs. Section 9 reports this rather than moving the window after the fact. If the group wants a
    clean pre-memorisation rate at the wide widths, the runs would need checkpoints finer than every 500 steps
    early on.
12. **"The slower one sets the time" (open).** The data point towards the faster of two routes setting the
    time. See the observation above. A model comparison is needed before the report takes a position.
13. **No-decay cells at the wide widths are all censored at 300,000 steps (open).** The steep width effect
    without decay rests on lower bounds. A longer budget for those ten runs would turn the bounds into
    measurements, at about 30 to 35 minutes per 300,000 steps at N = 1600 when the machine is awake.
8. **Width and alpha both act on laziness (open).** At fixed alpha, a wider network in NTK parameterisation is
   lazier. That is the intended mechanism, but it means the width exponent is not separate from laziness.
9. **Kim's clock (resolved on 26 September 2026).** Kim's time is counted from step 0 and Kim's scope excludes
   grokking without decay. See the source-check section.
10. **Checkpoint resolution (open, minor).** Checkpoints are every 500 steps, so each event is recorded at the
    first checkpoint after it happens and can be up to 500 steps late. The grokking time can therefore be off
    by up to 500 steps. That is about 12 percent for the fastest cells, which grok in about 4,000 steps, and
    about 1 percent for cells near 40,000 steps. It is well below the spread between seeds.

## Report follow-ups

- Add the Tier 2 row to the design table with the grid actually run, and state the decay-axis deviation and
  its reason.
- Correct the reciprocal-form paragraph so that a pure weight-decay clock requires c0 = c2 = 0, and report c0
  at the baseline width.
- Reword the sentence on the eta*lambda = 0 column. Kim's own Scope paragraph says the weight-decay mechanism
  does not explain grokking without decay, so a finite time there shows a second clock rather than refuting
  Kim.
- State that Kim's time is counted from step 0, and give the t_test fit next to the t_grok fit.
- Either verify the Dyer and Gur-Ari source for a = 1 and add it to `ref.bib`, or weaken the sentence to
  "a > 0".
- State in every width result that the NTK parameterisation was used.
- Report the head claim as it came out: both exponents nonzero, a = +0.31 and b = -0.49, with a significant
  interaction. Give the within-stratum slopes beside the (a, b) numbers.
- Add N = 1600 at decay 3e-4 as a second case, after the alpha 2 runs at 3e-4 in Tier 1, where the alignment
  reaches the band of the generalising runs without generalisation.
- Update the H1 discussion. Across width, the alignment at the event drifts up by about a quarter while R
  falls 27-fold, so width favours A over R as the threshold variable, which reverses the Tier 1 comparison.
- Revisit the sentence "two nonzero exponents mean two clocks, and the slower one sets the time" in the
  proposal and the report (open question 12).
