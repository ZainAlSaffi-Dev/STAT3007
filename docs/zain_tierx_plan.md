# Tier x: are scale, rotation and alignment linear combinations of each other?

This document is the plan for a side investigation that sits beside Tier 1 of the report.
It is written so that a later working session, by a person or by an assistant, can carry it
out without re-deriving anything. It records what is already known, what question is being
asked, which experiments answer it, what every run should log so the animations can be
driven from saved data, and in what order to do the work.

Everything here follows the root `CLAUDE.md`. Every number quoted below was read from the
repository on 24 September 2026, either by running `repoduced-code/term_dependence.py` or
by reading the grid runs in `repoduced-code/results/`. Where a formula is the group's own
algebra rather than a published result, the text says so.

## 1. Where this sits in the report

The report (`docs/report.tex`, Methods, "Scope: four reportable tiers") lists Tier 0, the
reproduction; Tier 1, the three terms over the alpha by weight-decay grid and the threshold
test; Tier 2, the fit of log grokking time on log width and log decay; Tier 3, the clamped
norm; and Tier 4, theory. The work here is a robustness and interpretation study for
Tier 1. It asks whether the three timing variables carry three separate pieces of
information, and if they do not, which one is redundant. It also produces the per-step logs
that the deck needs. It is called tier x because it is not one of the numbered tiers and
should not be presented as one.

The three terms are defined in the report's Kernel metrics section on the centred probe
kernel. In a saved run they are the history fields `S_c`, `R_c` and `A_t`. The fields
`S_t` and `R_t` hold the raw versions and are not used here except for comparison.

## 2. What is already known

`repoduced-code/term_dependence.py` settled part of the question before this plan was
written. Its docstring and its printed output on 24 September 2026 give the following.

By definition, all three terms are computed on the centred kernel. The scale term depends
only on the norm of that kernel. The rotation term and the alignment depend only on the
kernel divided by its norm. So the scale term cannot be recovered from the other two, and
neither of them can be recovered from it. This needs no experiment.

Rotation and alignment are tied by an exact identity. Write k_t for the unit centred kernel
and g for the unit centred target Gram matrix. Split k_t into its part along k_0 and a
remainder orthogonal to k_0. Taking the inner product with g gives

    A_t = A_0 (1 - R_t) + gamma_t sqrt(1 - A_0^2) sqrt(R_t (2 - R_t)),

where gamma_t, the aim, is the cosine between the remainder and the part of g orthogonal
to k_0. This identity is the group's own algebra, in the same way as Equation 1 of the
proposal. Given A_0, the triple (S_t, R_t, gamma_t) carries the same information as
(S_t, R_t, A_t), and none of the three is fixed by the other two.

The script checks the identity on the three dense runs and finds the largest difference
between the aim computed from the saved kernels and the aim computed from the recorded
terms to be below 1e-5 over 120 checkpoints each.

The numbers below are from the grid after it was trained again with the mixed probe of
Section 11. The numbers from the training probe are given in brackets. None of the
conclusions changed.

Along a run, the share of the variance of the scale term that a straight line in the
rotation term explains has median 0.993 over the 20 grid runs without decay (0.993), 0.850
over the nine runs at eta lambda 1e-4 (0.834), and 0.036 in the one run at eta lambda 1e-3
(0.045). So scale and rotation move in lockstep only when there is no decay.

At the generalisation event, which the Tier 1 notebook takes as test accuracy 1, 45 of the
63 grid runs have an event. Over those runs the correlations are 0.54 between scale and
rotation (0.54), 0.07 between rotation and alignment (0.11), and minus 0.31 between scale
and alignment (minus 0.34). Alignment and aim correlate at 0.99 (0.98). The aim at the
event lies between 0.045 and 0.161 (0.05 and 0.18) and rises with decay within each
alpha.

## 3. The question, made precise

The phrase "linear combination" hides four different claims. Each is stated below with the
test that would falsify it.

### L1. By definition

The scale term is a function of the kernel norm alone. The rotation term and the alignment
are functions of the unit kernel alone. So no linear relation between the scale term and
either of the others holds by definition. The alignment is an exact affine function of the
two quantities (1 - R_t) and sqrt(R_t (2 - R_t)), with coefficients A_0 and
gamma_t sqrt(1 - A_0^2). It follows that alignment is a fixed function of rotation along a
run only when the aim is constant along that run, and even then the relation is linear in
sqrt(R_t) for small R_t, never in R_t itself. This claim is settled by the identity and
needs no data. The notebook states it and moves on.

### L2. Along a run

Does an affine map from the rotation term predict the scale term, and does an affine map
from the rotation term, or from its square root, predict the alignment? The test is the
share of variance explained by a least-squares line, one value per run, reported by decay
level as `term_dependence.lockstep` already does for scale on rotation. The kill criterion
for the claim that scale and rotation are one variable is any decay level at which the
median share falls below 0.9. The known result already fails this at eta lambda 1e-4, so
the expected outcome is that the lockstep is a property of runs without decay and nothing
more. The traced runs of Section 4 confirm it at a finer time resolution and across alpha.

### L3. Across cells at the event

Are the three values at the generalisation event on one plane? The test is a least-squares
fit of the alignment at the event on the scale and rotation at the event with an intercept,
over every run with an event, with a bootstrap interval over runs. The known correlations
make a plane unlikely. The kill criterion for the claim that the terms at the event are
linearly dependent is a share of variance below 0.5.

### L4. In matrix space

Does the kernel move inside the plane spanned by the initial unit kernel and the unit
target? If it did, the aim would be plus or minus one and alignment would be a fixed
function of rotation. The share of the turn that leaves that plane is 1 - gamma_t^2. With
an aim near 0.1 at the event, about 99 percent of the turn is in directions that neither
the initial kernel nor the target sees. Finding those directions is the real content of
tier x, and Section 5 describes the experiment.

## 4. The runs

Seven runs are traced with the logger of Section 6. All are NTK parameterisation, width
100, base learning rate 100, seed 0, data seed 42, train fraction 0.9, and the mixed probe
of 203 training pairs and all 53 test pairs, as `ntk_lib.load_cell` fixes for the grid.
Section 11 says why the probe changed.

| Run | alpha | eta lambda | steps | Why |
| --- | --- | --- | --- | --- |
| `trace_N100_a1_wd0_s0` | 1 | 0 | 30,000 | a dense run exists; the kernel grows and turns and the network generalises |
| `trace_N100_a1_wd0.0003_s0` | 1 | 3e-4 | 30,000 | a dense run exists; the strongest decay in the grid that still generalises |
| `trace_N100_a1_wd0.001_s0` | 1 | 1e-3 | 30,000 | a dense run exists; decay collapses the weights |
| `trace_N100_a0.5_wd0_s0` | 0.5 | 0 | 30,000 | richer than alpha 1 without decay |
| `trace_N100_a0.5_wd0.0003_s0` | 0.5 | 3e-4 | 30,000 | richer than alpha 1 with decay |
| `trace_N100_a2_wd7.5e-05_s0` | 2 | 7.5e-5 | 60,000 | lazier than alpha 1; groks at about step 47,500 |
| `trace_N100_a2_wd0.0001_s0` | 2 | 1e-4 | 60,000 | lazier than alpha 1; groks at about step 31,500 |

The alpha 2 rows need 60,000 steps because the grid runs at seed 0 show the accuracy
crossing at test accuracy 1 at step 31,500 for eta lambda 1e-4 and at step 47,500 for
7.5e-5, and no crossing within 100,000 steps at eta lambda 0, 1e-5 or 3e-4. Alpha 0.5
groks by step 6,000 at every decay level in the grid, and alpha 1 by step 17,000.

The three cells that have dense runs are traced again rather than reused, because the
trace holds more than the kernel. Their new histories must agree with the saved grid runs
and dense runs at every shared checkpoint, as `ntk_trace.compare_with_saved` prints. If
they do not, stop and find out why before anything else is built on them.

Result on 24 September 2026. A teammate first computed the grid runs on another machine,
with the training probe. Against those, the fields that depend only on the weights agreed
exactly over 61 shared checkpoints, and the kernel fields differed by at most 1e-5 of their
largest value. The largest difference was in y^T K^+ y, whose pseudo-inverse magnifies the
float32 rounding of the kernel. After the probe changed, the grid was trained again on this
machine, and the three traced histories now agree exactly with it in every field. The
dense runs keep the training probe, so only their training fields can be compared, and
those agree exactly over 121 shared checkpoints.

Naming: the traced runs are called `trace_N100_a{alpha}_wd{eta lambda}_s0`, built by
`ntk_trace.run_name`. The trace files sit next to the run JSON. The first version of this
plan kept the dense prefix. That changed on 24 September 2026, because a traced history
also has a checkpoint at every log-spaced step, and four scenes read the dense histories
row by row with a checkpoint every 250 steps. The dense files stay as they are.

## 5. The experiments

### E1. Along a run and at the event, from the saved grid

No new training. Extend `term_dependence.lockstep` so that it reports three shares per run:
scale on rotation, alignment on rotation, and alignment on the square root of rotation.
Report one table per decay level with minimum, median and maximum. Then fit the plane of
L3 over the runs with an event and report the share of variance with a bootstrap interval.
Output: the tables, a scatter of alignment at the event against rotation at the event
coloured by scale at the event, and a scatter of alignment against aim at the event to show
which one alignment tracks.

### E2. The traced runs

Train the four new runs and re-trace the three dense ones with the logger. Print the
comparison with the grid runs. Repeat the E1 fits on the traced runs, which have finer
checkpoints and the early log-spaced steps that the grid runs lack.

### E3. Where the turn goes

At each checkpoint, split the unit centred kernel change into its component along k_0, its
component along the part of g orthogonal to k_0, and the residual orthogonal to both. The
squared lengths of the three components sum to one. Record them as the three shares and
plot them against step for each traced run. The expected picture, from the aim numbers, is
that the residual share stays above 0.9 throughout.

Then ask what the residual looks like. Take the leading eigenvector of the residual matrix
at each checkpoint. It is a function on the 256 probe points, each of which is a pair
(a, b). Project it by least squares onto four subspaces of functions on the probe set: the
functions of (a + b) mod p, the functions of (a - b) mod p, the functions of a alone, and
the functions of b alone. Each has dimension p - 1 after centring. Least squares is needed
because the probe set is a subset of the p squared pairs, so the indicator functions are
not orthogonal on it. Report the four energy shares. The `KernelOnTheTorus` scene already
shows the change forming an X on the torus with one arm on the sum line and one on the
difference line, so the expectation is that the difference subspace carries most of the
residual. The labels live in the sum subspace, which is why the target does not see the
other arm.

### E4. The eigenvectors

Eigen-decompose the centred kernel at every checkpoint, eigenvalues in descending order.
Keep the top k = 88 = 4(p - 1) eigenvectors. The first choice was 32. The alpha 1 traces
showed that the leading 44 eigenvectors are functions of a alone and of b alone, and that
the largest relative gap in the spectrum comes right after them. They also showed that the
functions of the sum and of the difference rise out of the bulk to indices 44 to 87 during
training. A cut at 32 fell inside the first block and never saw the second. Match each eigenvector at a checkpoint to one at the
previous checkpoint by the overlap matrix of absolute inner products and a linear
assignment (`scipy.optimize.linear_sum_assignment` on minus the overlap). Fix the sign of
each matched vector so that the inner product with its predecessor is positive. Record the
match index, the overlap with the predecessor, the overlap with the matched vector at
step 0, the principal angles between the top-k subspace at the checkpoint and at step 0,
the principal angles between the top-k subspace and the label subspace (the column space of
H Y), and the four energy shares of E3 for each eigenvector. This gives the position of the
eigenvectors over training in a form the deck can draw. An eigenvector whose best overlap
with the previous checkpoint is below 0.5 is a new arrival and is marked with match index
minus one.

What the alpha 1 traces with the mixed probe showed about these conventions on 24 September
2026. The sign rule
works, and no matched pair has a negative inner product. From one checkpoint to the next
the matching is stable. The median assigned overlap is 0.96 to 0.99. New arrivals occur
only at indices 44 and above, where the sum and difference directions enter. Over a whole
run the chain of matches drifts. Inside the block of functions of a and of b, neighbouring
eigenvalues are 1 to 3 percent apart and the vectors mix slowly. At step 30,000 a vector
that traces back to step 0 has a median overlap of 0.17 to 0.18 with its origin, over
the three runs. So the
identity of a single eigenvector means something only over short spans. For longer spans,
use the principal angles and the energy shares. The array `origin_index` records the chain,
and `overlap_zero` shows how far it has drifted. The label angles do move with k = 88. At
step 0, 1 of the 22 angles is below 45 degrees. At step 30,000, 18 to 22 of them are.

### E5. A synthetic control

Build two artificial kernel paths from the step 0 kernel of one traced run. The first is
pure scale, K_t = c_t K_0 with c_t running from 1 to 5. The second is pure rotation
towards the target: the unit kernel moves along the great circle from k_0 towards the unit
target at fixed norm. Run every estimator of E1 to E4 on both paths and show what each
reports when the answer is known. Pure scale must give a rotation term of zero, an
undefined aim, and eigenvectors that do not move. Pure rotation must give a scale term of
zero, an aim of one, and a residual share of zero. These figures are schematic in the sense
of `animations/CLAUDE.md` and must say so on screen if they reach the deck.

### E6. The kernel from the weights, in numpy (optional)

The traced runs use `ntk_lib.NTKMLP`, which has no biases. For that network the
sum-over-logits kernel has a closed form. Write X for the probe inputs (n by 2p), H for
the hidden preactivations X W1 transposed over sqrt(2p) (n by N), Phi for relu(H), M for
the ReLU derivative mask, and s_i for the sum over outputs of W2 squared in column i. Then

    K = (p / N) Phi Phi^T + (1 / (N 2p)) (X X^T) * (M diag(s) M^T),

where the star is the elementwise product. The first term is the gradient with respect to
W2 and the second the gradient with respect to W1. This is the group's own derivation from
the definition of the kernel in `RepoducedCode.compute_entk`. It is not from a paper.
Implement it in `animations/common/kernels.py`, check it against the saved torch kernel to
1e-4 on every checkpoint of one traced run, and then the animations environment, which has
no torch, can recompute any kernel quantity from the saved weights.

## 6. The logger and its file format

### What changes in `ntk_lib`

`ntk_lib.train_run` gains two optional arguments and is otherwise unchanged.

- `checkpoint_steps`: an explicit sorted list of steps at which to evaluate. When given, it
  replaces the `eval_interval` rule. The default is `None`.
- `on_checkpoint`: a callable receiving the step, the model, the probe kernel K_t, and the
  history row just recorded. The default is `None`.

`run_or_load` compares a saved configuration with the requested one. Old saved runs do not
carry the new fields, so the comparison fills them with their defaults, in the same way it
fills `probe` today. The check that the Tier 0 and Tier 1 notebooks still load every saved
run without retraining is the first thing to do after this change.

### The tracer

New module `repoduced-code/ntk_trace.py`. It builds the checkpoint grid, supplies the
callback, and writes the trace at the end of the run. It does not fork the training loop.

The checkpoint grid is the union of three sets: step 0, every 250 steps up to the run
length, and about 60 log-spaced steps from 1 to the run length rounded to integers. The
report's Kernel metrics section asks for a log-spaced grid because grokking spans three to
five orders of magnitude in step. The every-250 part keeps the comparison with the existing
dense runs exact.

The tracer writes `results/{name}_trace.npz` and `results/{name}_trace.json`. The JSON
is the schema: for every array, its shape, its meaning, the formula it evaluates, and the
report equation or the `term_dependence.py` docstring the formula comes from. The npz is
ignored by git next to `*_kernels.npz`. The schema JSON is committed.

### Arrays in the trace

All arrays carry a leading checkpoint axis unless the table says once.

| Name | Shape | What it is |
| --- | --- | --- |
| `step` | (T,) | the checkpoint steps |
| `probe_a`, `probe_b` | (n,) once | the probe pairs, as `kernel_snapshots.add_probe_to_kernel_file` writes them |
| `W1`, `W2` | (T, N, 2p), (T, p, N) | the weights; everything below can be recomputed from them |
| `K` | (T, n, n) float32 | the raw probe kernel, as today |
| `f_probe` | (T, n, p) | the centred and rescaled predictor on the probe set |
| `train_loss`, `test_loss`, `train_acc`, `test_acc`, `param_dist`, `weight_norm`, `yKy` | (T,) | copied from the history row |
| `K_norm_centred` | (T,) | Frobenius norm of H K_t H |
| `inner_K0` | (T,) | Frobenius inner product of H K_t H with H K_0 H |
| `inner_G` | (T,) | Frobenius inner product of H K_t H with H Y Y^T H |
| `G_norm`, `A_0` | once | norm of the centred target Gram, and the alignment at step 0 |
| `S`, `R`, `A`, `A_uncentred` | (T,) | the three terms and the uncentred alignment, matching `S_c`, `R_c`, `A_t`, `A_u` |
| `D` | (T,) | the movement statistic on the centred kernel |
| `exp2S_term`, `cross_term` | (T,) | e^{2S} and 2 e^{S} (1 - R), the pieces of Equation eq:decomp |
| `gamma` | (T,) | the aim, NaN where R is zero |
| `share_k0`, `share_gperp`, `share_residual` | (T,) | the three squared shares of E3, summing to one |
| `eigvals` | (T, n) | eigenvalues of the centred kernel, descending |
| `eigvecs_topk` | (T, n, k) | the top k eigenvectors, matched and sign-fixed as in E4 |
| `match_index` | (T, k) | which index at the previous checkpoint each vector continues, minus one if new |
| `overlap_prev`, `overlap_zero` | (T, k) | absolute inner product with the predecessor and with the step 0 vector that `origin_index` names |
| `origin_index` | (T, k) | the index at step 0 that the chain of matches leads back to, minus one if the chain passes through a new arrival; added in step 2 |
| `principal_angles_zero` | (T, k) | between the top-k subspace and the step 0 top-k subspace |
| `principal_angles_label` | (T, min(k, p - 1)) | between the top-k subspace and the column space of H Y |
| `subspace_energy` | (T, k, 4) | the four energy shares of E3 for each eigenvector, in the order sum, difference, a, b |
| `torus_K_unit`, `torus_change`, `torus_residual` | (T, p, p) | torus averages, via `kernels.Torus`, of the unit centred kernel, of its change from step 0, and of the residual of E3 |

Conventions the schema records: k = 88; the label subspace is the column space of H Y with
Y the one-hot probe labels; eigenvalues descending; sign fixed by continuity; the probe
kernel is the sum-over-logits kernel of `RepoducedCode.compute_entk` in mode `trace`; all
three terms and the shares are on the centred kernel, per Cortes, Mohri and Rostamizadeh
(2012), Lemma 1, and the report's Kernel metrics section.

### Disk

The kernel is about 260 KB per checkpoint, the weights about 27 KB, the predictor on the
probe set about 24 KB, and the eigenvectors about 90 KB at k = 88. Compression saves
little. A 30,000-step run has 175 checkpoints, and each of its trace files measured 65 MB
on 24 September 2026. A 60,000-step run has 295 checkpoints and should be near 110 MB. The
seven runs together should be near 540 MB, all ignored by git.

## 7. The animation retrofit

The scenes under `animations/` load measured curves through `animations/common/data.py`
and may not carry numbers in their source. The trace extends what they can load.

- `data.py` gains `load_trace(name)`, returning an object with attribute access to every
  array and to the schema, and `TRACED_RUNS`, a list of the seven runs with their alpha and
  decay in the same shape as `DENSE_RUNS`. Its error message when the file is missing
  points at `ntk_trace.py`, in the way `load_kernels` points at `kernel_snapshots.py`.
- `kernels.py` gains the numpy kernel of E6 and the four-subspace projection of E3.
- `notation.py` gains symbols for the three shares, the residual direction and the
  eigenvectors before any scene uses them. The report has no macro for these yet; when the
  report gets one, the two must match.
- Existing scenes keep working from the run JSON. `KernelOnTheTorus` and `TurnAndAim`
  compute the torus change and the aim on the fly today and may read `torus_change` and
  `gamma` from the trace instead once it exists.
- New scenes the trace makes possible, to be added to the README table as planned and
  built after the notebook has confirmed what they should show:
  `EigenvectorsTurn`, the top eigenvectors of the centred kernel drawn on the torus and
  following their matched trajectory, with the eigenvalue as a bar;
  `OffPlaneShare`, the three shares against step and then the residual direction on the
  torus;
  `FormulaLive`, the report's definitions of S, R and A with the numbers filling in at a
  moving step, every number read from the trace.

Every scene still goes through the checklist at the end of `animations/CLAUDE.md`.

## 8. The notebook

`repoduced-code/zain_tierx.ipynb`, in the style of `tier1_alignment_threshold.ipynb`. It
imports `ntk_lib as L` and `ntk_trace`. Every figure is preceded by a markdown cell that
says what the figure shows and what would falsify the claim it supports. Sections:

1. The question and the identity. States L1 and shows the identity check from
   `term_dependence.check_identity`.
2. Along a run. E1 on the grid, with the tables by decay level.
3. At the event. E1 plane fit and the two scatters.
4. The traced runs. The comparison with the grid runs at shared checkpoints, then the E1
   fits on the traces.
5. Where the turn goes. E3, the three shares against step and the four energy shares of
   the residual.
6. The eigenvectors. E4, eigenvalue trajectories, persistence, principal angles.
7. The synthetic control. E5.
8. What this means for the report and for the deck. One paragraph per claim L1 to L4 with
   the numbers, and a list of which scenes to build.

## 9. Order of work for the next sessions

Steps 1 and 2 were done on 24 September 2026. Step 1 is commit cadebfe. Step 2 is the
commit that adds `ntk_trace.py`. Each step is committed on its own before the next starts.

1. Add `checkpoint_steps` and `on_checkpoint` to `ntk_lib.train_run` and `run_or_load`.
   Open `tier0_reproduction.ipynb` and `tier1_alignment_threshold.ipynb` and confirm every
   saved run loads without retraining.
2. Write `ntk_trace.py`. Trace the three dense cells. Print the comparison with the grid
   runs. Look at the matched eigenvectors on one run to confirm the sign and matching
   conventions behave.
3. Trace the four new cells. The alpha 2 cells take about a minute each on a laptop, by
   extrapolation from the 20 seconds `kernel_snapshots.py` reports per 30,000-step cell.
4. Write notebook sections 1 to 3 from the grid runs, then 4 to 8 from the traces.
5. Add `load_trace` and the numpy kernel to `animations/common/`. Check the numpy kernel
   against the saved torch kernel.
6. Update the run list in `animations/README.md` and the root `README.md`, then build the
   scenes in the order the notebook's last section recommends.

## 10. Sources this plan rests on

- The centred kernel and centred alignment: Cortes, Mohri and Rostamizadeh (2012),
  Journal of Machine Learning Research 13, pages 795 to 828, Lemma 1 and Definition 4.
- The exact kernel evolution that motivates looking for shape change outside the pure
  scale term: Lewkowycz and Gur-Ari (2020), arXiv:2006.08643, Theorems 1 and 2 and
  Equation S5, as cited in the report's Methods section.
- The Fourier structure of the generalising solution, which motivates the four subspaces
  of E3: Nanda et al. (2023), arXiv:2301.05217, as cited in the report's Related Work.
- The decomposition of the movement statistic, Equation eq:decomp of the report, and the
  aim identity in `term_dependence.py`: the group's own algebra.
- The kernel closed form of E6: the group's own derivation from the definition in
  `RepoducedCode.compute_entk`.

## 11. Where the report and the code disagreed, 24 September 2026

The code should match `docs/report.tex`. Where they differ, the source papers decide
which one is wrong. The report is still a skeleton, so a design line in it may be a draft.
Four differences were found while step 2 was done.

1. The kernel. The Methods line called it the gradient of the summed output. The
   report's Introduction, the proposal and `compute_entk` use the sum over outputs of the
   per-output kernels. That is the trace of the matrix-valued NTK of Jacot, Gabriel and
   Hongler (2018). Their Theorem 1 (arXiv v4) makes the two agree only in the
   infinite-width limit at initialisation. On the traced runs the two differ a lot once
   training starts. The report line was wrong and was corrected in commit c507abb.
2. The probe set. The report asks for train and test pairs in the probe, with metrics on
   each part and pooled. The code used the first 256 training pairs. No paper settles
   this. Kumar et al. (2024) evaluate their alignment on the test set (Section 5, arXiv
   v3), which supports the report. The code now follows the report. The mixed probe of
   `ntk_lib.select_probe` holds 203 training pairs and all 53 test pairs, and the history
   gains the terms on each part. The 63 grid runs that used the training probe were
   trained again. The weights follow the same path, so only the kernel fields changed.
   The dense runs keep the training probe because the scenes were built on them. The
   traced runs use the mixed probe.
3. The checkpoint grid. The report asks for a log-spaced grid. The grid runs check every
   500 steps. The report line is a draft, so the grid runs were left as they are. The
   traced runs carry the log-spaced grid.
4. The grokking time. The report and the proposal pre-register it on loss thresholds,
   whose values the report still marks as to do. The Tier 0 notebook makes the accuracy
   definition primary, because the test loss settles between 0.01 and 0.03 and rarely
   crosses 0.01. The Tier 1 notebook uses accuracy only. Kumar et al. (2024, Section 16,
   arXiv v3) call the loss and accuracy definitions of grokking equivalent. Changing a
   pre-registered definition is a decision for the group, so nothing was changed.
