# Plan: MNIST grokking on a network with a closed-form tangent kernel

> Note added after the run: every MNIST file now lives in the folder `mnist/`, and this plan with its scripts and results in `mnist/exploration/`. Paths below of the form `repoduced-code/mnist_*.py`, `repoduced-code/results/mnist_*` and `docs/PLAN_mnist_closed_form.md` now read `mnist/...`. `ntk_lib.py` stays in `repoduced-code/`.

This plan is for an overnight run by two agents. Agent 1 investigates Option B
in depth. Agent 2 starts only if Agent 1 does not find reliable grokking, and
searches for the network closest to the modular addition network that groks on
MNIST and keeps a closed-form kernel. Each section says what to do, what to
record and when to stop.

## 0. Rules for both agents

1. Read `CLAUDE.md` at the repository root before anything else. Its rules on
   sources, writing style and event definitions apply to every file and report
   you write. Do not use em dashes or en dashes as punctuation.
2. Do not commit, push, branch or stash. Do not run `git checkout` or
   `git reset`. The branch `sam` has uncommitted work by the user that must
   stay as it is.
3. Do not edit `docs/report_golden.tex`, `docs/report.tex`, anything under
   `model_fitting/`, or any existing file under `repoduced-code/results/`.
4. Changes to `repoduced-code/ntk_lib.py` must be additions that leave every
   existing default unchanged. After any change to it, run the regression
   check in section 2.4 and stop if it fails.
5. Put new code in new files in `repoduced-code/`. Put results in
   `repoduced-code/results/mnist_optionB/` (Agent 1) or
   `repoduced-code/results/mnist_fallback/` (Agent 2).
6. Use at most 12 worker processes at once, with `torch.set_num_threads(1)` in
   each. The machine is an M4 Pro with 12 cores.
7. Train through `ntk_lib.run_or_load`, so that a run that has finished is
   loaded and not trained again if the agent restarts.
8. Do not cite a paper that is not in the golden report without checking it on
   arXiv as `CLAUDE.md` describes. Mark anything unchecked `[UNVERIFIED]`.
9. Write a progress note to the agent's report file after every batch, so that
   the user can read where things stand if the run is stopped early.

## 1. What is already known

The user's project measures the scale S_t, the shape change R_t, the
variation D_t and the centred alignment A_t of the empirical tangent kernel
during grokking. The modular addition study uses a two layer bias-free ReLU
network in mean-field parameterisation with a centred predictor
alpha [f(theta) - f(theta_0)] and full-batch gradient descent. Its kernel has a
closed form, `ntk_lib.entk_closed_form`.

MNIST runs so far are in `repoduced-code/results/mnist_pilot/`, driven by
`repoduced-code/mnist_pilot.py`, with MNIST loaded by
`repoduced-code/mnist_data.py`. All use 1000 training images drawn by
`data_seed = 42`, the full 10000 test images, MSE on one-hot targets, and a
mixed probe of 128 training and 128 test images. All are one seed.

1. The modular addition network on MNIST, centred, with weight decay, never
   learned. Weight decay shrank the weights first.
2. The two layer bias-free network, uncentred, full-batch gradient descent,
   initial scale 3 to 9, eta_kappa up to 3e-5, trained slowly and showed no
   delayed generalisation.
3. Every centred run failed to grok at every alpha. Centring removes the large
   random initial output, and its removal is what produces the delay in the
   uncentred runs. Under AdamW, centred runs also lost test accuracy late in
   training, with and without weight decay.
4. Omnigrok's network (`OmnigrokMLP` in `ntk_lib`, three layers of width 200
   with biases, PyTorch default initialisation) grokked when uncentred:
   - AdamW, learning rate 1e-3, batch 200, initial scale 9, with and without
     weight decay. Test accuracy was 0.11 to 0.14 at memorisation and rose to
     0.83 to 0.92.
   - Plain SGD without momentum, batch 200, eta = 1 / lam_0, eta_kappa = 1e-5:
     - At initial scale 3 it memorised at step 3023 with test accuracy 0.56,
       and test accuracy reached 0.90 at about 30 times the memorisation step.
     - At initial scale 9, training accuracy was 0.93 and test accuracy 0.35
       at step 100000. Test accuracy then rose to 0.67 by step 200000, but
       training accuracy fell to 0.79 and the run never memorised.
5. With eta_kappa of 1e-4 or more under SGD, the weights shrank to near zero
   and the network predicted one class. PyTorch SGD with momentum 0.9 folds
   weight decay into the momentum buffer, so the effective decay is ten times
   larger.
6. In the runs that started near chance, test accuracy rose while the kernel
   norm fell, and A_t rose only afterwards.
7. Step timing on one core: a 3 layer width 200 training step at batch 200
   takes about 0.5 ms, and an autograd probe kernel takes about 1.5 s. Twelve
   parallel runs of 200000 steps with about 190 autograd kernels each took 34
   to 36 minutes.

The learning rate rule in use is eta = ratio / lam_0, where lam_0 is the top
eigenvalue of the Gauss-Newton matrix of the loss at initialisation, from
`mnist_pilot.top_gauss_newton_eigenvalue`. The curvature falls as weight decay
shrinks the weights, so a fixed eta becomes small relative to the curvature
late in a run. This is the likely reason the scale 9 SGD run stalled.

## 2. Shared groundwork (Agent 1 does this first)

### 2.1 A bias-free deep ReLU network

Add a model class to `ntk_lib.py`, for example `DeepBiasFreeMLP`, with these
properties.

- Its depth L (number of linear layers) and width are set by configuration.
  Add the configuration keys with defaults that leave existing runs unchanged,
  and add them to `LEGACY_VALUES`.
- It has no biases.
- It supports two parameterisations:
  - "standard": weights drawn uniformly from (-1/sqrt(fan_in), 1/sqrt(fan_in)),
    as PyTorch's `nn.Linear` does, and no factors outside the weights. This
    matches `OmnigrokMLP` without its biases.
  - "ntk": weights drawn from N(0, 1) with a factor 1/sqrt(fan_in) outside
    each layer, matching `NTKMLP` when L = 2.
- Every weight is multiplied by `init_scale` after drawing.

The network is L-homogeneous in its weights.

### 2.2 A general closed-form kernel

Write the sum-over-outputs tangent kernel of a bias-free L layer ReLU network
in closed form. Use this notation:

- c_l is the factor outside layer l (1 under "standard").
- a_0 = x, h_l = c_l W_l a_{l-1}, a_l = ReLU(h_l) for l < L, and f = h_L.
- M_l is the mask of positive entries of h_l.

The gradient of output c with respect to W_l is c_l d_{l,c} a_{l-1}^T, where
d_{L,c} is the unit vector e_c and

    d_{l,c} = M_l * (c_{l+1} W_{l+1}^T d_{l+1,c})  for l < L.

The kernel is a sum over layers,

    K(x, x') = sum_l c_l^2 <a_{l-1}(x), a_{l-1}(x')> <D_l(x), D_l(x')>_F,

where D_l(x) is the width_l by C matrix whose column c is d_{l,c}(x). For the
last layer the second factor is C. For L = 2 this reduces to the existing
`closed_form_terms`. Compute it in float64. Return the per-layer terms as well
as their sum, so that layer-wise statistics such as A_W1, A_W2 and A_W3 can be
recorded. The derivation is the group's own. Say so in the docstring.

### 2.3 Verification of the closed form

Write `repoduced-code/verify_closed_form_deep.py`. It must check the following
and print pass or fail for each.

1. The new function equals `entk_closed_form` for the two layer `NTKMLP` and
   `MeanFieldMLP` on a modular addition probe, to a relative Frobenius error
   below 1e-10 in float64.
2. The new function equals the autograd kernel `ntk_lib.entk` for L in
   {2, 3, 4}, both parameterisations, init_scale in {1, 9}, on a probe of 64
   MNIST images. Use a relative Frobenius error below 1e-6, comparing in
   float64 where possible.
3. The same holds at a trained state: after 500 SGD steps of a 3 layer network.
4. The cost of one probe kernel of 256 images for the 3 layer width 200
   network. Record it.

If any check fails, stop and write down what failed. Do not use the closed
form in any experiment until all checks pass.

### 2.4 Regression check on modular addition

Run a 1500-step modular addition run with
`probe="mixed", alpha=2.0, eta_kappa=1e-4, kernel_method="closed_form", seed=1`
under the edited `ntk_lib`. Compare it with the same run under the committed
version, which is `git show HEAD:repoduced-code/ntk_lib.py` saved to a scratch
folder with `RepoducedCode.py` beside it. The maximum absolute difference over
`HISTORY_KEYS` must be 0.0.

### 2.5 Wiring the closed form into training

Let `kernel_method="closed_form"` dispatch to the new function for
`DeepBiasFreeMLP`, keeping `entk_closed_form` for the existing classes. Record
the per-layer alignments and scales at each checkpoint, through the
`on_checkpoint` callback or extra history keys. Extra keys must not change the
history of existing configurations.

## 3. The grokking criterion for MNIST

The user chose memorisation at 95 percent training accuracy, with test
thresholds read at several levels. Use these definitions for every MNIST run
in this plan.

- t_mem is the first checkpoint at which training accuracy is at least 0.95.
- For each level l in {0.30, 0.35, ..., 0.90}, t_l is the first checkpoint at
  which test accuracy is at least l. The delay ratio at level l is t_l / t_mem
  if t_l > t_mem. The level is "pre" if t_l <= t_mem, and "never" if it is not
  reached.
- A run shows **strong grokking** if all of these hold:
  - test accuracy at t_mem is at most 0.30,
  - the best test accuracy is at least 0.80,
  - the delay ratio at level 0.50 is at least 3.
- A run shows **weak grokking** if it is not strong and both of these hold:
  - the gain after memorisation, best test accuracy minus test accuracy at
    t_mem, is at least 0.25,
  - the delay ratio at the level halfway through that gain is at least 3.
- A configuration **groks reliably** if at least 2 of 3 seeds meet the same
  criterion.

These definitions are for this investigation. The report's definitions in
`CLAUDE.md` and golden report section 3.5 are unchanged. Report every run's
full level table as well as the verdict.

For reference, the scale 3 SGD run of section 1 item 4 is weak grokking: test
accuracy 0.56 at memorisation, best 0.91.

## 4. Agent 1: Option B in depth

Option B is `OmnigrokMLP` without biases: three linear layers, width 200, ReLU,
standard parameterisation, uncentred predictor. Keep MSE, one-hot targets,
`dataset="mnist"`, `n_train=1000`, `data_seed=42`, `probe="mixed"`,
`probe_size=256` and `optimizer="sgd"` unless a stage says otherwise. Use the
closed-form kernel once section 2.3 passes. Use a checkpoint grid that is log
spaced in step together with a linear grid, like `mnist_pilot.log_grid`.

Write a driver, `repoduced-code/mnist_optionB.py`, in the style of
`mnist_pilot.py`, with one named batch per stage below.

### Stage B1. Replication (12 runs, 200000 steps)

Cross initial scale {3, 9} with eta_kappa {1e-5, 3e-5}, momentum 0, batch 200,
and seeds {0, 1, 2}. Set eta = 1 / lam_0 for each scale.

This answers two questions. Does removing the biases keep the behaviour of
section 1 item 4? Is it stable across seeds?

### Stage B2. The late-training stall (up to 12 runs)

The scale 9 run stalled as the curvature fell. Test these fixes in order.
Use initial scale 9, eta_kappa 1e-5, seeds {0, 1}, and 500000 steps.

1. A fixed learning rate of 1.9 / lam_0. This is below the stability bound
   2 / lam_0 at initialisation, and the curvature only falls afterwards.
2. A learning rate re-estimated from the current curvature. Every 10000 steps,
   recompute lam_t by power iteration on the current weights and set
   eta = 1 / lam_t. Implement this as a new optional configuration key, with
   the default off, and record eta at every checkpoint.
3. A learning rate that grows with the shrinking weight norm. For a
   3-homogeneous network the curvature scales as the fourth power of the norm,
   so use eta_t = eta_0 (||theta_0|| / ||theta_t||)^4. Cap the factor at 1000.

Variant 2 changes the training rule over time. If it is the one that works,
state this clearly in the report, because it means the kernel step size is no
longer fixed.

### Stage B3. Mapping the window (up to 24 runs)

Around the best cell of B1 and B2, use 2 seeds and cross:

- initial scale {3, 5, 7, 9},
- eta_kappa {3e-6, 1e-5, 2e-5},
- the best learning-rate rule from B2.

Use 300000 steps, or 500000 for scale 7 and 9. Record which cells show strong
grokking, which show weak grokking, and which fail. Note whether a failure
means the run never fitted or the weights collapsed to zero.

### Stage B4. Robustness of the best cell (up to 12 runs)

Take the cell with the strongest reliable grokking and run each of these with
3 seeds:

- **Five more seeds** (seeds 3 to 7).
- **Full batch** (`batch_size=None`). The tangent kernel then sets the exact
  function-space step. A positive result here is the most valuable outcome
  for the project.
- **Training set size.** Use n_train in {500, 2000}.
- **Width 100**, to move one step closer to the modular addition network.

### Stage B5. Bias control (6 runs)

Run the best cell on `OmnigrokMLP` with biases, with matched seeds and the
autograd kernel. Compare the delay ratios with the bias-free runs. This shows
what the biases contribute.

### Stage B6. Kernel analysis of grokking runs

For every run that groks, record and tabulate:

- t_mem, the full level table, and the gain after memorisation;
- S_t, R_t, D_t and A_t, including `A_t_test`, at t_mem, at t_0.5, at t_0.8
  and at the end of the run;
- the per-layer alignments and scales from the closed form at the same steps;
- the weight norm ratio at the same steps;
- the step at which A_t first exceeds its value at t_mem by 0.05, 0.10 and
  0.20. Compare it with t_0.5 and t_0.8, because the earlier runs suggest that
  A_t rises after test accuracy does;
- whether S_t follows the 3-homogeneous prediction of Lewkowycz and Gur-Ari
  (2021), S_t = -4 lambda t with R_t = 0, while the training loss is near
  zero. Compare the measured slope of log kernel norm against
  -2 (k - 1) eta_kappa per step with k = 3. Before writing this up, check the
  theorem's statement in the paper as `CLAUDE.md` rule 3 asks. The S_t
  recorded in `ntk_lib` is the log of the norm ratio.

Produce one figure per grokking configuration with four panels:

1. train and test accuracy,
2. A_t and A_t_test,
3. S_t and R_t,
4. the weight norm ratio.

Use a log step axis. Follow the plot style in `ntk_lib.set_plot_style`. Use the
colours already in `ntk_lib`: train is blue and test is orange.

### Agent 1 stopping rule and handover

Option B **succeeds** if at least one configuration groks reliably, strong or
weak, under the bias-free network with the closed-form kernel. If it
succeeds, finish B4 to B6 and stop. Do not start Agent 2.

Option B **fails** if no configuration groks reliably after B1 to B3. Then
write the report, set the first line of the report to
`STATUS: OPTION B FAILED`, and hand over to Agent 2. Pass on the best partial
results and the failure pattern, for example "never fitted", "collapsed" or
"fitted but test stayed at chance".

If the closed-form verification in section 2.3 fails and cannot be fixed,
run B1 with the autograd kernel to keep the experiment moving. Mark the kernel
as unverified in the report.

Agent 1 writes its report to `repoduced-code/results/mnist_optionB/REPORT.md`.

## 5. Agent 2: the closest network that groks (only if Option B fails)

The aim is a network that groks MNIST reasonably and has a closed-form kernel.
It should be as close as possible to the modular addition network: two layers,
bias-free, ReLU, mean-field parameterisation, width 100.

Agent 2 reads `CLAUDE.md`, section 1 of this plan, and Agent 1's report first.
It reuses the groundwork of section 2. It must not repeat Agent 1's runs.

### The ladder

Climb this ladder from the closest rung. Each rung changes one thing. At each
rung, run a small search; a starting grid is given below. Move up a rung only
when the current rung fails.

| Rung | Network | Change from the rung above |
|---|---|---|
| A0 | 2 layers, bias-free, mean-field, width 100 | the modular addition network |
| A1 | 2 layers, bias-free, NTK parameterisation, width 100 | parameterisation |
| A2 | 2 layers, bias-free, standard parameterisation, width 100 | parameterisation |
| A3 | the best of A0 to A2 at width 200, 400 and 800 | width |
| A4 | 3 layers, bias-free, the parameterisation of the best 2-layer rung, width 100 | depth |

A4 differs from Option B only in width and perhaps parameterisation. If A4 is
reached, use Agent 1's results to avoid repeating its cells.

### Search grid at each rung

Start from the recipe that worked in section 1 item 4: uncentred predictor,
large initialisation, weak weight decay, SGD without momentum, batch 200.

- **Initial scale.** For a 2-homogeneous network the initial output grows as
  s^2, against s^3 for three layers. To match the output size of Option B at
  scales 3 and 9, use s in {5, 9, 15, 27}. For the mean-field rung, first
  measure the size of the initial output at scale 1, because the 1/D and 1/N
  factors make it small. Choose scales that give initial output norms
  comparable to the Option B runs that grokked.
- **eta_kappa** in {3e-6, 1e-5, 3e-5}.
- **The learning rate rule** that worked best in B2, or 1 / lam_0 if B2 found
  none. Use the same three variants of B2 if a rung stalls.
- **Seeds** {0, 1}. Use a third seed only for cells that meet the criterion.
- **Steps.** 300000, or 500000 for the largest scales.

With the closed-form kernel, a 2-layer run is cheap. Use the full budget of
12 workers.

### Levers that do not change the network

If no rung of the ladder groks, try these on the best rung found, one at a
time:

- training set size n_train in {300, 500, 2000};
- full batch;
- the centred predictor with alpha > 1 and no weight decay, the modular
  addition recipe. This is the one route that keeps the report's exact
  setup. Earlier runs suggest it will not grok, because the initial MNIST
  kernel is already aligned with the labels. One clean negative result with
  2 seeds is still worth recording.

Do not change the loss to cross-entropy, change the labels, or use a different
dataset. Those change the question.

### Agent 2 stopping rule

Stop at the first rung that groks reliably. Then run 3 more seeds and stage
B6's kernel analysis on that rung, and finish. If nothing groks after the
ladder and the levers, stop and report the best partial result for each rung,
with the failure pattern.

Agent 2 writes its report to `repoduced-code/results/mnist_fallback/REPORT.md`.

## 6. What each report contains

Write each report in plain full sentences, following `CLAUDE.md` rule 2.

1. The status line: succeeded, failed, or stopped early, with the reason.
2. The exact network, training rule and configuration of the best result,
   with the run names so that the JSON files can be found.
3. A table of every run: configuration, seed, t_mem, test accuracy at t_mem,
   best test accuracy, delay ratios at 0.5, 0.8 and 0.9, and the verdict
   (strong, weak, none). Give a failure type for each run that did not grok.
4. The verification results of section 2.3.
5. The kernel analysis of section 4 stage B6 for every grokking
   configuration, with the figure paths.
6. Differences from the modular addition setup that remain. Say which of
   these break the kernel's role as the driver of training: momentum, a
   learning rate that changes over time, and minibatches.
7. Problems met and decisions made, for example a fix to the learning-rate
   rule, with the reason for each.
8. What the user should look at first in the morning.

## 7. Time budget

These estimates assume the closed form makes a 3 layer probe kernel much
cheaper than the 1.5 s autograd kernel.

| Step | Estimate |
|---|---|
| Groundwork (section 2) | 1 to 1.5 hours |
| B1 | about 30 minutes |
| B2 | about 1.5 hours (500000 steps) |
| B3 | about 1.5 to 2 hours |
| B4 | about 1 hour |
| B5 and B6 | about 1 hour |
| Agent 1 total | about 6 to 8 hours |
| Agent 2, if needed | the time left in the night |

The 2-layer rungs are fast.

If B1 shows reliable strong grokking, B2 is optional. Spend its time on B4
instead.

Check the load before each batch with `uptime`. If the load average is above
20 because of other programs, reduce the number of workers.

## 8. Launch prompts

### Agent 1

> Read `docs/PLAN_mnist_closed_form.md` in the repository
> `STAT3007` and carry out section 0, section 2, section 3 and section 4 as
> written. You are Agent 1. Do not commit anything. Work through the stages
> in order and respect the stopping rules. Write progress to
> `repoduced-code/results/mnist_optionB/REPORT.md` after every batch. If
> Option B fails, set the first line of the report to
> `STATUS: OPTION B FAILED` and stop.

### Agent 2

Launch this only if Agent 1's report starts with `STATUS: OPTION B FAILED`.

> Read `docs/PLAN_mnist_closed_form.md` and
> `repoduced-code/results/mnist_optionB/REPORT.md` in the repository
> `STAT3007`. You are Agent 2. Carry out section 0, section 3 and section 5.
> Reuse the groundwork of section 2, which Agent 1 built. Do not commit
> anything. Write progress to
> `repoduced-code/results/mnist_fallback/REPORT.md` after every batch.
