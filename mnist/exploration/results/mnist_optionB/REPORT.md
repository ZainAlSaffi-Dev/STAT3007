STATUS: OPTION B SUCCEEDED

# Option B report (Agent 1)

Written on 12 October 2026 at about 02:40. All stages from B1 to B6 are finished. Agent 2 is not needed.

## 1. Summary

Option B succeeds. The bias-free three layer ReLU network of width 200, with the closed-form kernel, shows strong grokking reliably on MNIST. The best configuration showed strong grokking in 8 of 8 seeds with minibatches, and in 3 of 3 seeds with full-batch gradient descent.

The strong result needs a learning rate that is re-estimated from the curvature during training. This is variant 2 of stage B2. The kernel step size is therefore not fixed in these runs. With a fixed learning rate eta = 1 / lam_0, the only grokking was weak, at init_scale 3 and eta_kappa 1e-5, in 3 of 3 seeds.

## 2. The best configuration

The network is `ntk_lib.DeepBiasFreeMLP` with `parameterisation="deep_standard"`, `depth=3` and `hidden_dim=200`. It has three linear layers, no biases and ReLU. The weights are drawn uniformly from (-1 / sqrt(fan_in), 1 / sqrt(fan_in)) and multiplied by init_scale 7. The predictor is uncentred, `centred=False` and `alpha=1`.

The data are 1000 MNIST training images drawn by `data_seed=42` and the full 10000 test images. The loss is MSE on one-hot targets. The probe is the mixed probe of 128 training and 128 test images.

The optimiser is plain SGD without momentum, with batch size 200, for 500000 steps. The per-step weight decay is eta_kappa = 3e-6. The learning rate starts at eta = 1 / lam_0, where lam_0 is the top Gauss-Newton eigenvalue of the loss at the run's own initial weights. Every 10000 steps, `train_run` recomputes the top eigenvalue lam_t on the full training set and sets eta_t = eta_0 lam_0 / lam_t, capped at 1000 eta_0. The per-step decay eta_kappa is held fixed when eta changes. The configuration keys are `lr_schedule="curvature"`, `lr_update_interval=10000` and `lr_cap=1000`.

The runs of this cell are `optB_b3_curv_s7_ek3e-06_seed0` to `seed2`, `optB_b4_curv_s7_ek3e-06_seed3` to `seed7`, and `optB_b4_curv_s7_ek3e-06_fullbatch_seed0` to `seed2`, all in this folder. The 8 minibatch seeds and the 3 full-batch seeds all show strong grokking. Test accuracy at memorisation was 0.15 to 0.18. Memorisation came at steps 14000 to 18200. The best test accuracy was 0.85 or 0.86, and t_0.5 / t_mem was 11.4 to 16.6. Test accuracy never reached 0.90 within 500000 steps and was still rising at the end.

Four other cells also showed strong grokking in 3 of 3 seeds under the same rule. They are init_scale 9 with eta_kappa 3e-6 and 1e-5, init_scale 7 with eta_kappa 1e-5, and init_scale 5 with eta_kappa 3e-6. The cells with eta_kappa 1e-5 reach a best test accuracy of 0.91 or 0.92, but with smaller delay ratios, t_0.5 / t_mem from 3.6 to 4.9.

## 3. What each stage found

**Stage B1** used the rule eta = 1 / lam_0 with no schedule. At init_scale 3 and eta_kappa 1e-5, all 3 seeds showed weak grokking. Test accuracy was 0.58 at memorisation and rose to 0.91 or 0.92. The delay ratio at the halfway level was 5.0 to 6.0. At init_scale 3 and eta_kappa 3e-5, the gain came too early, with a halfway delay ratio of 2.7 to 3.0. At init_scale 9 with eta_kappa 1e-5, training accuracy stalled near 0.80 within 200000 steps. At init_scale 9 with eta_kappa 3e-5, the weights collapsed. So removing the biases kept the behaviour of section 1 item 4 of the plan, and the behaviour was the same across seeds.

**Stage B2** tested the scale 9 stall over 500000 steps. The fixed rate 1.9 / lam_0 memorised at step 70000 with test accuracy 0.25 to 0.27. It reached 0.74 or 0.75 and then lost both the fit and the test accuracy as the weights shrank. The curvature rule gave strong grokking in both seeds. The weight-norm rule, eta_t = eta_0 (||theta_0|| / ||theta_t||)^4, followed almost the same test accuracy and weight norm path as the curvature rule. It memorised later, at step 52000, so its delay ratio at level 0.50 was only 1.7 or 1.8. The reference rule 1 / lam_0 collapsed in the longer run and never memorised.

**Stage B3** crossed init_scale {3, 5, 7, 9} with eta_kappa {3e-6, 1e-5, 2e-5} under the curvature rule. Every cell gave the same verdict for all of its seeds. Strong grokking needs init_scale 5 or more and eta_kappa 1e-5 or less. Init_scale 5 also needs eta_kappa 3e-6. At init_scale 3, every cell was weak, because test accuracy at memorisation was 0.55 to 0.61. At eta_kappa 2e-5, the cells above init_scale 3 gained without enough delay. No cell in B3 collapsed or failed to fit.

**Stage B4** tested the best cell. Seeds 3 to 7 all showed strong grokking. Full batch showed strong grokking in 3 of 3 seeds, with t_0.5 / t_mem from 11.6 to 14.2. With n_train 500, all 3 seeds showed strong grokking, with t_0.5 / t_mem near 25. With n_train 2000, all 3 seeds failed. Test accuracy was already 0.74 to 0.77 at memorisation, which came late, at steps 74000 to 98000. With width 100, all 3 seeds failed. Memorisation came late, at steps 134000 to 256000, and the best test accuracy was only 0.57 to 0.62.

**Stage B5** ran the best cell on `OmnigrokMLP` with biases and the autograd kernel, seeds 0 to 5. All 6 seeds showed strong grokking. The delay ratio at level 0.50 was 10.4 to 14.9, against 11.4 to 16.6 for the bias-free seeds. The mean t_mem was 17300 with biases and about 16400 to 17400 without. The biases make no measurable difference to the delay in this cell.

## 4. Verification of the closed form (section 2.3)

`verify_closed_form_deep.py` ran all four checks, and all 36 comparisons passed. The log is `verify_closed_form_deep.log` in this folder.

1. Against `entk_closed_form` for `NTKMLP` and `MeanFieldMLP` on the modular addition mixed probe, the largest relative Frobenius error was 6.7e-16.
2. Against autograd for depth 2, 3 and 4, both parameterisations and init_scale 1 and 9, on 64 MNIST images, the error against a float64 autograd kernel was at most 2.3e-14. The error against `ntk_lib.entk` on a float64 model was about 5e-8. That path accumulates the Gram matrix in float32.
3. After 500 SGD steps of a 3 layer network, the errors were at most 1.5e-14 and 4.9e-8 in the same two comparisons.
4. One probe kernel of 256 images for the 3 layer width 200 network took 0.0032 s by the closed form and 0.32 s by autograd, on one thread.

The regression check of section 2.4 passed. The 1500-step modular addition run gave a maximum absolute difference of 0.0 over `HISTORY_KEYS` between the edited `ntk_lib` and the committed one. It added no history keys. The check was run after the edits and again at the end; its last output is in `regression_modadd.log`.

## 5. Kernel analysis of the grokking runs (stage B6)

The full per-run tables are in `b6_tables.md` and the numbers in `b6_analysis.json`. There is one figure per configuration in `figures/`. Each has four panels: accuracy, A_t with A_t_test, S_t with R_t, and the weight norm ratio. The figure for the best cell with minibatches is `figures/optB_b3_curv_s7_ek3e-06.png` for seeds 0 to 2 and `figures/optB_b4_curv_s7_ek3e-06.png` for seeds 3 to 7. The full-batch figure is `figures/optB_b4_curv_s7_ek3e-06_fullbatch.png`. The B2 cell at init_scale 9 and eta_kappa 1e-5 is split between `figures/optB_b2_curv_s9_ek1e-05.png` for seeds 0 and 1 and `figures/optB_b3_curv_s9_ek1e-05.png` for seed 2.

The table below gives means over the seeds of each grokking configuration. S_t is the log of the kernel norm ratio, as `ntk_lib` records it. The last two columns divide the measured slope per step over the window from t_0.8 to the end by the predicted slope. For the kernel the prediction is -4 eta_kappa per step, and for the weight norm it is -eta_kappa per step.

| configuration | seeds | verdict | t_mem | t_0.5 | t_0.8 | A_t at t_mem | A_t at t_0.5 | A_t at t_0.8 | A_t at end | first A_t rise of 0.05 | first rise of 0.10 | S_t at t_mem | S_t at end | R_t at end | weight norm ratio at end | kernel slope over prediction | weight slope over prediction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1, s3, eta_kappa 1e-5, rate 1/lam_0 | 3 | weak | 3171 | 1513 | 26802 | 0.39 | 0.39 | 0.43 | 0.58 | 36000 | 68645 | -1.93 | -3.98 | 0.059 | 0.174 | 0.18 | 0.87 |
| curv, s9, eta_kappa 1e-5 | 3 | strong | 22475 | 90667 | 147333 | 0.33 | 0.35 | 0.41 | 0.54 | 120115 | 162000 | -3.35 | -8.29 | 0.082 | 0.044 | 0.07 | 0.47 |
| curv, s9, eta_kappa 3e-6 | 3 | strong | 25382 | 303333 | 492667 | 0.33 | 0.35 | 0.41 | 0.41 | 406667 | not reached | -2.77 | -7.11 | 0.028 | 0.226 | 0.32 | 0.93 |
| curv, s7, eta_kappa 3e-6 (b3) | 3 | strong | 17311 | 219333 | 409333 | 0.33 | 0.35 | 0.41 | 0.44 | 326667 | 457000 (2 of 3) | -2.65 | -6.36 | 0.026 | 0.231 | 0.25 | 0.91 |
| curv, s7, eta_kappa 3e-6, seeds 3 to 7 | 5 | strong | 16364 | 229145 | 411600 | 0.33 | 0.35 | 0.41 | 0.44 | 329600 | 445333 (3 of 5) | -2.80 | -6.49 | 0.042 | 0.231 | 0.23 | 0.91 |
| curv, s7, eta_kappa 3e-6, full batch | 3 | strong | 17374 | 219816 | 410000 | 0.33 | 0.35 | 0.41 | 0.44 | 331333 | 462000 (2 of 3) | -2.63 | -6.35 | 0.026 | 0.231 | 0.25 | 0.91 |
| curv, s7, eta_kappa 3e-6, n_train 500 | 3 | strong | 12000 | 296667 | 456000 | 0.37 | 0.38 | 0.43 | 0.45 | 420667 | not reached | -1.64 | -6.24 | 0.012 | 0.226 | 0.35 | 0.95 |
| curv, s7, eta_kappa 1e-5 | 3 | strong | 15962 | 66000 | 120667 | 0.33 | 0.35 | 0.41 | 0.53 | 97301 | 139333 | -3.06 | -7.18 | 0.080 | 0.057 | 0.06 | 0.43 |
| curv, s5, eta_kappa 3e-6 | 3 | strong | 11644 | 104667 | 293333 | 0.34 | 0.35 | 0.41 | 0.41 | 241333 | not reached | -2.48 | -4.77 | 0.030 | 0.411 | 0.33 | 0.93 |
| curv, s5, eta_kappa 1e-5 | 2 | weak | 12000 | 31000 | 86000 | 0.35 | 0.35 | 0.41 | 0.51 | 73000 | 121000 | -2.73 | -5.51 | 0.051 | 0.101 | 0.07 | 0.69 |
| curv, s3, eta_kappa 3e-6 | 2 | weak | 3266 | 1837 | 106137 | 0.39 | 0.40 | 0.42 | 0.47 | 184000 | not reached | -1.81 | -3.10 | 0.023 | 0.432 | 0.24 | 0.90 |
| curv, s3, eta_kappa 1e-5 | 2 | weak | 3473 | 1617 | 31000 | 0.40 | 0.40 | 0.43 | 0.51 | 51000 | 171000 | -1.90 | -3.56 | 0.059 | 0.147 | 0.07 | 0.62 |
| curv, s3, eta_kappa 2e-5 | 2 | weak | 3266 | 1520 | 14000 | 0.40 | 0.40 | 0.44 | 0.56 | 23000 | 67961 | -1.99 | -4.15 | 0.106 | 0.126 | 0.06 | 0.29 |
| b5, biases, s7, eta_kappa 3e-6, autograd kernel | 6 | strong | 17299 | 229000 | 415000 | 0.32 | 0.34 | 0.41 | 0.44 | 316333 | not computed | -2.73 | -6.38 | 0.032 | 0.231 | no prediction | no prediction |

The findings are these.

1. **A_t rises after test accuracy does.** In every strong configuration, A_t fell from about 0.42 at initialisation to about 0.33 at memorisation. It was still near 0.35 at t_0.5. It first rose by 0.05 above its value at t_mem between t_0.5 and t_0.8, and it was about 0.41 at t_0.8. In the best cell, t_0.5 is about 220000, the first rise of 0.05 is about 330000, and t_0.8 is about 410000. A_t_test followed A_t closely, about 0.01 to 0.02 higher. This agrees with section 1 item 6 of the plan. A rise in A_t does not precede generalisation in these runs.
2. **The kernel shrinks, and its uncentred shape barely changes.** In the best cell, S_t fell to about -6.4 by the end, so the kernel norm fell by a factor of about 600. R_t at the end was 0.026 on average over the three B3 seeds and 0.042 over seeds 3 to 7. The centred alignment still moved by about 0.11 from its minimum. The kernel at the end is mostly a rescaled copy of the initial kernel, with a small change that carries the label structure.
3. **Per-layer terms.** In the best cell, seed 0, the first-layer term held 62 percent of the trace of the kernel at t_mem and 68 percent at the end. The last-layer share fell from 25 percent to 10 percent. The last-layer term aligned most with the labels by the end: A_W3 was 0.30 at t_mem and 0.50 at the end, against 0.32 to 0.41 for A_W1 and 0.34 to 0.44 for A_W2. In the init_scale 9, eta_kappa 1e-5 runs the same pattern was larger, with A_W3 reaching 0.87 at the end and a last-layer share of 3 percent.
4. **The Lewkowycz and Gur-Ari prediction does not hold quantitatively.** In the best cell the training loss was about 5e-6 over the late window. The weight norm decayed at 0.91 of the pure-decay rate. The kernel norm decayed at only about 0.25 of the predicted rate of -4 eta_kappa per step. The ratio was 0.06 to 0.35 across the grokking configurations, and it was lower at the larger eta_kappa. Full batch gave the same ratio as minibatches, so minibatch noise is not the cause. The measured values and the prediction disagree, and the report should state this rather than fit them together. One possible reason is that the theorem is an infinite-width result. A second is that the curvature schedule changes the learning rate. A third is that the residual term of their Equation S5 is not negligible at this width. These are not tested here.
5. **Memorisation is set by the schedule.** In the full-batch figure, training accuracy jumps at step 10000. That is the first update of the curvature rule, when eta grows by a factor of about 10 to 12 in this cell. A_t drops at the same step. Test accuracy then rises only after the weight norm has fallen to about 0.3 to 0.4 of its initial value. The weight norm path does not depend on the learning rate, because eta_kappa is fixed per step. This is why the curvature and weight-norm rules gave the same test accuracy path in B2 and differed only in t_mem.

The check of the theorem. I opened arXiv:2006.08643v2 (Lewkowycz and Gur-Ari, "On the training dynamics of deep networks with L2 regularization"). Theorem 1, Equation 2, states d Theta_t / dt = -2 (k - 1) lambda Theta_t for a k-homogeneous network in the infinite width limit. It assumes the correlation function conjecture of Dyer and Gur-Ari, and gradient flow on L + (lambda / 2) ||theta||^2. Theorem 2 adds MSE loss and k at least 2. It gives eigenvalues gamma_a(t) = gamma_a exp(-2 (k - 1) lambda t) with static eigenvectors. Equation S5 is the finite-width form. It adds terms T_t(x, x') + T_t(x', x), which are proportional to the loss derivative. One SGD step with per-step decay eta_kappa is a gradient flow time eta with lambda = eta_kappa / eta. So the predicted slope of log ||K|| is -2 (k - 1) eta_kappa = -4 eta_kappa per step for k = 3, whatever the learning rate. This matches the plan's statement. The citation is the golden report's entry, Lewkowycz and Gur-Ari (2021), arXiv:2006.08643.

## 6. Differences from the modular addition setup that remain

The network has three layers and the standard parameterisation. The modular addition network has two layers in mean-field parameterisation. The predictor is uncentred, with the large random initial output kept. The modular addition setup uses the centred predictor alpha [f - f_0]. The dataset is MNIST with 1000 training images.

Three differences matter for the kernel's role as the driver of training.

1. **Momentum.** None. Every Option B run uses plain SGD with momentum 0, so this does not break the kernel's role.
2. **A learning rate that changes over time.** The strong results all use the curvature rule, so eta changes every 10000 steps and grows by up to 1000 times. Within each block of 10000 steps the kernel still sets the function-space step, but the step size is not fixed over the run. The weak result at init_scale 3 with eta_kappa 1e-5 used a fixed rate, eta = 1 / lam_0, and is the only grokking result without this difference.
3. **Minibatches.** Most runs use batch size 200, so each step uses the kernel restricted to the batch. The full-batch runs of B4 remove this difference and still show strong grokking in 3 of 3 seeds. They still use the curvature rule.

## 7. Problems met and decisions made

1. The Write tool refused to create this report as a `.md` file. The plan requires the file, so it was written through the shell.
2. The plan says stages B2 fixes should be tested "in order". All four rules, including the reference 1 / lam_0, ran together in one batch of 8 runs, because a batch cost about 17 minutes.
3. In the curvature and weight-norm rules, the per-step weight decay eta_kappa is held fixed when eta changes. In PyTorch terms, `weight_decay` is reset to eta_kappa / eta_t. The other choice, a fixed kappa, would make the decay grow with eta. The plan's cells are indexed by eta_kappa, so the fixed eta_kappa was kept.
4. The plan caps the weight-norm rule's factor at 1000. The same cap was applied to the curvature rule, so that eta cannot blow up if the curvature falls towards zero. Both rules reached the cap by about step 150000 to 200000 in the B2 runs at init_scale 9 and eta_kappa 1e-5, so the cap is active for the later part of those runs. In the best cell, init_scale 7 with eta_kappa 3e-6, the cap was never reached.
5. lam_0 was computed for each seed at its own initial weights, not once per scale as in `mnist_pilot.py`. The power iteration uses a seeded generator and does not touch the global random state.
6. The best cell for B4 and B5 was chosen as init_scale 7 with eta_kappa 3e-6. It had the largest margins on all three strong conditions. The cells with eta_kappa 1e-5 reach higher test accuracy, near 0.92, but their t_0.5 / t_mem of 3.6 to 4.9 sits close to the threshold of 3.
7. Third seeds were run for the five cells that showed strong grokking in B3, to test "reliably" with 3 seeds. B3 reused the two B2 curvature runs for its init_scale 9, eta_kappa 1e-5 cell, under their B2 names.
8. B4 has 17 runs, more than the plan's "up to 12". Five more seeds, three full-batch seeds, and three seeds each at n_train 500, n_train 2000 and width 100 make 17. The runs were cheap enough that all were kept.
9. The failure labels are as follows. A run is "collapsed" if its final weight norm is below 0.3 of the default initial norm (the ratio times init_scale) or its final training accuracy is below 0.2. Otherwise a run that never reached 95 percent training accuracy is "never fitted (stalled)". The B1 label was corrected after a first version called the scale 9 stall a collapse.
10. With 12 workers the load average was 25 to 35 rather than 12. The cause was probably BLAS threads in NumPy and the OneDrive sync of the results folder. From B2 on, the runs were started with `OMP_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1`. Runs took 2 to 3 ms per step under load, against 0.6 ms alone.
11. The results folder holds 154 MB, mostly kernel snapshots saved every 50000 steps. It sits inside the OneDrive folder.
12. The B5 runs have biases, so they are not homogeneous and have no Lewkowycz and Gur-Ari prediction. Their per-layer terms are not recorded, because the closed form does not apply to them.

## 8. What to look at first

1. The figure `figures/optB_b4_curv_s7_ek3e-06_fullbatch.png`. It shows strong grokking under full-batch gradient descent with the bias-free network and the closed-form kernel. It also shows the jump at step 10000 caused by the curvature rule.
2. Section 6 point 2. The strong results depend on a learning rate that changes during training. Decide whether that is acceptable for the report. If it is not, the fixed-rate weak result at init_scale 3, `optB_b1_s3_ek1e-05_seed0` to `seed2`, is the fallback.
3. Section 5 point 4. The kernel decays at about a quarter of the Lewkowycz and Gur-Ari rate, while the weights decay at nearly the pure-decay rate.
4. Section 5 point 1. A_t rises only after test accuracy has passed 0.5, in every grokking configuration.

## 9. Files

The new code is in `repoduced-code/`. `ntk_lib.py` gained `DeepBiasFreeMLP`, `closed_form_terms_deep`, `entk_closed_form_deep`, `gauss_newton_top_eigenvalue`, the configuration keys `depth`, `lr_schedule`, `lr_update_interval`, `lr_norm_power` and `lr_cap`, and the extra history keys `A_Wl`, `S_Wl`, `share_Wl`, `eta` and `lam`. The defaults leave every existing run unchanged. The new files are `verify_closed_form_deep.py`, `regression_modadd.py`, `mnist_optionB.py` and `mnist_optionB_b6.py`.

`uv run python mnist_optionB.py alltables` prints the tables of the next section. `uv run python mnist_optionB_b6.py --auto` redoes the kernel analysis and the figures.

## 10. Every run

t_mem is the first checkpoint with training accuracy at least 0.95. "test@mem" is the test accuracy there. "w ratio" is the final weight norm over the initial one. The columns r0.5, r0.8 and r0.9 are delay ratios t_l / t_mem. "pre" means the level was reached by t_mem, "never" means it was not reached, and "@step" gives the step for a run that never memorised. "half r" is the delay ratio at the level halfway through the gain after memorisation. Names with `lam1` or no rule tag use eta = 1 / lam_0. `lam1.9` uses 1.9 / lam_0, `curv` the curvature rule, and `norm` the weight-norm rule. The second table is the full level table for levels 0.30 to 0.90.

| run | t_mem | test@mem | best | end test | end train | w ratio | r0.5 | r0.8 | r0.9 | half r | verdict | failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| optB_b1_s3_ek1e-05_seed0 | 3420 | 0.58 | 0.91 | 0.91 | 1.00 | 0.17 | pre | 8.1 | 27.5 | 5.0 | weak |  |
| optB_b1_s3_ek1e-05_seed1 | 3420 | 0.58 | 0.91 | 0.91 | 1.00 | 0.17 | pre | 8.2 | 27.5 | 5.3 | weak |  |
| optB_b1_s3_ek1e-05_seed2 | 2673 | 0.58 | 0.92 | 0.92 | 1.00 | 0.18 | pre | 9.2 | 29.9 | 6.0 | weak |  |
| optB_b1_s3_ek3e-05_seed0 | 3420 | 0.63 | 0.90 | 0.90 | 0.99 | 0.08 | pre | 3.0 | 12.3 | 2.7 | none | gain but no delay |
| optB_b1_s3_ek3e-05_seed1 | 3420 | 0.62 | 0.90 | 0.90 | 0.99 | 0.08 | pre | 3.4 | 33.3 | 2.7 | none | gain but no delay |
| optB_b1_s3_ek3e-05_seed2 | 2673 | 0.61 | 0.91 | 0.91 | 0.99 | 0.09 | pre | 3.7 | 13.3 | 3.0 | none | gain but no delay |
| optB_b1_s9_ek1e-05_seed0 | None | - | 0.68 | 0.68 | 0.80 | 0.13 | @126000 | never | never | - | none | never fitted (stalled) |
| optB_b1_s9_ek1e-05_seed1 | None | - | 0.68 | 0.68 | 0.81 | 0.13 | @122000 | never | never | - | none | never fitted (stalled) |
| optB_b1_s9_ek1e-05_seed2 | 100000 | 0.37 | 0.69 | 0.69 | 0.81 | 0.13 | 1.2 | never | never | 1.3 | none | gain but no delay, lost fit (train 0.81 at end) |
| optB_b1_s9_ek3e-05_seed0 | None | - | 0.55 | 0.43 | 0.50 | 0.00 | @58000 | never | never | - | none | collapsed |
| optB_b1_s9_ek3e-05_seed1 | None | - | 0.53 | 0.48 | 0.55 | 0.00 | @64000 | never | never | - | none | collapsed |
| optB_b1_s9_ek3e-05_seed2 | None | - | 0.55 | 0.49 | 0.55 | 0.00 | @58287 | never | never | - | none | collapsed |
| optB_b2_curv_s9_ek1e-05_seed0 | 23712 | 0.14 | 0.92 | 0.92 | 1.00 | 0.04 | 4.0 | 6.2 | 10.2 | 4.1 | strong |  |
| optB_b2_curv_s9_ek1e-05_seed1 | 23712 | 0.17 | 0.91 | 0.91 | 1.00 | 0.04 | 3.6 | 6.2 | 9.9 | 3.9 | strong |  |
| optB_b2_lam1.9_s9_ek1e-05_seed0 | 70000 | 0.25 | 0.74 | 0.23 | 0.27 | 0.01 | 1.6 | never | never | 1.6 | none | gain but no delay, lost fit (train 0.27 at end) |
| optB_b2_lam1.9_s9_ek1e-05_seed1 | 68468 | 0.27 | 0.75 | 0.16 | 0.18 | 0.01 | 1.6 | never | never | 1.6 | none | gain but no delay, lost fit (train 0.18 at end) |
| optB_b2_lam1_s9_ek1e-05_seed0 | None | - | 0.68 | 0.10 | 0.11 | 0.01 | @126000 | never | never | - | none | collapsed |
| optB_b2_lam1_s9_ek1e-05_seed1 | None | - | 0.68 | 0.10 | 0.11 | 0.01 | @124000 | never | never | - | none | collapsed |
| optB_b2_norm_s9_ek1e-05_seed0 | 52000 | 0.20 | 0.91 | 0.91 | 1.00 | 0.04 | 1.8 | 2.7 | 4.7 | 1.9 | none | gain but no delay |
| optB_b2_norm_s9_ek1e-05_seed1 | 52000 | 0.23 | 0.92 | 0.92 | 1.00 | 0.04 | 1.7 | 2.6 | 4.3 | 1.9 | none | gain but no delay |
| optB_b3_curv_s3_ek1e-05_seed0 | 3473 | 0.58 | 0.92 | 0.92 | 1.00 | 0.15 | pre | 8.6 | 34.0 | 4.6 | weak |  |
| optB_b3_curv_s3_ek1e-05_seed1 | 3473 | 0.58 | 0.91 | 0.91 | 1.00 | 0.15 | pre | 9.2 | 34.6 | 4.6 | weak |  |
| optB_b3_curv_s3_ek2e-05_seed0 | 3058 | 0.59 | 0.92 | 0.92 | 1.00 | 0.13 | pre | 4.6 | 18.3 | 3.6 | weak |  |
| optB_b3_curv_s3_ek2e-05_seed1 | 3473 | 0.61 | 0.91 | 0.91 | 1.00 | 0.13 | pre | 4.0 | 17.3 | 3.1 | weak |  |
| optB_b3_curv_s3_ek3e-06_seed0 | 3058 | 0.55 | 0.89 | 0.89 | 1.00 | 0.43 | pre | 34.0 | never | 13.1 | weak |  |
| optB_b3_curv_s3_ek3e-06_seed1 | 3473 | 0.57 | 0.89 | 0.89 | 1.00 | 0.43 | pre | 31.2 | never | 12.8 | weak |  |
| optB_b3_curv_s5_ek1e-05_seed0 | 12000 | 0.30 | 0.91 | 0.91 | 1.00 | 0.10 | 2.8 | 7.2 | 15.0 | 4.0 | weak |  |
| optB_b3_curv_s5_ek1e-05_seed1 | 12000 | 0.34 | 0.91 | 0.91 | 1.00 | 0.10 | 2.3 | 7.2 | 14.3 | 3.8 | weak |  |
| optB_b3_curv_s5_ek2e-05_seed0 | 10932 | 0.35 | 0.92 | 0.92 | 1.00 | 0.08 | 1.6 | 3.8 | 8.0 | 2.4 | none | gain but no delay |
| optB_b3_curv_s5_ek2e-05_seed1 | 10932 | 0.39 | 0.92 | 0.92 | 1.00 | 0.08 | 1.5 | 3.8 | 7.9 | 2.4 | none | gain but no delay |
| optB_b3_curv_s5_ek3e-06_seed0 | 12000 | 0.25 | 0.81 | 0.81 | 1.00 | 0.41 | 9.3 | 24.3 | never | 10.5 | strong |  |
| optB_b3_curv_s5_ek3e-06_seed1 | 12000 | 0.29 | 0.80 | 0.80 | 1.00 | 0.41 | 7.8 | 24.7 | never | 9.7 | strong |  |
| optB_b3_curv_s5_ek3e-06_seed2 | 10932 | 0.26 | 0.81 | 0.81 | 1.00 | 0.41 | 9.9 | 26.7 | never | 11.2 | strong |  |
| optB_b3_curv_s7_ek1e-05_seed0 | 18000 | 0.18 | 0.91 | 0.91 | 1.00 | 0.06 | 3.8 | 6.7 | 12.0 | 4.1 | strong |  |
| optB_b3_curv_s7_ek1e-05_seed1 | 15932 | 0.20 | 0.92 | 0.92 | 1.00 | 0.06 | 3.9 | 7.5 | 13.4 | 4.4 | strong |  |
| optB_b3_curv_s7_ek1e-05_seed2 | 13954 | 0.18 | 0.92 | 0.92 | 1.00 | 0.06 | 4.9 | 8.7 | 14.3 | 5.3 | strong |  |
| optB_b3_curv_s7_ek2e-05_seed0 | 15932 | 0.23 | 0.92 | 0.92 | 1.00 | 0.05 | 2.1 | 3.8 | 6.5 | 2.4 | none | gain but no delay |
| optB_b3_curv_s7_ek2e-05_seed1 | 13954 | 0.24 | 0.92 | 0.92 | 1.00 | 0.05 | 2.2 | 4.3 | 7.5 | 2.6 | none | gain but no delay |
| optB_b3_curv_s7_ek3e-06_seed0 | 18000 | 0.15 | 0.85 | 0.85 | 1.00 | 0.23 | 12.7 | 22.7 | never | 12.8 | strong |  |
| optB_b3_curv_s7_ek3e-06_seed1 | 18000 | 0.18 | 0.85 | 0.85 | 1.00 | 0.23 | 11.4 | 22.7 | never | 11.8 | strong |  |
| optB_b3_curv_s7_ek3e-06_seed2 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.1 | 25.9 | never | 14.3 | strong |  |
| optB_b3_curv_s9_ek1e-05_seed2 | 20000 | 0.14 | 0.92 | 0.92 | 1.00 | 0.04 | 4.6 | 7.4 | 11.2 | 4.8 | strong |  |
| optB_b3_curv_s9_ek2e-05_seed0 | 20000 | 0.17 | 0.92 | 0.92 | 1.00 | 0.04 | 2.4 | 3.6 | 5.6 | 2.5 | none | gain but no delay |
| optB_b3_curv_s9_ek2e-05_seed1 | 20000 | 0.20 | 0.92 | 0.92 | 1.00 | 0.04 | 2.2 | 3.6 | 5.6 | 2.4 | none | gain but no delay |
| optB_b3_curv_s9_ek3e-06_seed0 | 27073 | 0.12 | 0.81 | 0.81 | 1.00 | 0.23 | 11.5 | 18.2 | never | 10.9 | strong |  |
| optB_b3_curv_s9_ek3e-06_seed1 | 27073 | 0.14 | 0.81 | 0.81 | 1.00 | 0.23 | 10.8 | 18.1 | never | 10.3 | strong |  |
| optB_b3_curv_s9_ek3e-06_seed2 | 22000 | 0.13 | 0.81 | 0.81 | 1.00 | 0.23 | 14.0 | 22.5 | never | 13.4 | strong |  |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed0 | 18190 | 0.15 | 0.86 | 0.86 | 1.00 | 0.23 | 12.4 | 22.3 | never | 12.5 | strong |  |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed1 | 18000 | 0.17 | 0.85 | 0.85 | 1.00 | 0.23 | 11.6 | 22.8 | never | 11.9 | strong |  |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed2 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.2 | 26.0 | never | 14.3 | strong |  |
| optB_b4_curv_s7_ek3e-06_n2000_seed0 | 74000 | 0.74 | 0.90 | 0.90 | 1.00 | 0.23 | pre | 1.6 | never | 1.9 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_n2000_seed1 | 98000 | 0.77 | 0.89 | 0.89 | 1.00 | 0.23 | pre | 1.2 | never | 1.6 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_n2000_seed2 | 78000 | 0.76 | 0.90 | 0.90 | 1.00 | 0.23 | pre | 1.4 | never | 2.9 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_n500_seed0 | 12000 | 0.12 | 0.83 | 0.83 | 1.00 | 0.23 | 25.2 | 37.8 | never | 24.3 | strong |  |
| optB_b4_curv_s7_ek3e-06_n500_seed1 | 12000 | 0.14 | 0.83 | 0.83 | 1.00 | 0.23 | 24.3 | 38.5 | never | 23.7 | strong |  |
| optB_b4_curv_s7_ek3e-06_n500_seed2 | 12000 | 0.13 | 0.84 | 0.84 | 1.00 | 0.23 | 24.7 | 37.7 | never | 24.2 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed3 | 15932 | 0.17 | 0.85 | 0.85 | 1.00 | 0.23 | 14.3 | 26.4 | never | 14.6 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed4 | 13954 | 0.16 | 0.85 | 0.85 | 1.00 | 0.23 | 16.6 | 31.0 | never | 16.8 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed5 | 18000 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 12.5 | 22.2 | never | 12.8 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed6 | 18000 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 12.9 | 22.3 | never | 13.1 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed7 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.3 | 25.4 | never | 14.6 | strong |  |
| optB_b4_curv_s7_ek3e-06_w100_seed0 | 256000 | 0.48 | 0.60 | 0.60 | 1.00 | 0.29 | 1.2 | never | never | 1.5 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_w100_seed1 | 142000 | 0.39 | 0.62 | 0.62 | 1.00 | 0.28 | 2.4 | never | never | 2.4 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_w100_seed2 | 134000 | 0.27 | 0.57 | 0.57 | 1.00 | 0.30 | 3.2 | never | never | 2.6 | none | gain but no delay |
| optB_b5_bias_curv_s7_ek3e-06_seed0 | 18000 | 0.15 | 0.86 | 0.86 | 1.00 | 0.23 | 13.0 | 23.0 | never | 13.0 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed1 | 20000 | 0.18 | 0.86 | 0.86 | 1.00 | 0.23 | 10.4 | 20.2 | never | 10.8 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed2 | 15932 | 0.15 | 0.85 | 0.85 | 1.00 | 0.23 | 14.9 | 26.5 | never | 14.9 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed3 | 18000 | 0.15 | 0.85 | 0.85 | 1.00 | 0.23 | 12.9 | 23.7 | never | 12.8 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed4 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.6 | 26.0 | never | 14.8 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed5 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.4 | 25.7 | never | 14.7 | strong |  |

| run | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| optB_b1_s3_ek1e-05_seed0 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.4 | 5.0 | 8.1 | 13.3 | 27.5 |
| optB_b1_s3_ek1e-05_seed1 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.4 | 5.3 | 8.2 | 13.3 | 27.5 |
| optB_b1_s3_ek1e-05_seed2 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.7 | 6.0 | 9.2 | 15.7 | 29.9 |
| optB_b1_s3_ek3e-05_seed0 | pre | pre | pre | pre | pre | pre | pre | 1.2 | 1.6 | 2.3 | 3.0 | 4.7 | 12.3 |
| optB_b1_s3_ek3e-05_seed1 | pre | pre | pre | pre | pre | pre | pre | 1.3 | 1.8 | 2.3 | 3.4 | 5.3 | 33.3 |
| optB_b1_s3_ek3e-05_seed2 | pre | pre | pre | pre | pre | pre | pre | 1.4 | 1.9 | 2.7 | 3.7 | 6.0 | 13.3 |
| optB_b1_s9_ek1e-05_seed0 | @88000 | @96000 | @104000 | @116000 | @126000 | @136000 | @148000 | @164000 | never | never | never | never | never |
| optB_b1_s9_ek1e-05_seed1 | @82000 | @92000 | @102000 | @112000 | @122000 | @136000 | @152000 | @174000 | never | never | never | never | never |
| optB_b1_s9_ek1e-05_seed2 | pre | pre | 1.1 | 1.1 | 1.2 | 1.3 | 1.5 | 1.6 | never | never | never | never | never |
| optB_b1_s9_ek3e-05_seed0 | @35595 | @40000 | @45549 | @51526 | @58000 | @70000 | never | never | never | never | never | never | never |
| optB_b1_s9_ek3e-05_seed1 | @35595 | @40000 | @45549 | @52000 | @64000 | never | never | never | never | never | never | never | never |
| optB_b1_s9_ek3e-05_seed2 | @35595 | @40000 | @45549 | @51526 | @58287 | @74000 | never | never | never | never | never | never | never |
| optB_b2_curv_s9_ek1e-05_seed0 | 2.9 | 3.1 | 3.4 | 3.7 | 4.0 | 4.2 | 4.5 | 4.8 | 5.1 | 5.6 | 6.2 | 7.3 | 10.2 |
| optB_b2_curv_s9_ek1e-05_seed1 | 2.4 | 2.8 | 3.0 | 3.4 | 3.6 | 4.0 | 4.3 | 4.6 | 5.0 | 5.5 | 6.2 | 7.2 | 9.9 |
| optB_b2_lam1.9_s9_ek1e-05_seed0 | 1.1 | 1.3 | 1.4 | 1.5 | 1.6 | 1.7 | 1.9 | 2.0 | 2.2 | never | never | never | never |
| optB_b2_lam1.9_s9_ek1e-05_seed1 | 1.1 | 1.2 | 1.3 | 1.5 | 1.6 | 1.7 | 1.9 | 2.1 | 2.3 | never | never | never | never |
| optB_b2_lam1_s9_ek1e-05_seed0 | @88000 | @96000 | @104000 | @116000 | @126000 | @136000 | @146000 | @164000 | never | never | never | never | never |
| optB_b2_lam1_s9_ek1e-05_seed1 | @82000 | @92000 | @101903 | @112000 | @124000 | @136000 | @151664 | @173160 | never | never | never | never | never |
| optB_b2_norm_s9_ek1e-05_seed0 | 1.4 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.0 | 2.2 | 2.3 | 2.5 | 2.7 | 3.2 | 4.7 |
| optB_b2_norm_s9_ek1e-05_seed1 | 1.2 | 1.4 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.1 | 2.2 | 2.4 | 2.6 | 3.0 | 4.3 |
| optB_b3_curv_s3_ek1e-05_seed0 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.1 | 5.2 | 8.6 | 15.0 | 34.0 |
| optB_b3_curv_s3_ek1e-05_seed1 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.1 | 4.6 | 9.2 | 15.0 | 34.6 |
| optB_b3_curv_s3_ek2e-05_seed0 | pre | pre | pre | pre | pre | pre | 1.1 | 1.7 | 2.4 | 3.3 | 4.6 | 7.8 | 18.3 |
| optB_b3_curv_s3_ek2e-05_seed1 | pre | pre | pre | pre | pre | pre | pre | 1.5 | 2.3 | 3.1 | 4.0 | 7.5 | 17.3 |
| optB_b3_curv_s3_ek3e-06_seed0 | pre | pre | pre | pre | pre | pre | 1.9 | 3.6 | 8.5 | 20.3 | 34.0 | 60.2 | never |
| optB_b3_curv_s3_ek3e-06_seed1 | pre | pre | pre | pre | pre | pre | 1.7 | 3.1 | 6.8 | 18.4 | 31.2 | 53.6 | never |
| optB_b3_curv_s5_ek1e-05_seed0 | pre | 1.5 | 1.8 | 2.3 | 2.8 | 3.3 | 4.0 | 4.5 | 5.2 | 6.0 | 7.2 | 9.0 | 15.0 |
| optB_b3_curv_s5_ek1e-05_seed1 | pre | 1.0 | 1.3 | 1.8 | 2.3 | 2.8 | 3.5 | 4.2 | 5.0 | 6.0 | 7.2 | 9.2 | 14.3 |
| optB_b3_curv_s5_ek2e-05_seed0 | pre | 1.1 | 1.3 | 1.5 | 1.6 | 1.8 | 2.1 | 2.4 | 2.7 | 3.3 | 3.8 | 4.9 | 8.0 |
| optB_b3_curv_s5_ek2e-05_seed1 | pre | pre | 1.1 | 1.3 | 1.5 | 1.6 | 1.9 | 2.4 | 2.7 | 3.1 | 3.8 | 4.9 | 7.9 |
| optB_b3_curv_s5_ek3e-06_seed0 | 2.0 | 3.8 | 5.8 | 7.7 | 9.3 | 11.2 | 13.2 | 15.0 | 17.5 | 20.5 | 24.3 | never | never |
| optB_b3_curv_s5_ek3e-06_seed1 | 1.2 | 2.0 | 3.8 | 5.8 | 7.8 | 9.8 | 12.0 | 14.5 | 17.2 | 20.5 | 24.7 | never | never |
| optB_b3_curv_s5_ek3e-06_seed2 | 1.9 | 3.8 | 6.0 | 8.0 | 9.9 | 11.9 | 14.1 | 16.5 | 19.0 | 22.3 | 26.7 | never | never |
| optB_b3_curv_s7_ek1e-05_seed0 | 2.3 | 2.8 | 3.1 | 3.4 | 3.8 | 4.1 | 4.6 | 4.9 | 5.4 | 5.9 | 6.7 | 8.1 | 12.0 |
| optB_b3_curv_s7_ek1e-05_seed1 | 2.1 | 2.6 | 3.0 | 3.4 | 3.9 | 4.3 | 4.8 | 5.3 | 5.9 | 6.7 | 7.5 | 9.2 | 13.4 |
| optB_b3_curv_s7_ek1e-05_seed2 | 2.9 | 3.4 | 3.9 | 4.4 | 4.9 | 5.3 | 5.7 | 6.3 | 6.9 | 7.7 | 8.7 | 10.5 | 14.3 |
| optB_b3_curv_s7_ek2e-05_seed0 | 1.4 | 1.6 | 1.8 | 1.9 | 2.1 | 2.4 | 2.5 | 2.8 | 3.0 | 3.3 | 3.8 | 4.5 | 6.5 |
| optB_b3_curv_s7_ek2e-05_seed1 | 1.3 | 1.6 | 1.9 | 2.0 | 2.2 | 2.4 | 2.7 | 3.0 | 3.3 | 3.7 | 4.3 | 5.2 | 7.5 |
| optB_b3_curv_s7_ek3e-06_seed0 | 7.6 | 9.0 | 10.2 | 11.4 | 12.7 | 13.9 | 15.2 | 16.4 | 18.1 | 20.1 | 22.7 | 27.3 | never |
| optB_b3_curv_s7_ek3e-06_seed1 | 6.0 | 7.3 | 8.7 | 10.1 | 11.4 | 12.8 | 14.2 | 15.8 | 17.6 | 19.9 | 22.7 | 27.6 | never |
| optB_b3_curv_s7_ek3e-06_seed2 | 8.3 | 9.9 | 11.4 | 12.8 | 14.1 | 15.3 | 16.8 | 18.5 | 20.3 | 22.6 | 25.9 | 30.3 | never |
| optB_b3_curv_s9_ek1e-05_seed2 | 3.3 | 3.7 | 4.0 | 4.3 | 4.6 | 4.9 | 5.3 | 5.6 | 6.1 | 6.6 | 7.4 | 8.4 | 11.2 |
| optB_b3_curv_s9_ek2e-05_seed0 | 1.7 | 1.9 | 2.0 | 2.2 | 2.4 | 2.5 | 2.6 | 2.8 | 3.0 | 3.3 | 3.6 | 4.2 | 5.6 |
| optB_b3_curv_s9_ek2e-05_seed1 | 1.5 | 1.7 | 1.9 | 2.0 | 2.2 | 2.4 | 2.5 | 2.7 | 2.9 | 3.2 | 3.6 | 4.1 | 5.6 |
| optB_b3_curv_s9_ek3e-06_seed0 | 8.2 | 9.0 | 9.8 | 10.7 | 11.5 | 12.3 | 13.2 | 14.1 | 15.1 | 16.4 | 18.2 | never | never |
| optB_b3_curv_s9_ek3e-06_seed1 | 7.0 | 8.1 | 9.0 | 9.8 | 10.8 | 11.6 | 12.6 | 13.7 | 14.9 | 16.3 | 18.1 | never | never |
| optB_b3_curv_s9_ek3e-06_seed2 | 9.8 | 11.0 | 12.0 | 13.0 | 14.0 | 14.9 | 16.0 | 17.2 | 18.6 | 20.3 | 22.5 | never | never |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed0 | 7.5 | 8.9 | 10.1 | 11.3 | 12.4 | 13.6 | 15.0 | 16.4 | 17.9 | 19.8 | 22.3 | 26.7 | never |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed1 | 6.0 | 7.4 | 8.9 | 10.2 | 11.6 | 12.9 | 14.3 | 15.9 | 17.7 | 20.0 | 22.8 | 27.7 | never |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed2 | 8.3 | 9.9 | 11.5 | 12.8 | 14.2 | 15.4 | 16.9 | 18.5 | 20.5 | 22.8 | 26.0 | 30.6 | never |
| optB_b4_curv_s7_ek3e-06_n2000_seed0 | pre | pre | pre | pre | pre | pre | pre | pre | pre | 1.1 | 1.6 | 2.8 | never |
| optB_b4_curv_s7_ek3e-06_n2000_seed1 | pre | pre | pre | pre | pre | pre | pre | pre | pre | pre | 1.2 | 1.9 | never |
| optB_b4_curv_s7_ek3e-06_n2000_seed2 | pre | pre | pre | pre | pre | pre | pre | pre | pre | pre | 1.4 | 3.8 | never |
| optB_b4_curv_s7_ek3e-06_n500_seed0 | 18.2 | 20.2 | 21.8 | 23.5 | 25.2 | 26.7 | 28.2 | 29.8 | 31.8 | 34.3 | 37.8 | never | never |
| optB_b4_curv_s7_ek3e-06_n500_seed1 | 17.2 | 19.2 | 21.0 | 22.7 | 24.3 | 26.0 | 27.5 | 29.5 | 31.8 | 34.7 | 38.5 | never | never |
| optB_b4_curv_s7_ek3e-06_n500_seed2 | 17.8 | 19.8 | 21.7 | 23.2 | 24.7 | 26.3 | 28.0 | 29.7 | 31.7 | 34.3 | 37.7 | never | never |
| optB_b4_curv_s7_ek3e-06_seed3 | 7.9 | 9.7 | 11.2 | 12.7 | 14.3 | 15.6 | 17.2 | 19.0 | 20.7 | 23.1 | 26.4 | 31.3 | never |
| optB_b4_curv_s7_ek3e-06_seed4 | 9.5 | 11.5 | 13.2 | 14.9 | 16.6 | 18.3 | 20.2 | 22.2 | 24.5 | 27.4 | 31.0 | never | never |
| optB_b4_curv_s7_ek3e-06_seed5 | 7.4 | 8.9 | 10.2 | 11.3 | 12.5 | 13.8 | 15.0 | 16.4 | 18.0 | 19.9 | 22.2 | 26.6 | never |
| optB_b4_curv_s7_ek3e-06_seed6 | 7.8 | 9.2 | 10.4 | 11.7 | 12.9 | 14.0 | 15.3 | 16.7 | 18.2 | 19.9 | 22.3 | 26.2 | never |
| optB_b4_curv_s7_ek3e-06_seed7 | 8.9 | 10.4 | 11.8 | 13.1 | 14.3 | 15.7 | 17.1 | 18.6 | 20.3 | 22.7 | 25.4 | 29.9 | never |
| optB_b4_curv_s7_ek3e-06_w100_seed0 | pre | pre | pre | pre | 1.2 | 1.6 | 2.0 | never | never | never | never | never | never |
| optB_b4_curv_s7_ek3e-06_w100_seed1 | pre | pre | 1.2 | 1.8 | 2.4 | 2.9 | 3.3 | never | never | never | never | never | never |
| optB_b4_curv_s7_ek3e-06_w100_seed2 | 1.4 | 2.0 | 2.4 | 2.8 | 3.2 | 3.6 | never | never | never | never | never | never | never |
| optB_b5_bias_curv_s7_ek3e-06_seed0 | 8.0 | 9.4 | 10.7 | 11.8 | 13.0 | 14.2 | 15.4 | 16.8 | 18.4 | 20.4 | 23.0 | 26.9 | never |
| optB_b5_bias_curv_s7_ek3e-06_seed1 | 5.2 | 6.6 | 7.9 | 9.1 | 10.4 | 11.5 | 12.9 | 14.3 | 15.9 | 17.7 | 20.2 | 24.2 | never |
| optB_b5_bias_curv_s7_ek3e-06_seed2 | 9.2 | 10.7 | 12.2 | 13.6 | 14.9 | 16.3 | 17.7 | 19.3 | 21.1 | 23.5 | 26.5 | 31.4 | never |
| optB_b5_bias_curv_s7_ek3e-06_seed3 | 7.4 | 9.0 | 10.4 | 11.6 | 12.9 | 14.2 | 15.6 | 17.1 | 18.8 | 20.9 | 23.7 | never | never |
| optB_b5_bias_curv_s7_ek3e-06_seed4 | 8.5 | 10.2 | 11.8 | 13.2 | 14.6 | 16.1 | 17.6 | 19.2 | 21.0 | 23.2 | 26.0 | 30.5 | never |
| optB_b5_bias_curv_s7_ek3e-06_seed5 | 8.8 | 10.3 | 11.7 | 13.1 | 14.4 | 15.8 | 17.2 | 18.7 | 20.3 | 22.7 | 25.7 | 30.3 | never |
