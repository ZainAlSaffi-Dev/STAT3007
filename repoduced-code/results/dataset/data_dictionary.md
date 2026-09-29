# Data dictionary

This file is written by `dataset_sweep.py assemble` from `dataset_features.COLUMNS` and `dataset_sweep.RUN_COLUMNS`. Edit those, not this file.

`checkpoints.parquet` has one row per run and checkpoint. `runs.parquet` and `runs.csv` have one row per run. The two tables join on `run_id`. The weights at about 50 checkpoints per run are in `runs/<run_id>_weights.npz`, with arrays `steps`, `W1` of shape (T, N, 2p) and `W2` of shape (T, p, N). `ntk_lib.entk_closed_form` rebuilds any kernel from them.

Notation. H = I - (1/n) 1 1^T. Kc = H K H is the centred kernel and k = Kc / ||Kc||_F the unit centred kernel. Y is the one-hot label matrix and G = H Y Y^T H. <A, B> is the Frobenius inner product. The probe is the mixed probe of 203 training and 53 test pairs.

## checkpoints

### ids

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `run_id` | The name of the run, the grid cell name of ntk_lib.cell_name. | - | ntk_lib.cell_name |
| `alpha` | The laziness knob alpha of the rescaled predictor alpha (f - f_0). | - | Kumar et al. (2024), Appendix 8.1, Equation 7, arXiv v3 |
| `width` | The hidden width N. | - | - |
| `eta_kappa` | The weight decay as the product of learning rate and decay, eta lambda in the report. | - | docs/report.tex, The laziness and weight-decay knobs |
| `seed` | The model seed. The data seed is 42 for every run. | - | - |

### grid

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `step` | The checkpoint step. | - | - |
| `log10_step1` | Base 10 logarithm of the step plus one. | log10(step + 1) | - |
| `dt_prev` | Steps since the previous checkpoint. NaN at step 0. | - | - |
| `on_linear_grid` | True if the step is a multiple of the linear checkpoint interval of 1,000 steps. | - | - |
| `on_log_grid` | True if the step is one of the log-spaced checkpoints. | - | - |
| `wall_seconds` | Wall-clock seconds since the run started. It depends on the machine and the load. | - | - |

### performance

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `train_loss` | Mean squared error of the predictor on the train set. | - | ntk_lib.train_run history, copied at the checkpoint |
| `train_acc` | Accuracy of the predictor on the train set, by the largest output. | - | ntk_lib.train_run history, copied at the checkpoint |
| `test_loss` | Mean squared error of the predictor on the test set. | - | ntk_lib.train_run history, copied at the checkpoint |
| `test_acc` | Accuracy of the predictor on the test set, by the largest output. | - | ntk_lib.train_run history, copied at the checkpoint |
| `train_margin_mean` | Mean over the train set of the correct output minus the largest other output. | mean_i (f_i[y_i] - max_{c != y_i} f_i[c]) | The group's own definition |
| `train_margin_min` | Smallest margin over the train set. | - | The group's own definition |
| `train_correct_out_mean` | Mean output at the correct class on the train set. The MSE target there is one. | - | The group's own definition |
| `train_wrong_out_rms` | Root mean square output at the wrong classes on the train set. The MSE target there is zero. | - | The group's own definition |
| `train_out_rms` | Root mean square of all outputs on the train set, the output scale. | - | The group's own definition |
| `train_out_max_mean` | Mean over the train set of the largest output. | - | The group's own definition |
| `test_margin_mean` | Mean over the test set of the correct output minus the largest other output. | mean_i (f_i[y_i] - max_{c != y_i} f_i[c]) | The group's own definition |
| `test_margin_min` | Smallest margin over the test set. | - | The group's own definition |
| `test_correct_out_mean` | Mean output at the correct class on the test set. The MSE target there is one. | - | The group's own definition |
| `test_wrong_out_rms` | Root mean square output at the wrong classes on the test set. The MSE target there is zero. | - | The group's own definition |
| `test_out_rms` | Root mean square of all outputs on the test set, the output scale. | - | The group's own definition |
| `test_out_max_mean` | Mean over the test set of the largest output. | - | The group's own definition |

### optimisation

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `grad_norm` | Norm of the gradient of the training loss, without the decay term. | \|\|dL/dtheta\|\| | The group's own definition |
| `grad_norm_W1` | Norm of the gradient of the training loss with respect to W1. | - | The group's own definition |
| `grad_norm_W2` | Norm of the gradient of the training loss with respect to W2. | - | The group's own definition |
| `update_norm` | Norm of the next gradient-descent step, decay included. | lr \|\|g + wd theta\|\| with lr = eta_0 / alpha^2 and wd = eta_kappa / lr | ntk_lib.train_run |
| `grad_weight_cos` | Cosine between the loss gradient and the weights. Near minus one at a point where decay and gradient cancel. | <g, theta> / (\|\|g\|\| \|\|theta\|\|) | The group's own definition |
| `grad_decay_ratio` | Size of the loss gradient over the size of the decay pull. NaN without decay. | \|\|g\|\| / (wd \|\|theta\|\|) | The group's own definition |
| `decay_residual` | How far the gradient and the decay pull are from cancelling. Zero at a fixed point of gradient descent with decay, one when the two are orthogonal. | \|\|g + wd theta\|\| / (\|\|g\|\| + wd \|\|theta\|\|) | The group's own definition |
| `stationary` | True if the run has decay and decay_residual is below 0.001. | - | The group's own definition |
| `sharpness` | Largest eigenvalue of the Hessian of the training loss, without the decay term, by 20 power iterations with Hessian-vector products, warm started. | lambda_max(d^2 L / d theta^2) | The group's own definition |
| `lr_sharpness` | Learning rate times sharpness. Gradient descent on a quadratic with this curvature is stable when it is below 2. | lr lambda_max | The group's own definition |

### weights

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `weight_norm` | Norm of all the parameters. | - | ntk_lib.train_run history, copied at the checkpoint |
| `param_dist` | Relative parameter movement. | \|\|theta_t - theta_0\|\| / \|\|theta_0\|\| | ntk_lib.train_run history, copied at the checkpoint |
| `log_weight_norm_ratio` | Log of the weight norm over its value at step 0. The kernel of a 2-homogeneous network rescaled by c has norm c^2 times larger, so twice this value is the scale term a uniform rescaling of the weights would give. | log(\|\|theta_t\|\| / \|\|theta_0\|\|) | The group's own definition |
| `W1_norm` | Frobenius norm of W1. | - | The group's own definition |
| `W1_dist` | Relative movement of W1. | \|\|W1_t - W1_0\|\| / \|\|W1_0\|\| | The group's own definition |
| `W1_stable_rank` | Stable rank of W1. | \|\|W\|\|_F^2 / sigma_max^2 | The group's own definition |
| `W1_eff_rank` | Effective rank of W1, the exponential of the entropy of its singular values normalised to sum to one. | exp(-sum q log q), q = sigma / sum sigma | The group's own definition |
| `W2_norm` | Frobenius norm of W2. | - | The group's own definition |
| `W2_dist` | Relative movement of W2. | \|\|W2_t - W2_0\|\| / \|\|W2_0\|\| | The group's own definition |
| `W2_stable_rank` | Stable rank of W2. | \|\|W\|\|_F^2 / sigma_max^2 | The group's own definition |
| `W2_eff_rank` | Effective rank of W2, the exponential of the entropy of its singular values normalised to sum to one. | exp(-sum q log q), q = sigma / sum sigma | The group's own definition |
| `layer_imbalance` | Difference of the squared layer norms over the squared total norm. Gradient flow without decay keeps \|\|W1\|\|^2 - \|\|W2\|\|^2 fixed for this bias-free ReLU network. With decay it shrinks by (1 - eta_kappa)^2 per step, so it mostly records decay. | (\|\|W1\|\|^2 - \|\|W2\|\|^2) / \|\|theta\|\|^2 | The group's own definition |
| `dead_frac` | Fraction of hidden units that are inactive on every training input. | - | The group's own definition |
| `active_frac` | Mean over training inputs of the fraction of hidden units that are active. | - | The group's own definition |

### fourier

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `fourier_W_a_topshare` | Mean over hidden units, weighted by the unit's norm, of the share of nonzero-frequency power at the unit's largest frequency, for the a-half of W1 along the token axis, in the weights. Frequencies k and p - k are folded together. | sum_i w_i max_k P_ik / sum_k P_ik / sum_i w_i | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_W_a_neff` | Effective number of nonzero frequencies in the pooled power of the a-half of W1 along the token axis, in the weights. | exp(entropy of sum_i P_ik / sum_ik P_ik) | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_W_a_dcshare` | Share of the power at frequency zero for the a-half of W1 along the token axis, in the weights. | sum_i P_i0 / sum_ik P_ik over all k | The group's own definition |
| `fourier_W_b_topshare` | Mean over hidden units, weighted by the unit's norm, of the share of nonzero-frequency power at the unit's largest frequency, for the b-half of W1 along the token axis, in the weights. Frequencies k and p - k are folded together. | sum_i w_i max_k P_ik / sum_k P_ik / sum_i w_i | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_W_b_neff` | Effective number of nonzero frequencies in the pooled power of the b-half of W1 along the token axis, in the weights. | exp(entropy of sum_i P_ik / sum_ik P_ik) | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_W_b_dcshare` | Share of the power at frequency zero for the b-half of W1 along the token axis, in the weights. | sum_i P_i0 / sum_ik P_ik over all k | The group's own definition |
| `fourier_W_c_topshare` | Mean over hidden units, weighted by the unit's norm, of the share of nonzero-frequency power at the unit's largest frequency, for W2 along the class axis, in the weights. Frequencies k and p - k are folded together. | sum_i w_i max_k P_ik / sum_k P_ik / sum_i w_i | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_W_c_neff` | Effective number of nonzero frequencies in the pooled power of W2 along the class axis, in the weights. | exp(entropy of sum_i P_ik / sum_ik P_ik) | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_W_c_dcshare` | Share of the power at frequency zero for W2 along the class axis, in the weights. | sum_i P_i0 / sum_ik P_ik over all k | The group's own definition |
| `fourier_W_match` | Weighted fraction of hidden units whose largest frequency is the same for a, b and the output class, in the weights. | - | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_W_phase` | Weighted mean of cos(phi_a + phi_b - phi_c) over the matched units, with the phases of the DFT at the shared frequency, in the weights. Equation 12 of Gromov (2023) makes this one for the exact solution. NaN if no unit matches. | - | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_dW_a_topshare` | Mean over hidden units, weighted by the unit's norm, of the share of nonzero-frequency power at the unit's largest frequency, for the a-half of W1 along the token axis, in the change of the weights since step 0. Frequencies k and p - k are folded together. | sum_i w_i max_k P_ik / sum_k P_ik / sum_i w_i | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_dW_a_neff` | Effective number of nonzero frequencies in the pooled power of the a-half of W1 along the token axis, in the change of the weights since step 0. | exp(entropy of sum_i P_ik / sum_ik P_ik) | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_dW_a_dcshare` | Share of the power at frequency zero for the a-half of W1 along the token axis, in the change of the weights since step 0. | sum_i P_i0 / sum_ik P_ik over all k | The group's own definition |
| `fourier_dW_b_topshare` | Mean over hidden units, weighted by the unit's norm, of the share of nonzero-frequency power at the unit's largest frequency, for the b-half of W1 along the token axis, in the change of the weights since step 0. Frequencies k and p - k are folded together. | sum_i w_i max_k P_ik / sum_k P_ik / sum_i w_i | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_dW_b_neff` | Effective number of nonzero frequencies in the pooled power of the b-half of W1 along the token axis, in the change of the weights since step 0. | exp(entropy of sum_i P_ik / sum_ik P_ik) | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_dW_b_dcshare` | Share of the power at frequency zero for the b-half of W1 along the token axis, in the change of the weights since step 0. | sum_i P_i0 / sum_ik P_ik over all k | The group's own definition |
| `fourier_dW_c_topshare` | Mean over hidden units, weighted by the unit's norm, of the share of nonzero-frequency power at the unit's largest frequency, for W2 along the class axis, in the change of the weights since step 0. Frequencies k and p - k are folded together. | sum_i w_i max_k P_ik / sum_k P_ik / sum_i w_i | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_dW_c_neff` | Effective number of nonzero frequencies in the pooled power of W2 along the class axis, in the change of the weights since step 0. | exp(entropy of sum_i P_ik / sum_ik P_ik) | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_dW_c_dcshare` | Share of the power at frequency zero for W2 along the class axis, in the change of the weights since step 0. | sum_i P_i0 / sum_ik P_ik over all k | The group's own definition |
| `fourier_dW_match` | Weighted fraction of hidden units whose largest frequency is the same for a, b and the output class, in the change of the weights since step 0. | - | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |
| `fourier_dW_phase` | Weighted mean of cos(phi_a + phi_b - phi_c) over the matched units, with the phases of the DFT at the shared frequency, in the change of the weights since step 0. Equation 12 of Gromov (2023) makes this one for the exact solution. NaN if no unit matches. | - | Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation |

### kernel_raw

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_t` | Scale term on the raw probe kernel. | log(\|\|K_t\|\|_F / \|\|K_0\|\|_F) | ntk_lib.train_run history, copied at the checkpoint |
| `R_t` | Rotation term on the raw probe kernel. | 1 - <K_t / \|\|K_t\|\|, K_0 / \|\|K_0\|\|> | ntk_lib.train_run history, copied at the checkpoint |
| `K_norm` | Frobenius norm of the raw probe kernel. K is the kernel of f, not of the rescaled predictor. With lr = eta_0 / alpha^2 the step in function space is eta_0 K at every alpha, so this can be compared across alpha. | - | ntk_lib.train_run history, copied at the checkpoint |
| `yKy` | y^T K^+ y on the probe set, summed over the label columns. | - | ntk_lib.train_run history, copied at the checkpoint |

### kernel_centred

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_c` | Scale term on the centred probe kernel. | log(\|\|Kc_t\|\|_F / \|\|Kc_0\|\|_F) | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `R_c` | Rotation term on the centred probe kernel. | 1 - <k_t, k_0> | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `A_t` | Centred kernel-target alignment on the probe. | <Kc_t, G> / (\|\|Kc_t\|\| \|\|G\|\|), G = H Y Y^T H | Cortes, Mohri and Rostamizadeh (2012), Definition 4; docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `A_u` | Uncentred alignment on the probe. | <K_t, Y Y^T> / (\|\|K_t\|\| \|\|Y Y^T\|\|) | ntk_lib.uncentred_alignment; ntk_lib.train_run history, copied at the checkpoint |
| `S_c_train` | The scale term on the train part of the probe, with the kernel restricted to those pairs. | - | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `R_c_train` | The rotation term on the train part of the probe, with the kernel restricted to those pairs. | - | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `A_t_train` | The centred alignment on the train part of the probe, with the kernel restricted to those pairs. | - | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `A_u_train` | The uncentred alignment on the train part of the probe, with the kernel restricted to those pairs. | - | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `S_c_test` | The scale term on the test part of the probe, with the kernel restricted to those pairs. | - | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `R_c_test` | The rotation term on the test part of the probe, with the kernel restricted to those pairs. | - | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `A_t_test` | The centred alignment on the test part of the probe, with the kernel restricted to those pairs. | - | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `A_u_test` | The uncentred alignment on the test part of the probe, with the kernel restricted to those pairs. | - | docs/report.tex, Kernel metrics section; ntk_lib.train_run history, copied at the checkpoint |
| `K_norm_centred` | Frobenius norm of the centred probe kernel. | \|\|Kc_t\|\|_F | Cortes, Mohri and Rostamizadeh (2012), Lemma 1, for H K H; ntk_trace.Tracer.measure |
| `inner_K0` | Inner product of the centred kernel with the centred kernel at step 0. | <Kc_t, Kc_0> | ntk_trace.Tracer.measure |
| `inner_G` | Inner product of the centred kernel with the centred target Gram matrix. | <Kc_t, G> | ntk_trace.Tracer.measure |
| `D` | The usual kernel movement statistic on the centred kernel. | \|\|Kc_t - Kc_0\|\|_F / \|\|Kc_0\|\|_F | docs/report.tex, Equation eq:decomp |
| `exp2S_term` | First piece of Equation eq:decomp. D^2 = exp2S_term + 1 - cross_term. | e^(2 S_c) | docs/report.tex, Equation eq:decomp, the group's own algebra |
| `cross_term` | Last piece of Equation eq:decomp. | 2 e^S_c (1 - R_c) | docs/report.tex, Equation eq:decomp, the group's own algebra |
| `gamma` | The aim: the cosine between the part of k_t orthogonal to k_0 and the part of the unit target orthogonal to k_0. NaN at step 0. | <v_t, g_perp> / \|\|v_t\|\|, v_t = k_t - <k_t, k_0> k_0 | term_dependence.py docstring, the group's own algebra; ntk_trace.Tracer.measure |
| `share_k0` | Share of the squared change of the unit kernel along k_0. It equals R_c / 2. | <d_t, k_0>^2 / \|\|d_t\|\|^2, d_t = k_t - k_0 | docs/zain_tierx_plan.md, E3; ntk_trace.Tracer.measure |
| `share_gperp` | Share of the squared change of the unit kernel along g_perp. | <d_t, g_perp>^2 / \|\|d_t\|\|^2 | docs/zain_tierx_plan.md, E3; ntk_trace.Tracer.measure |
| `share_residual` | Share of the squared change of the unit kernel orthogonal to both k_0 and g_perp. The three shares sum to one. | - | docs/zain_tierx_plan.md, E3; ntk_trace.Tracer.measure |
| `kernel_degenerate` | True if the centred probe kernel norm is below 1e-30. The unit-kernel columns are then NaN. | - | The group's own definition |

### layer_kernels

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_W1` | Scale term of the centred kernel term from the gradient of W1 alone. | log(\|\|Kc_W1,t\|\| / \|\|Kc_W1,0\|\|) | ntk_lib.entk_closed_form, The group's own definition |
| `R_W1` | Rotation term of the centred kernel term from W1 alone. | - | ntk_lib.entk_closed_form, The group's own definition |
| `A_W1` | Centred alignment of the kernel term from W1 alone. | - | ntk_lib.entk_closed_form, The group's own definition |
| `S_W2` | Scale term of the centred kernel term from the gradient of W2 alone. | log(\|\|Kc_W2,t\|\| / \|\|Kc_W2,0\|\|) | ntk_lib.entk_closed_form, The group's own definition |
| `R_W2` | Rotation term of the centred kernel term from W2 alone. | - | ntk_lib.entk_closed_form, The group's own definition |
| `A_W2` | Centred alignment of the kernel term from W2 alone. | - | ntk_lib.entk_closed_form, The group's own definition |
| `W1_kernel_share` | Share of the trace of the centred probe kernel that comes from W1. | tr(Kc_W1) / (tr(Kc_W1) + tr(Kc_W2)) | ntk_lib.entk_closed_form, The group's own definition |

### first_logit

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_first_logit` | The scale term of the centred kernel of the first output alone. The report names this kernel as a robustness check. | - | docs/report.tex, Kernel metrics section; ntk_lib.entk_closed_form |
| `R_first_logit` | The rotation term of the centred kernel of the first output alone. The report names this kernel as a robustness check. | - | docs/report.tex, Kernel metrics section; ntk_lib.entk_closed_form |
| `A_first_logit` | The centred alignment of the centred kernel of the first output alone. The report names this kernel as a robustness check. | - | docs/report.tex, Kernel metrics section; ntk_lib.entk_closed_form |

### spectrum

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `eig_trace` | Trace of the centred probe kernel. | - | The group's own definition |
| `eig_top1` | Largest eigenvalue of the centred probe kernel. | - | The group's own definition |
| `eig_top1_share` | Largest eigenvalue over the trace. | - | The group's own definition |
| `eig_eff_rank` | Exponential of the entropy of the eigenvalues normalised to sum to one. The zero eigenvalue that centring creates is left out, and so is every column below. | - | The group's own definition |
| `eig_participation` | Participation ratio of the eigenvalues. | (sum lambda)^2 / sum lambda^2 | The group's own definition |
| `eig_n90` | Number of the largest eigenvalues that hold 90 percent of the trace. | - | The group's own definition |
| `eig_gap_ab` | Ratio of eigenvalue 2(p - 1) to eigenvalue 2(p - 1) + 1, counting from one. In the tier x traces the functions of a alone and of b alone filled the top 2(p - 1) places. | - | docs/zain_tierx_plan.md, E4 |
| `eig_gap_top88` | Ratio of eigenvalue 4(p - 1) to eigenvalue 4(p - 1) + 1, the edge of the eigenvectors that the tracking columns follow. | - | docs/zain_tierx_plan.md, E4 |

### subspaces

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `probe_energy_sum` | Share of the trace of the centred probe kernel inside the centred functions of (a + b) mod p on the probe. The subspaces are not orthogonal on the probe, so the four shares need not sum to one. This is the label subspace. | tr(Q_s^T Kc Q_s) / tr(Kc) | docs/zain_tierx_plan.md, E3 |
| `probe_energy_diff` | Share of the trace of the centred probe kernel inside the centred functions of (a - b) mod p on the probe. The subspaces are not orthogonal on the probe, so the four shares need not sum to one. | tr(Q_s^T Kc Q_s) / tr(Kc) | docs/zain_tierx_plan.md, E3 |
| `probe_energy_a` | Share of the trace of the centred probe kernel inside the centred functions of a alone on the probe. The subspaces are not orthogonal on the probe, so the four shares need not sum to one. | tr(Q_s^T Kc Q_s) / tr(Kc) | docs/zain_tierx_plan.md, E3 |
| `probe_energy_b` | Share of the trace of the centred probe kernel inside the centred functions of b alone on the probe. The subspaces are not orthogonal on the probe, so the four shares need not sum to one. | tr(Q_s^T Kc Q_s) / tr(Kc) | docs/zain_tierx_plan.md, E3 |

### eigvec_tracking

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `label_capture_22` | Share of the centred label matrix inside the top 22 eigenvectors of the centred probe kernel. | \|\|V_22^T H Y\|\|_F^2 / \|\|H Y\|\|_F^2 | The group's own definition |
| `label_capture_44` | Share of the centred label matrix inside the top 44 eigenvectors of the centred probe kernel. | \|\|V_44^T H Y\|\|_F^2 / \|\|H Y\|\|_F^2 | The group's own definition |
| `label_capture_88` | Share of the centred label matrix inside the top 88 eigenvectors of the centred probe kernel. | \|\|V_88^T H Y\|\|_F^2 / \|\|H Y\|\|_F^2 | The group's own definition |
| `topk_drift` | Mean squared sine of the principal angles between the top 4(p - 1) eigenvectors and the same span at step 0. At large width the cut falls in a nearly degenerate bulk, so this is noisy there. | - | docs/zain_tierx_plan.md, E4; ntk_trace.Tracer.measure |
| `overlap_zero_median` | Median over the tracked top eigenvectors of the absolute overlap with the step 0 vector their chain of matches leads back to. Noisy at large width for the same reason. | - | docs/zain_tierx_plan.md, E4; ntk_trace.Tracer.measure |

### full_grid

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_full` | Scale term of the centred kernel on all p^2 pairs. | - | ntk_lib.entk_closed_form |
| `R_full` | Rotation term of the centred kernel on all p^2 pairs. | - | ntk_lib.entk_closed_form |
| `A_full` | Centred alignment on all p^2 pairs, with the labels of every pair. | - | ntk_lib.entk_closed_form |
| `full_energy_sum` | Share of the trace of the centred full-grid kernel inside the centred functions of (a + b) mod p. On the full grid the four subspaces are orthogonal. | tr(Q_s^T Kc Q_s) / tr(Kc) | The group's own definition |
| `full_energy_diff` | Share of the trace of the centred full-grid kernel inside the centred functions of (a - b) mod p. On the full grid the four subspaces are orthogonal. | tr(Q_s^T Kc Q_s) / tr(Kc) | The group's own definition |
| `full_energy_a` | Share of the trace of the centred full-grid kernel inside the centred functions of a alone. On the full grid the four subspaces are orthogonal. | tr(Q_s^T Kc Q_s) / tr(Kc) | The group's own definition |
| `full_energy_b` | Share of the trace of the centred full-grid kernel inside the centred functions of b alone. On the full grid the four subspaces are orthogonal. | tr(Q_s^T Kc Q_s) / tr(Kc) | The group's own definition |
| `full_energy_rest` | Share of the trace of the centred full-grid kernel outside the four subspaces. | 1 - sum of the four shares | The group's own definition |
| `krr_full_test_acc` | Accuracy of kernel ridge regression with the current kernel from the real training pairs to the real test pairs, starting from a zero function as the predictor does. The ridge is 0.001 times the mean diagonal of the training block. | argmax K_te,tr (K_tr,tr + r I)^-1 Y_tr | The group's own definition |
| `krr_full_test_mse` | Mean squared error of the same kernel ridge regression on the real test pairs. | - | The group's own definition |

### probe_krr

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `krr_probe_test_acc` | Accuracy of kernel ridge regression with the current kernel from the training part of the probe to its test part, starting from a zero function as the predictor does. The ridge is 0.001 times the mean diagonal of the training block. | argmax K_te,tr (K_tr,tr + r I)^-1 Y_tr | The group's own definition |
| `krr_probe_test_mse` | Mean squared error of the same kernel ridge regression on its test part. | - | The group's own definition |

### instant_rates

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `dS_step` | Change of the centred scale term over the next gradient-descent step, from a virtual step with the current gradient and decay, in float64. | S(K_{t+1}) - S(K_t) | The group's own definition |
| `dA_step` | Change of the centred alignment over the next step, from the same virtual step. | A(K_{t+1}) - A(K_t) | The group's own definition |
| `R_step` | Rotation of the centred unit kernel over the next step, from the same virtual step. | 1 - <k_{t+1}, k_t> | The group's own definition |

## runs

### run

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `run_id` | The name of the run, the grid cell name of ntk_lib.cell_name. | - | ntk_lib.cell_name |
| `alpha` | The laziness knob alpha. | - | Kumar et al. (2024), Appendix 8.1, Equation 7, arXiv v3 |
| `width` | The hidden width N. | - | - |
| `eta_kappa` | The weight decay eta lambda. | - | docs/report.tex, The laziness and weight-decay knobs |
| `seed` | The model seed. | - | - |
| `lr` | The learning rate eta_0 / alpha^2. | - | ntk_lib.train_run |
| `weight_decay` | The decay coefficient eta_kappa / lr passed to torch.optim.SGD. | - | ntk_lib.train_run |
| `status` | ok, diverged (the training loss stopped being finite) or failed (an exception). | - | - |
| `error` | The traceback of a failed run, empty otherwise. | - | - |
| `diverged_step` | The checkpoint at which the training loss was first not finite. | - | - |
| `steps` | The step budget. | - | - |
| `last_step` | The last recorded checkpoint. | - | - |
| `n_checkpoints` | The number of recorded checkpoints. | - | - |
| `seconds` | Wall-clock seconds for the run, logger included. | - | - |
| `callback_seconds` | Wall-clock seconds spent in the feature logger. | - | - |
| `feature_version` | dataset_features.FEATURE_VERSION when the run was made. | - | - |
| `git_commit` | The commit of the repository when the run was made. | - | - |
| `git_dirty` | True if repoduced-code had uncommitted changes when the run was made. | - | - |
| `closed_form_check` | Largest relative difference between the autograd and closed-form probe kernels at the first and last checkpoints. | - | ntk_lib.entk_closed_form |
| `init_weight_norm` | Weight norm at step 0. | - | ntk_lib.train_run |
| `final_weight_norm` | Weight norm at the last checkpoint. | - | - |
| `weight_norm_growth` | final_weight_norm / init_weight_norm. | - | - |
| `A_t_final` | A_t at the last checkpoint. | - | - |
| `A_full_final` | A_full at the last checkpoint. | - | - |
| `S_c_final` | S_c at the last checkpoint. | - | - |
| `R_c_final` | R_c at the last checkpoint. | - | - |
| `train_acc_final` | train_acc at the last checkpoint. | - | - |
| `test_acc_final` | test_acc at the last checkpoint. | - | - |
| `train_loss_final` | train_loss at the last checkpoint. | - | - |
| `test_loss_final` | test_loss at the last checkpoint. | - | - |
| `A_0` | A_t at step 0. | - | - |
| `A_full_0` | A_full at step 0. | - | - |
| `A_peak` | Largest A_t over the run. | - | - |
| `A_peak_step` | The step of A_peak. | - | - |
| `memorised` | True if the training accuracy reached 1 at some checkpoint. A run that never memorises has no grokking time, which is different from a censored one. | - | - |
| `t_mem` | First checkpoint at which the training accuracy is 1. | - | ntk_lib.accuracy_crossings |
| `t_mem_prev` | The checkpoint before t_mem, so the event lies in (t_mem_prev, t_mem]. | - | - |
| `t_gen_acc95` | First checkpoint at which the test accuracy reaches 0.95. | - | ntk_lib.accuracy_crossings |
| `t_gen_acc95_prev` | The checkpoint before t_gen_acc95. | - | - |
| `t_grok_acc95` | Grokking time under the accuracy definition at test level 0.95: t_gen_acc95 - t_mem, or last_step - t_mem if censored. Empty if the run never memorised. | - | ntk_lib.accuracy_crossings |
| `censored_acc95` | True if the test accuracy never reached 0.95, or the run never memorised. | - | ntk_lib.accuracy_crossings |
| `t_gen_acc99` | First checkpoint at which the test accuracy reaches 0.99. | - | ntk_lib.accuracy_crossings |
| `t_gen_acc99_prev` | The checkpoint before t_gen_acc99. | - | - |
| `t_grok_acc99` | Grokking time under the accuracy definition at test level 0.99: t_gen_acc99 - t_mem, or last_step - t_mem if censored. Empty if the run never memorised. | - | ntk_lib.accuracy_crossings |
| `censored_acc99` | True if the test accuracy never reached 0.99, or the run never memorised. | - | ntk_lib.accuracy_crossings |
| `t_gen_acc100` | First checkpoint at which the test accuracy reaches 1. | - | ntk_lib.accuracy_crossings |
| `t_gen_acc100_prev` | The checkpoint before t_gen_acc100. | - | - |
| `t_grok_acc100` | Grokking time under the accuracy definition at test level 1: t_gen_acc100 - t_mem, or last_step - t_mem if censored. Empty if the run never memorised. | - | ntk_lib.accuracy_crossings |
| `censored_acc100` | True if the test accuracy never reached 1, or the run never memorised. | - | ntk_lib.accuracy_crossings |
| `t_train_loss3e-2` | First checkpoint at which the training loss is below 0.03. | - | ntk_lib.grokking_time |
| `t_test_loss3e-2` | First checkpoint at which the test loss is below 0.03. | - | ntk_lib.grokking_time |
| `t_grok_loss3e-2` | Grokking time under the pre-registered loss definition with both thresholds 0.03. | - | ntk_lib.grokking_time; docs/report.tex, Defining t_grok |
| `censored_loss3e-2` | True if the test loss never fell below 0.03, or the training loss never did. | - | ntk_lib.grokking_time |
| `t_train_loss2e-2` | First checkpoint at which the training loss is below 0.02. | - | ntk_lib.grokking_time |
| `t_test_loss2e-2` | First checkpoint at which the test loss is below 0.02. | - | ntk_lib.grokking_time |
| `t_grok_loss2e-2` | Grokking time under the pre-registered loss definition with both thresholds 0.02. | - | ntk_lib.grokking_time; docs/report.tex, Defining t_grok |
| `censored_loss2e-2` | True if the test loss never fell below 0.02, or the training loss never did. | - | ntk_lib.grokking_time |
| `t_train_loss1e-2` | First checkpoint at which the training loss is below 0.01. | - | ntk_lib.grokking_time |
| `t_test_loss1e-2` | First checkpoint at which the test loss is below 0.01. | - | ntk_lib.grokking_time |
| `t_grok_loss1e-2` | Grokking time under the pre-registered loss definition with both thresholds 0.01. | - | ntk_lib.grokking_time; docs/report.tex, Defining t_grok |
| `censored_loss1e-2` | True if the test loss never fell below 0.01, or the training loss never did. | - | ntk_lib.grokking_time |
| `t_train_loss3e-3` | First checkpoint at which the training loss is below 0.003. | - | ntk_lib.grokking_time |
| `t_test_loss3e-3` | First checkpoint at which the test loss is below 0.003. | - | ntk_lib.grokking_time |
| `t_grok_loss3e-3` | Grokking time under the pre-registered loss definition with both thresholds 0.003. | - | ntk_lib.grokking_time; docs/report.tex, Defining t_grok |
| `censored_loss3e-3` | True if the test loss never fell below 0.003, or the training loss never did. | - | ntk_lib.grokking_time |
| `t_train_loss1e-3` | First checkpoint at which the training loss is below 0.001. | - | ntk_lib.grokking_time |
| `t_test_loss1e-3` | First checkpoint at which the test loss is below 0.001. | - | ntk_lib.grokking_time |
| `t_grok_loss1e-3` | Grokking time under the pre-registered loss definition with both thresholds 0.001. | - | ntk_lib.grokking_time; docs/report.tex, Defining t_grok |
| `censored_loss1e-3` | True if the test loss never fell below 0.001, or the training loss never did. | - | ntk_lib.grokking_time |
| `A_t_at_mem` | A_t at t_mem. Empty if the event did not happen. | - | - |
| `S_c_at_mem` | S_c at t_mem. Empty if the event did not happen. | - | - |
| `R_c_at_mem` | R_c at t_mem. Empty if the event did not happen. | - | - |
| `gamma_at_mem` | gamma at t_mem. Empty if the event did not happen. | - | - |
| `D_at_mem` | D at t_mem. Empty if the event did not happen. | - | - |
| `weight_norm_at_mem` | weight_norm at t_mem. Empty if the event did not happen. | - | - |
| `A_full_at_mem` | A_full at t_mem. Empty if the event did not happen. | - | - |
| `S_W1_at_mem` | S_W1 at t_mem. Empty if the event did not happen. | - | - |
| `S_W2_at_mem` | S_W2 at t_mem. Empty if the event did not happen. | - | - |
| `A_t_at_gen` | A_t at t_gen_acc100. Empty if the event did not happen. | - | - |
| `S_c_at_gen` | S_c at t_gen_acc100. Empty if the event did not happen. | - | - |
| `R_c_at_gen` | R_c at t_gen_acc100. Empty if the event did not happen. | - | - |
| `gamma_at_gen` | gamma at t_gen_acc100. Empty if the event did not happen. | - | - |
| `D_at_gen` | D at t_gen_acc100. Empty if the event did not happen. | - | - |
| `weight_norm_at_gen` | weight_norm at t_gen_acc100. Empty if the event did not happen. | - | - |
| `A_full_at_gen` | A_full at t_gen_acc100. Empty if the event did not happen. | - | - |
| `S_W1_at_gen` | S_W1 at t_gen_acc100. Empty if the event did not happen. | - | - |
| `S_W2_at_gen` | S_W2 at t_gen_acc100. Empty if the event did not happen. | - | - |
| `t_A08` | First checkpoint at which A_t reaches 0.08. Empty if it never does. | - | - |
| `t_A10` | First checkpoint at which A_t reaches 0.1. Empty if it never does. | - | - |
| `t_A12` | First checkpoint at which A_t reaches 0.12. Empty if it never does. | - | - |
| `t_A14` | First checkpoint at which A_t reaches 0.14. Empty if it never does. | - | - |
