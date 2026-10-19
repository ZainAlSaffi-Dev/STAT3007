# Data dictionary of the MNIST sweep

This file is written by `mnist_sweep.py assemble` from `mnist_features.COLUMNS` and `mnist_sweep.RUN_COLUMNS`. Edit those, not this file.

`checkpoints.parquet` has one row per run and checkpoint. `runs.parquet` and `runs.csv` have one row per run. `runs.parquet` and `checkpoints.parquet` join on `run_id`. The weights are in `runs/<run_id>_weights.npz`, with arrays `steps`, `W1` of shape (T, N, 784), `W2` of shape (T, N, N) and `W3` of shape (T, 10, N). Seed 0 saves about 8 checkpoints and the other seeds the last one. `mnist_features.kernel_terms` rebuilds any kernel from them.

Notation. H = I - (1/n) 1 1^T. Kc = H K H is the centred kernel and k = Kc / ||Kc||_F the unit centred kernel. Y is the one-hot label matrix and G = H Y Y^T H. <A, B> is the Frobenius inner product. The probe is the mixed probe of 128 training and 128 test images.

## checkpoints

### ids

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `run_id` | The name of the run, mnist_sweep.run_id. | - | mnist_sweep.run_id |
| `init_scale` | The initial scale s. The initial weights are multiplied by s. For this 3-homogeneous network the initial output is s^3 and the initial kernel s^4 times its value at s = 1. | - | Liu, Michaud and Tegmark (2023), Omnigrok, arXiv:2210.01117, Section 4 |
| `width` | The width of both hidden layers. | - | - |
| `eta_kappa` | The weight decay per step, eta lambda in the report. It is unchanged when the learning rate changes. | - | docs/report.tex, The laziness and weight-decay knobs |
| `seed` | The model seed. | - | - |
| `data_seed` | The seed of the draw of the training images, 42 + seed. | - | - |

### grid

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `step` | The checkpoint step. | - | - |
| `log10_step1` | Base 10 logarithm of the step plus one. | log10(step + 1) | - |
| `dt_prev` | Steps since the previous checkpoint. NaN at step 0. | - | - |
| `on_linear_grid` | True if the step is a multiple of the linear checkpoint interval. | - | - |
| `on_log_grid` | True if the step is one of the log-spaced checkpoints. | - | - |
| `wall_seconds` | Wall-clock seconds since the run started. It depends on the machine and the load. | - | - |

### performance

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `train_loss` | Mean squared error of the network on the train set. | - | ntk_lib.train_run history, copied at the checkpoint |
| `train_acc` | Accuracy of the network on the train set, by the largest output. | - | ntk_lib.train_run history, copied at the checkpoint |
| `test_loss` | Mean squared error of the network on the test set. | - | ntk_lib.train_run history, copied at the checkpoint |
| `test_acc` | Accuracy of the network on the test set, by the largest output. | - | ntk_lib.train_run history, copied at the checkpoint |
| `train_margin_mean` | Mean over the train set of the correct output minus the largest other output. | mean_i (f_i[y_i] - max_{c != y_i} f_i[c]) | repoduced-code/dataset_features.py, the same column for modular addition |
| `train_margin_min` | Smallest margin over the train set. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `train_correct_out_mean` | Mean output at the correct class on the train set. The MSE target there is one. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `train_wrong_out_rms` | Root mean square output at the wrong classes on the train set. The MSE target there is zero. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `train_out_rms` | Root mean square of all outputs on the train set, the output scale. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `train_out_max_mean` | Mean over the train set of the largest output. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `test_margin_mean` | Mean over the test set of the correct output minus the largest other output. | mean_i (f_i[y_i] - max_{c != y_i} f_i[c]) | repoduced-code/dataset_features.py, the same column for modular addition |
| `test_margin_min` | Smallest margin over the test set. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `test_correct_out_mean` | Mean output at the correct class on the test set. The MSE target there is one. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `test_wrong_out_rms` | Root mean square output at the wrong classes on the test set. The MSE target there is zero. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `test_out_rms` | Root mean square of all outputs on the test set, the output scale. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `test_out_max_mean` | Mean over the test set of the largest output. | - | repoduced-code/dataset_features.py, the same column for modular addition |

### learning_rate

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `lr` | The learning rate in force for the steps after the checkpoint. The curvature rule sets it to eta_0 lam_0 / lam_t every lr_update_interval steps. | - | ntk_lib.train_run history, copied at the checkpoint (eta) |
| `lr_factor` | lr over its value at step 0. | lr / eta_0 | The group's own definition |
| `lam` | The latest top Gauss-Newton eigenvalue of the training loss, from the last update of the learning rate. | lambda_max((2 / (n C)) J^T J) | ntk_lib.train_run history, copied at the checkpoint |

### optimisation

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `grad_norm` | Norm of the gradient of the full training loss, without the decay term. | \|\|dL/dtheta\|\| | repoduced-code/dataset_features.py, the same column for modular addition |
| `grad_norm_W1` | Norm of the gradient of the full training loss with respect to W1. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `grad_norm_W2` | Norm of the gradient of the full training loss with respect to W2. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `grad_norm_W3` | Norm of the gradient of the full training loss with respect to W3. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `update_norm` | Norm of a full-batch gradient step with decay at the current learning rate. Training takes minibatch steps, so this is the size of the mean step. | lr \|\|g + wd theta\|\|, wd = eta_kappa / lr | repoduced-code/dataset_features.py, the same column for modular addition |
| `grad_weight_cos` | Cosine between the loss gradient and the weights. Near minus one at a point where decay and gradient cancel. | <g, theta> / (\|\|g\|\| \|\|theta\|\|) | repoduced-code/dataset_features.py, the same column for modular addition |
| `grad_decay_ratio` | Size of the loss gradient over the size of the decay pull. NaN without decay. | \|\|g\|\| / (wd \|\|theta\|\|) | repoduced-code/dataset_features.py, the same column for modular addition |
| `decay_residual` | How far the gradient and the decay pull are from cancelling. Zero at a fixed point of gradient descent with decay, one when the two are orthogonal. | \|\|g + wd theta\|\| / (\|\|g\|\| + wd \|\|theta\|\|) | repoduced-code/dataset_features.py, the same column for modular addition |
| `stationary` | True if the run has decay and decay_residual is below 0.001. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `sharpness` | Largest eigenvalue of the Hessian of the full training loss, without the decay term, by 20 power iterations with Hessian-vector products, warm started. | lambda_max(d^2 L / d theta^2) | repoduced-code/dataset_features.py, the same column for modular addition |
| `lr_sharpness` | Learning rate times sharpness. Gradient descent on a quadratic with this curvature is stable when it is below 2. | lr lambda_max | repoduced-code/dataset_features.py, the same column for modular addition |

### weights

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `weight_norm` | Norm of all the parameters. | - | ntk_lib.train_run history, copied at the checkpoint |
| `param_dist` | Relative parameter movement. | \|\|theta_t - theta_0\|\| / \|\|theta_0\|\| | ntk_lib.train_run history, copied at the checkpoint |
| `log_weight_norm_ratio` | Log of the weight norm over its value at step 0. The kernel of this 3-homogeneous network under a uniform rescaling of the weights by c is c^4 times larger, so four times this value is the scale term such a rescaling would give. | log(\|\|theta_t\|\| / \|\|theta_0\|\|) | The group's own definition |
| `W1_norm` | Frobenius norm of W1. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `W1_dist` | Relative movement of W1. | \|\|W1_t - W1_0\|\| / \|\|W1_0\|\| | repoduced-code/dataset_features.py, the same column for modular addition |
| `W1_stable_rank` | Stable rank of W1. | \|\|W\|\|_F^2 / sigma_max^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `W1_eff_rank` | Effective rank of W1, the exponential of the entropy of the singular values normalised to sum to one. | exp(-sum q log q), q = sigma / sum sigma | repoduced-code/dataset_features.py, the same column for modular addition |
| `W2_norm` | Frobenius norm of W2. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `W2_dist` | Relative movement of W2. | \|\|W2_t - W2_0\|\| / \|\|W2_0\|\| | repoduced-code/dataset_features.py, the same column for modular addition |
| `W2_stable_rank` | Stable rank of W2. | \|\|W\|\|_F^2 / sigma_max^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `W2_eff_rank` | Effective rank of W2, the exponential of the entropy of the singular values normalised to sum to one. | exp(-sum q log q), q = sigma / sum sigma | repoduced-code/dataset_features.py, the same column for modular addition |
| `W3_norm` | Frobenius norm of W3. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `W3_dist` | Relative movement of W3. | \|\|W3_t - W3_0\|\| / \|\|W3_0\|\| | repoduced-code/dataset_features.py, the same column for modular addition |
| `W3_stable_rank` | Stable rank of W3. | \|\|W\|\|_F^2 / sigma_max^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `W3_eff_rank` | Effective rank of W3, the exponential of the entropy of the singular values normalised to sum to one. | exp(-sum q log q), q = sigma / sum sigma | repoduced-code/dataset_features.py, the same column for modular addition |
| `layer_imbalance_12` | Difference of the squared norms of W1 and W2 over the squared total norm. Gradient flow without decay keeps \|\|W_l\|\|^2 - \|\|W_(l+1)\|\|^2 fixed for a bias-free ReLU network. With decay it shrinks by (1 - eta_kappa)^2 per step. | (\|\|W1\|\|^2 - \|\|W2\|\|^2) / \|\|theta\|\|^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `layer_imbalance_23` | Difference of the squared norms of W2 and W3 over the squared total norm. Gradient flow without decay keeps \|\|W_l\|\|^2 - \|\|W_(l+1)\|\|^2 fixed for a bias-free ReLU network. With decay it shrinks by (1 - eta_kappa)^2 per step. | (\|\|W2\|\|^2 - \|\|W3\|\|^2) / \|\|theta\|\|^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `dead_frac_h1` | Fraction of the units of hidden layer 1 that are inactive on every training image. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `active_frac_h1` | Mean over training images of the fraction of the units of hidden layer 1 that are active. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `dead_frac_h2` | Fraction of the units of hidden layer 2 that are inactive on every training image. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `active_frac_h2` | Mean over training images of the fraction of the units of hidden layer 2 that are active. | - | repoduced-code/dataset_features.py, the same column for modular addition |

### kernel_raw

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_t` | Scale term on the raw probe kernel. | log(\|\|K_t\|\|_F / \|\|K_0\|\|_F) | ntk_lib.train_run history, copied at the checkpoint |
| `R_t` | Rotation term on the raw probe kernel. | 1 - <K_t / \|\|K_t\|\|, K_0 / \|\|K_0\|\|> | ntk_lib.train_run history, copied at the checkpoint |
| `K_norm` | Frobenius norm of the raw probe kernel, the sum-over-outputs kernel. | - | ntk_lib.train_run history, copied at the checkpoint |
| `yKy` | y^T K^+ y on the probe set, summed over the label columns. | - | ntk_lib.train_run history, copied at the checkpoint |

### kernel_centred

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_c` | Scale term on the centred probe kernel. | log(\|\|Kc_t\|\|_F / \|\|Kc_0\|\|_F) | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `R_c` | Rotation term on the centred probe kernel. | 1 - <k_t, k_0> | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `A_t` | Centred kernel-target alignment on the probe. | <Kc_t, G> / (\|\|Kc_t\|\| \|\|G\|\|), G = H Y Y^T H | Cortes, Mohri and Rostamizadeh (2012), Definition 4. docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `A_u` | Uncentred alignment on the probe. | <K_t, Y Y^T> / (\|\|K_t\|\| \|\|Y Y^T\|\|) | ntk_lib.uncentred_alignment. ntk_lib.train_run history, copied at the checkpoint |
| `S_c_train` | The scale term on the train part of the probe, with the kernel restricted to those images. | - | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `R_c_train` | The rotation term on the train part of the probe, with the kernel restricted to those images. | - | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `A_t_train` | The centred alignment on the train part of the probe, with the kernel restricted to those images. | - | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `A_u_train` | The uncentred alignment on the train part of the probe, with the kernel restricted to those images. | - | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `S_c_test` | The scale term on the test part of the probe, with the kernel restricted to those images. | - | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `R_c_test` | The rotation term on the test part of the probe, with the kernel restricted to those images. | - | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `A_t_test` | The centred alignment on the test part of the probe, with the kernel restricted to those images. | - | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `A_u_test` | The uncentred alignment on the test part of the probe, with the kernel restricted to those images. | - | docs/report.tex, Kernel metrics section. ntk_lib.train_run history, copied at the checkpoint |
| `K_norm_centred` | Frobenius norm of the centred probe kernel. | \|\|Kc_t\|\|_F | Cortes, Mohri and Rostamizadeh (2012), Lemma 1, for H K H |
| `inner_K0` | Inner product of the centred kernel with the centred kernel at step 0. | <Kc_t, Kc_0> | repoduced-code/dataset_features.py, the same column for modular addition |
| `inner_G` | Inner product of the centred kernel with the centred target Gram matrix. | <Kc_t, G> | repoduced-code/dataset_features.py, the same column for modular addition |
| `D` | The usual kernel movement statistic on the centred kernel. | \|\|Kc_t - Kc_0\|\|_F / \|\|Kc_0\|\|_F | docs/report.tex, Equation eq:decomp |
| `exp2S_term` | First piece of Equation eq:decomp. D^2 = exp2S_term + 1 - cross_term. | e^(2 S_c) | docs/report.tex, Equation eq:decomp, the group's own algebra |
| `cross_term` | Last piece of Equation eq:decomp. | 2 e^S_c (1 - R_c) | docs/report.tex, Equation eq:decomp, the group's own algebra |
| `gamma` | The aim, the cosine between the part of k_t orthogonal to k_0 and the part of the unit target orthogonal to k_0. NaN at step 0. | <v_t, g_perp> / \|\|v_t\|\|, v_t = k_t - <k_t, k_0> k_0 | repoduced-code/dataset_features.py, the same column for modular addition |
| `share_k0` | Share of the squared change of the unit kernel along k_0. It equals R_c / 2. | <d_t, k_0>^2 / \|\|d_t\|\|^2, d_t = k_t - k_0 | repoduced-code/dataset_features.py, the same column for modular addition |
| `share_gperp` | Share of the squared change of the unit kernel along g_perp. | <d_t, g_perp>^2 / \|\|d_t\|\|^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `share_residual` | Share of the squared change of the unit kernel orthogonal to both k_0 and g_perp. share_k0, share_gperp and share_residual sum to one. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `kernel_degenerate` | True if the centred probe kernel norm is below 1e-30. The unit-kernel columns are then NaN. | - | repoduced-code/dataset_features.py, the same column for modular addition |

### layer_kernels

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_W1` | Scale term of the centred kernel term from the gradient of W1 alone. | log(\|\|Kc_W1,t\|\| / \|\|Kc_W1,0\|\|) | kernel_terms. The group's own definition |
| `R_W1` | Rotation term of the centred kernel term from W1 alone. | - | kernel_terms. The group's own definition |
| `A_W1` | Centred alignment of the kernel term from W1 alone. | - | kernel_terms. The group's own definition |
| `W1_kernel_share` | Share of the trace of the centred probe kernel that comes from W1. | tr(Kc_W1) / tr(Kc) | kernel_terms. The group's own definition |
| `S_W2` | Scale term of the centred kernel term from the gradient of W2 alone. | log(\|\|Kc_W2,t\|\| / \|\|Kc_W2,0\|\|) | kernel_terms. The group's own definition |
| `R_W2` | Rotation term of the centred kernel term from W2 alone. | - | kernel_terms. The group's own definition |
| `A_W2` | Centred alignment of the kernel term from W2 alone. | - | kernel_terms. The group's own definition |
| `W2_kernel_share` | Share of the trace of the centred probe kernel that comes from W2. | tr(Kc_W2) / tr(Kc) | kernel_terms. The group's own definition |
| `S_W3` | Scale term of the centred kernel term from the gradient of W3 alone. | log(\|\|Kc_W3,t\|\| / \|\|Kc_W3,0\|\|) | kernel_terms. The group's own definition |
| `R_W3` | Rotation term of the centred kernel term from W3 alone. | - | kernel_terms. The group's own definition |
| `A_W3` | Centred alignment of the kernel term from W3 alone. | - | kernel_terms. The group's own definition |
| `W3_kernel_share` | Share of the trace of the centred probe kernel that comes from W3. | tr(Kc_W3) / tr(Kc) | kernel_terms. The group's own definition |

### first_logit

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_first_logit` | The scale term of the centred kernel of the first output alone. The report uses this kernel as a robustness check. | - | docs/report.tex, Kernel metrics section. kernel_terms |
| `R_first_logit` | The rotation term of the centred kernel of the first output alone. The report uses this kernel as a robustness check. | - | docs/report.tex, Kernel metrics section. kernel_terms |
| `A_first_logit` | The centred alignment of the centred kernel of the first output alone. The report uses this kernel as a robustness check. | - | docs/report.tex, Kernel metrics section. kernel_terms |

### report_kernel

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `S_sum` | Scale, the plain ratio of Frobenius norms to step 0, of the kernel of the sum of the logits divided by sqrt(C), on a fixed draw of 1000 test images. These are the S_t, R_t, D_t and A_t of the golden report. | - | docs/report_golden.pdf, Equations 5 and 6. kernel_terms |
| `R_sum` | Shape change, one minus the uncentred cosine to step 0, of the kernel of the sum of the logits divided by sqrt(C), on a fixed draw of 1000 test images. These are the S_t, R_t, D_t and A_t of the golden report. | - | docs/report_golden.pdf, Equations 5 and 6. kernel_terms |
| `D_sum` | Variation, the Frobenius norm of the change from step 0 over the norm at step 0, of the kernel of the sum of the logits divided by sqrt(C), on a fixed draw of 1000 test images. These are the S_t, R_t, D_t and A_t of the golden report. | - | docs/report_golden.pdf, Equations 5 and 6. kernel_terms |
| `A_sum` | Centred alignment with Y Y^T, of the kernel of the sum of the logits divided by sqrt(C), on a fixed draw of 1000 test images. These are the S_t, R_t, D_t and A_t of the golden report. | - | docs/report_golden.pdf, Equations 5 and 6. kernel_terms |

### spectrum

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `eig_trace` | Trace of the centred probe kernel. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `eig_top1` | Largest eigenvalue of the centred probe kernel. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `eig_top1_share` | Largest eigenvalue over the trace. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `eig_eff_rank` | Exponential of the entropy of the eigenvalues normalised to sum to one. The zero eigenvalue that centring creates is left out, and so is every column below. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `eig_participation` | Participation ratio of the eigenvalues. | (sum lambda)^2 / sum lambda^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `eig_n90` | Number of the largest eigenvalues that hold 90 percent of the trace. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `eig_gap_label` | Ratio of eigenvalue 9 to eigenvalue 10, counting from one. The centred labels span 9 dimensions. | - | The group's own definition |
| `eig_gap_top36` | Ratio of eigenvalue 36 to eigenvalue 37, the edge of the eigenvectors that the tracking columns follow. | - | The group's own definition |

### subspaces

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `probe_energy_label` | Share of the trace of the centred probe kernel inside the centred label span on the probe. | tr(Q^T Kc Q) / tr(Kc), Q an orthonormal basis of H Y | repoduced-code/dataset_features.py, the same column for modular addition |

### eigvec_tracking

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `label_capture_9` | Share of the centred label matrix inside the top 9 eigenvectors of the centred probe kernel. | \|\|V_9^T H Y\|\|_F^2 / \|\|H Y\|\|_F^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `label_capture_18` | Share of the centred label matrix inside the top 18 eigenvectors of the centred probe kernel. | \|\|V_18^T H Y\|\|_F^2 / \|\|H Y\|\|_F^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `label_capture_36` | Share of the centred label matrix inside the top 36 eigenvectors of the centred probe kernel. | \|\|V_36^T H Y\|\|_F^2 / \|\|H Y\|\|_F^2 | repoduced-code/dataset_features.py, the same column for modular addition |
| `topk_drift` | Mean squared sine of the principal angles between the span of the top 36 eigenvectors and the same span at step 0. | - | repoduced-code/dataset_features.py, the same column for modular addition |
| `overlap_zero_median` | Median over the top 36 eigenvectors of the largest absolute overlap with any of the top 36 eigenvectors at step 0. dataset_features.py follows a chain of matches back to step 0 instead. | - | The group's own definition |

### probe_krr

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `krr_probe_test_acc` | Accuracy of kernel ridge regression with the current probe kernel from the training part of the probe to its test part. The ridge is 0.001 times the mean diagonal of the training block. | argmax K_te,tr (K_tr,tr + r I)^-1 Y_tr | repoduced-code/dataset_features.py, the same column for modular addition |
| `krr_probe_test_mse` | Mean squared error of the same kernel ridge regression on the test part. | - | repoduced-code/dataset_features.py, the same column for modular addition |

### instant_rates

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `dS_step` | Change of the centred scale term over one virtual full-batch gradient step with decay at the current learning rate, in float64. | S(K_{t+1}) - S(K_t) | repoduced-code/dataset_features.py, the same column for modular addition |
| `dA_step` | Change of the centred alignment over the same virtual step. | A(K_{t+1}) - A(K_t) | repoduced-code/dataset_features.py, the same column for modular addition |
| `R_step` | Rotation of the centred unit kernel over the same virtual step. | 1 - <k_{t+1}, k_t> | repoduced-code/dataset_features.py, the same column for modular addition |

## runs

### run

| name | meaning | formula | source |
| --- | --- | --- | --- |
| `run_id` | The name of the run, run_id. | - | - |
| `init_scale` | The initial scale s. | - | Liu, Michaud and Tegmark (2023), Omnigrok, arXiv:2210.01117, Section 4 |
| `width` | The width of both hidden layers. | - | - |
| `eta_kappa` | The weight decay per step, eta lambda. | - | docs/report.tex, The laziness and weight-decay knobs |
| `seed` | The model seed. | - | - |
| `data_seed` | The seed of the draw of the 1000 training images, 42 + seed. | - | - |
| `n_train` | The number of training images. | - | - |
| `batch_size` | The minibatch size. | - | - |
| `lam_0` | The top Gauss-Newton eigenvalue of the loss at the initial weights. | - | ntk_lib.gauss_newton_top_eigenvalue |
| `eta_0` | The initial learning rate, 1 / lam_0. | - | - |
| `status` | ok, diverged (the training loss stopped being finite) or failed (an exception). | - | - |
| `error` | The traceback of a failed run, empty otherwise. | - | - |
| `diverged_step` | The checkpoint at which the training loss was first not finite. | - | - |
| `steps` | The step budget. | - | - |
| `last_step` | The last recorded checkpoint. | - | - |
| `n_checkpoints` | The number of recorded checkpoints. | - | - |
| `seconds` | Wall-clock seconds for the run, logger included. | - | - |
| `callback_seconds` | Wall-clock seconds spent in the feature logger. | - | - |
| `feature_version` | mnist_features.FEATURE_VERSION when the run was made. | - | - |
| `git_commit` | The commit of the repository when the run was made. | - | - |
| `git_dirty` | True if mnist/ or ntk_lib.py had uncommitted changes when the run was made. | - | - |
| `closed_form_check` | Largest relative difference between the autograd and closed-form kernels on 32 probe images at the first and last checkpoints. | - | ntk_lib.entk_closed_form_deep |
| `init_weight_norm` | Weight norm at step 0. | - | ntk_lib.train_run |
| `final_weight_norm` | Weight norm at the last checkpoint. | - | - |
| `weight_norm_growth` | final_weight_norm / init_weight_norm. | - | - |
| `final_lr_factor` | The learning rate at the last checkpoint over eta_0. | - | - |
| `A_t_final` | A_t at the last checkpoint. | - | - |
| `S_c_final` | S_c at the last checkpoint. | - | - |
| `R_c_final` | R_c at the last checkpoint. | - | - |
| `S_sum_final` | S_sum at the last checkpoint. | - | - |
| `R_sum_final` | R_sum at the last checkpoint. | - | - |
| `D_sum_final` | D_sum at the last checkpoint. | - | - |
| `A_sum_final` | A_sum at the last checkpoint. | - | - |
| `train_acc_final` | train_acc at the last checkpoint. | - | - |
| `test_acc_final` | test_acc at the last checkpoint. | - | - |
| `train_loss_final` | train_loss at the last checkpoint. | - | - |
| `test_loss_final` | test_loss at the last checkpoint. | - | - |
| `A_0` | A_t at step 0. | - | - |
| `A_peak` | Largest A_t over the run. | - | - |
| `A_peak_step` | The step of A_peak. | - | - |
| `A_min` | Smallest A_t over the run. | - | - |
| `A_min_step` | The step of A_min. | - | - |
| `best_test_acc` | Largest test accuracy over the run. | - | - |
| `best_test_step` | The step of best_test_acc. | - | - |
| `memorised` | True if the training accuracy reached 0.99 at some checkpoint. A run that never memorises has no grokking time, which is different from a censored one. | - | docs/report_golden.pdf, section 3.5 |
| `t_mem` | First checkpoint at which the training accuracy is at least 0.99. | - | docs/report_golden.pdf, section 3.5 |
| `t_mem_prev` | The checkpoint before t_mem. The event lies in (t_mem_prev, t_mem]. | - | - |
| `t_mem100` | First checkpoint at which the training accuracy is 1, the memorisation of dataset_sweep.py. | - | ntk_lib.accuracy_crossings |
| `test_acc_at_mem` | Test accuracy at t_mem. | - | - |
| `t_gen_acc50` | First checkpoint at which the test accuracy reaches 0.5, counted from step 0. | - | - |
| `t_gen_acc50_prev` | The checkpoint before t_gen_acc50. | - | - |
| `t_grok_acc50` | Grokking time at test level 0.5: t_gen_acc50 - t_mem, or last_step - t_mem if censored. Empty if the run never memorised. Negative if the level came before memorisation. | - | dataset_sweep.py, the same column |
| `censored_acc50` | True if the test accuracy never reached 0.5, or the run never memorised. | - | - |
| `t_gen_acc60` | First checkpoint at which the test accuracy reaches 0.6, counted from step 0. | - | - |
| `t_gen_acc60_prev` | The checkpoint before t_gen_acc60. | - | - |
| `t_grok_acc60` | Grokking time at test level 0.6: t_gen_acc60 - t_mem, or last_step - t_mem if censored. Empty if the run never memorised. Negative if the level came before memorisation. | - | dataset_sweep.py, the same column |
| `censored_acc60` | True if the test accuracy never reached 0.6, or the run never memorised. | - | - |
| `t_gen_acc70` | First checkpoint at which the test accuracy reaches 0.7, counted from step 0. | - | - |
| `t_gen_acc70_prev` | The checkpoint before t_gen_acc70. | - | - |
| `t_grok_acc70` | Grokking time at test level 0.7: t_gen_acc70 - t_mem, or last_step - t_mem if censored. Empty if the run never memorised. Negative if the level came before memorisation. | - | dataset_sweep.py, the same column |
| `censored_acc70` | True if the test accuracy never reached 0.7, or the run never memorised. | - | - |
| `t_gen_acc80` | First checkpoint at which the test accuracy reaches 0.8, counted from step 0. | - | - |
| `t_gen_acc80_prev` | The checkpoint before t_gen_acc80. | - | - |
| `t_grok_acc80` | Grokking time at test level 0.8: t_gen_acc80 - t_mem, or last_step - t_mem if censored. Empty if the run never memorised. Negative if the level came before memorisation. | - | dataset_sweep.py, the same column |
| `censored_acc80` | True if the test accuracy never reached 0.8, or the run never memorised. | - | - |
| `t_gen_acc85` | First checkpoint at which the test accuracy reaches 0.85, counted from step 0. | - | - |
| `t_gen_acc85_prev` | The checkpoint before t_gen_acc85. | - | - |
| `t_grok_acc85` | Grokking time at test level 0.85: t_gen_acc85 - t_mem, or last_step - t_mem if censored. Empty if the run never memorised. Negative if the level came before memorisation. | - | dataset_sweep.py, the same column |
| `censored_acc85` | True if the test accuracy never reached 0.85, or the run never memorised. | - | - |
| `t_gen_acc90` | First checkpoint at which the test accuracy reaches 0.9, counted from step 0. | - | - |
| `t_gen_acc90_prev` | The checkpoint before t_gen_acc90. | - | - |
| `t_grok_acc90` | Grokking time at test level 0.9: t_gen_acc90 - t_mem, or last_step - t_mem if censored. Empty if the run never memorised. Negative if the level came before memorisation. | - | dataset_sweep.py, the same column |
| `censored_acc90` | True if the test accuracy never reached 0.9, or the run never memorised. | - | - |
| `t_train_loss8e-2` | First checkpoint at which the training loss is below 0.08. | - | ntk_lib.grokking_time |
| `t_test_loss8e-2` | First checkpoint at which the test loss is below 0.08. | - | ntk_lib.grokking_time |
| `t_grok_loss8e-2` | Grokking time under the loss definition with both thresholds 0.08. | - | ntk_lib.grokking_time. docs/report.tex, Defining t_grok |
| `censored_loss8e-2` | True if the test loss never fell below 0.08, or the training loss never did. | - | ntk_lib.grokking_time |
| `t_train_loss6e-2` | First checkpoint at which the training loss is below 0.06. | - | ntk_lib.grokking_time |
| `t_test_loss6e-2` | First checkpoint at which the test loss is below 0.06. | - | ntk_lib.grokking_time |
| `t_grok_loss6e-2` | Grokking time under the loss definition with both thresholds 0.06. | - | ntk_lib.grokking_time. docs/report.tex, Defining t_grok |
| `censored_loss6e-2` | True if the test loss never fell below 0.06, or the training loss never did. | - | ntk_lib.grokking_time |
| `t_train_loss4e-2` | First checkpoint at which the training loss is below 0.04. | - | ntk_lib.grokking_time |
| `t_test_loss4e-2` | First checkpoint at which the test loss is below 0.04. | - | ntk_lib.grokking_time |
| `t_grok_loss4e-2` | Grokking time under the loss definition with both thresholds 0.04. | - | ntk_lib.grokking_time. docs/report.tex, Defining t_grok |
| `censored_loss4e-2` | True if the test loss never fell below 0.04, or the training loss never did. | - | ntk_lib.grokking_time |
| `t_train_loss3e-2` | First checkpoint at which the training loss is below 0.03. | - | ntk_lib.grokking_time |
| `t_test_loss3e-2` | First checkpoint at which the test loss is below 0.03. | - | ntk_lib.grokking_time |
| `t_grok_loss3e-2` | Grokking time under the loss definition with both thresholds 0.03. | - | ntk_lib.grokking_time. docs/report.tex, Defining t_grok |
| `censored_loss3e-2` | True if the test loss never fell below 0.03, or the training loss never did. | - | ntk_lib.grokking_time |
| `t_train_loss2e-2` | First checkpoint at which the training loss is below 0.02. | - | ntk_lib.grokking_time |
| `t_test_loss2e-2` | First checkpoint at which the test loss is below 0.02. | - | ntk_lib.grokking_time |
| `t_grok_loss2e-2` | Grokking time under the loss definition with both thresholds 0.02. | - | ntk_lib.grokking_time. docs/report.tex, Defining t_grok |
| `censored_loss2e-2` | True if the test loss never fell below 0.02, or the training loss never did. | - | ntk_lib.grokking_time |
| `A_t_at_mem` | A_t at t_mem. Empty if the event did not happen. | - | - |
| `S_c_at_mem` | S_c at t_mem. Empty if the event did not happen. | - | - |
| `R_c_at_mem` | R_c at t_mem. Empty if the event did not happen. | - | - |
| `gamma_at_mem` | gamma at t_mem. Empty if the event did not happen. | - | - |
| `D_at_mem` | D at t_mem. Empty if the event did not happen. | - | - |
| `weight_norm_at_mem` | weight_norm at t_mem. Empty if the event did not happen. | - | - |
| `lr_factor_at_mem` | lr_factor at t_mem. Empty if the event did not happen. | - | - |
| `S_sum_at_mem` | S_sum at t_mem. Empty if the event did not happen. | - | - |
| `R_sum_at_mem` | R_sum at t_mem. Empty if the event did not happen. | - | - |
| `D_sum_at_mem` | D_sum at t_mem. Empty if the event did not happen. | - | - |
| `A_sum_at_mem` | A_sum at t_mem. Empty if the event did not happen. | - | - |
| `S_W1_at_mem` | S_W1 at t_mem. Empty if the event did not happen. | - | - |
| `S_W2_at_mem` | S_W2 at t_mem. Empty if the event did not happen. | - | - |
| `S_W3_at_mem` | S_W3 at t_mem. Empty if the event did not happen. | - | - |
| `A_t_at_gen` | A_t at t_gen_acc80. Empty if the event did not happen. | - | - |
| `S_c_at_gen` | S_c at t_gen_acc80. Empty if the event did not happen. | - | - |
| `R_c_at_gen` | R_c at t_gen_acc80. Empty if the event did not happen. | - | - |
| `gamma_at_gen` | gamma at t_gen_acc80. Empty if the event did not happen. | - | - |
| `D_at_gen` | D at t_gen_acc80. Empty if the event did not happen. | - | - |
| `weight_norm_at_gen` | weight_norm at t_gen_acc80. Empty if the event did not happen. | - | - |
| `lr_factor_at_gen` | lr_factor at t_gen_acc80. Empty if the event did not happen. | - | - |
| `S_sum_at_gen` | S_sum at t_gen_acc80. Empty if the event did not happen. | - | - |
| `R_sum_at_gen` | R_sum at t_gen_acc80. Empty if the event did not happen. | - | - |
| `D_sum_at_gen` | D_sum at t_gen_acc80. Empty if the event did not happen. | - | - |
| `A_sum_at_gen` | A_sum at t_gen_acc80. Empty if the event did not happen. | - | - |
| `S_W1_at_gen` | S_W1 at t_gen_acc80. Empty if the event did not happen. | - | - |
| `S_W2_at_gen` | S_W2 at t_gen_acc80. Empty if the event did not happen. | - | - |
| `S_W3_at_gen` | S_W3 at t_gen_acc80. Empty if the event did not happen. | - | - |
| `t_A45` | First checkpoint at which A_t reaches 0.45. Empty if it never does. | - | - |
| `t_A50` | First checkpoint at which A_t reaches 0.5. Empty if it never does. | - | - |
| `verdict` | strong, weak or none, by the criterion of exploration/PLAN_mnist_closed_form.md section 3 with memorisation at 0.99. Strong: test accuracy at t_mem at most 0.30, best test accuracy at least 0.80, and t_gen_acc50 / t_mem at least 3. Weak: not strong, a gain after t_mem of at least 0.25, and the step at half that gain at least 3 t_mem. | - | exploration/PLAN_mnist_closed_form.md, section 3 |
