"""The features that the MNIST sweep records at every checkpoint of a run.

This is the MNIST counterpart of repoduced-code/dataset_features.py, which
records the features of Zain's modular addition sweep. It follows the same
design. FeatureLogger is the on_checkpoint callback of ntk_lib.train_run, so
the training loop is not forked, and it builds one row per checkpoint. Every
column of the row is listed in COLUMNS with its group, meaning, formula and
source. mnist_sweep.py writes the data dictionary from that list and refuses
to assemble a table with a column the list does not name.

The network is the bias-free three layer ReLU network of the MNIST study,
ntk_lib.DeepBiasFreeMLP with standard parameterisation, trained uncentred. The
columns keep the names of dataset_features.py wherever the quantity is the
same, so the two datasets can be read with the same code. They differ in the
following ways.

- The per-layer columns run over three weight matrices, W1 to W3.
- The Fourier columns, the subspace energies of (a + b), (a - b), a and b, and
  the full grid of p^2 pairs exist only for modular addition and are left out.
  The energy in the centred label span, probe_energy_label, is kept.
- The eigenvalue gaps and the label capture use C - 1 = 9, the dimension of
  the centred label span, where dataset_features.py uses multiples of p - 1.
- The report kernel of golden report Equations 5 and 6 is computed on a fixed
  draw of 1000 test images. The full test set of 10000 would need a kernel of
  10^8 entries at every checkpoint.
- The learning rate changes during training under the curvature rule, so the
  current rate and the latest Gauss-Newton eigenvalue are recorded.
- The kernel ridge regression runs on the probe only.

The closed-form kernel with output directions, kernel_terms, generalises
ntk_lib.closed_form_terms_deep. With the identity it is the sum-over-outputs
kernel of the probe. With e_0 it is the first-logit kernel. With the vector of
ones over sqrt(C) it is the kernel of the sum of the logits over sqrt(C), the
kernel of golden report Equation 5. mnist_sweep.py check compares all three
with autograd.

Sources. Cortes, Mohri, and Rostamizadeh (2012), Journal of Machine Learning
Research 13, pages 795 to 828, Lemma 1, for the centred kernel H K H. Every
other column is the group's own definition or a value copied from the
history of ntk_lib.train_run, as COLUMNS states for each.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

import ntk_lib as L

# Raise this when a column is added, removed or computed differently. The
# runner compares it with the version saved next to each run.
FEATURE_VERSION = 1
# The ridge of the kernel regression, as a fraction of the mean diagonal of
# the training block of the kernel.
KRR_RIDGE = 1e-3
# Power iterations for the sharpness, warm started from the previous checkpoint.
SHARPNESS_ITERS = 20
# A centred kernel whose Frobenius norm is below this is treated as zero.
DEGENERATE = 1e-30
# A run with weight decay is called stationary when the gradient and the decay
# pull cancel to this relative size.
STATIONARY = 1e-3
SPLITS = ("train", "test")
DEPTH = 3
N_CLASSES = 10
LAYERS = tuple(f"W{l}" for l in range(1, DEPTH + 1))
HIDDEN = tuple(f"h{l}" for l in range(1, DEPTH))
# The dimension of the centred label span, and the eigenvector counts tracked.
LABEL_DIM = N_CLASSES - 1
TOP_K = 4 * LABEL_DIM
CAPTURE = (LABEL_DIM, 2 * LABEL_DIM, TOP_K)
# The report kernel is computed on this many test images, drawn once by this seed.
REPORT_SIZE, REPORT_SEED = 1000, 0


# =====================================================================
# Column specification
# =====================================================================
COLUMNS = []


def col(name, group, meaning, formula="-", source="-", units="-"):
    COLUMNS.append(dict(name=name, group=group, meaning=meaning, formula=formula, source=source, units=units))


HIST = "ntk_lib.train_run history, copied at the checkpoint"
OWN = "The group's own definition"
REPORT = "docs/report.tex, Kernel metrics section"
CMR = "Cortes, Mohri and Rostamizadeh (2012), Lemma 1, for H K H"
GOLDEN = "docs/report_golden.pdf, Equations 5 and 6"
ZAIN = "repoduced-code/dataset_features.py, the same column for modular addition"

col("run_id", "ids", "The name of the run, mnist_sweep.run_id.", source="mnist_sweep.run_id")
col("init_scale", "ids", "The initial scale s. Every initial weight is multiplied by s. For this 3-homogeneous "
    "network the initial output is s^3 and the initial kernel s^4 times its value at s = 1.",
    source="Liu, Michaud and Tegmark (2023), Omnigrok, arXiv:2210.01117, Section 4")
col("width", "ids", "The width of both hidden layers.")
col("eta_kappa", "ids", "The weight decay per step, eta lambda in the report. It stays fixed when the learning "
    "rate changes.", source="docs/report.tex, The laziness and weight-decay knobs")
col("seed", "ids", "The model seed.")
col("data_seed", "ids", "The seed of the draw of the training images, 42 + seed.")
col("step", "grid", "The checkpoint step.", units="steps")
col("log10_step1", "grid", "Base 10 logarithm of the step plus one.", formula="log10(step + 1)")
col("dt_prev", "grid", "Steps since the previous checkpoint. NaN at step 0.", units="steps")
col("on_linear_grid", "grid", "True if the step is a multiple of the linear checkpoint interval.")
col("on_log_grid", "grid", "True if the step is one of the log-spaced checkpoints.")
col("wall_seconds", "grid", "Wall-clock seconds since the run started. It depends on the machine and the load.",
    units="s")

for split in SPLITS:
    col(f"{split}_loss", "performance", f"Mean squared error of the network on the {split} set.", source=HIST)
    col(f"{split}_acc", "performance", f"Accuracy of the network on the {split} set, by the largest output.",
        source=HIST)
for split in SPLITS:
    col(f"{split}_margin_mean", "performance",
        f"Mean over the {split} set of the correct output minus the largest other output.",
        formula="mean_i (f_i[y_i] - max_{c != y_i} f_i[c])", source=ZAIN)
    col(f"{split}_margin_min", "performance", f"Smallest margin over the {split} set.", source=ZAIN)
    col(f"{split}_correct_out_mean", "performance",
        f"Mean output at the correct class on the {split} set. The MSE target there is one.", source=ZAIN)
    col(f"{split}_wrong_out_rms", "performance",
        f"Root mean square output at the wrong classes on the {split} set. The MSE target there is zero.",
        source=ZAIN)
    col(f"{split}_out_rms", "performance", f"Root mean square of all outputs on the {split} set, the output scale.",
        source=ZAIN)
    col(f"{split}_out_max_mean", "performance", f"Mean over the {split} set of the largest output.", source=ZAIN)

col("lr", "learning_rate", "The learning rate in force for the steps after the checkpoint. The curvature rule "
    "sets it to eta_0 lam_0 / lam_t every lr_update_interval steps.", source=HIST + " (eta)")
col("lr_factor", "learning_rate", "lr over its value at step 0.", formula="lr / eta_0", source=OWN)
col("lam", "learning_rate", "The latest top Gauss-Newton eigenvalue of the training loss, from the last update "
    "of the learning rate.", formula="lambda_max((2 / (n C)) J^T J)", source=HIST)

col("grad_norm", "optimisation", "Norm of the gradient of the full training loss, without the decay term.",
    formula="||dL/dtheta||", source=ZAIN)
for layer in LAYERS:
    col(f"grad_norm_{layer}", "optimisation", f"Norm of the gradient of the full training loss with respect to "
        f"{layer}.", source=ZAIN)
col("update_norm", "optimisation", "Norm of a full-batch gradient step with decay at the current learning rate. "
    "Training takes minibatch steps, so this is the size of the mean step.",
    formula="lr ||g + wd theta||, wd = eta_kappa / lr", source=ZAIN)
col("grad_weight_cos", "optimisation", "Cosine between the loss gradient and the weights. Near minus one at a "
    "point where decay and gradient cancel.", formula="<g, theta> / (||g|| ||theta||)", source=ZAIN)
col("grad_decay_ratio", "optimisation", "Size of the loss gradient over the size of the decay pull. NaN without "
    "decay.", formula="||g|| / (wd ||theta||)", source=ZAIN)
col("decay_residual", "optimisation", "How far the gradient and the decay pull are from cancelling. Zero at a "
    "fixed point of gradient descent with decay, one when the two are orthogonal.",
    formula="||g + wd theta|| / (||g|| + wd ||theta||)", source=ZAIN)
col("stationary", "optimisation", f"True if the run has decay and decay_residual is below {STATIONARY}.",
    source=ZAIN)
col("sharpness", "optimisation", "Largest eigenvalue of the Hessian of the full training loss, without the decay "
    f"term, by {SHARPNESS_ITERS} power iterations with Hessian-vector products, warm started.",
    formula="lambda_max(d^2 L / d theta^2)", source=ZAIN)
col("lr_sharpness", "optimisation", "Learning rate times sharpness. Gradient descent on a quadratic with this "
    "curvature is stable when it is below 2.", formula="lr lambda_max", source=ZAIN)

col("weight_norm", "weights", "Norm of all the parameters.", source=HIST)
col("param_dist", "weights", "Relative parameter movement.", formula="||theta_t - theta_0|| / ||theta_0||",
    source=HIST)
col("log_weight_norm_ratio", "weights", "Log of the weight norm over its value at step 0. The kernel of this "
    "3-homogeneous network under a uniform rescaling of the weights by c is c^4 times larger, so four times this "
    "value is the scale term such a rescaling would give.", formula="log(||theta_t|| / ||theta_0||)", source=OWN)
for layer in LAYERS:
    col(f"{layer}_norm", "weights", f"Frobenius norm of {layer}.", source=ZAIN)
    col(f"{layer}_dist", "weights", f"Relative movement of {layer}.",
        formula=f"||{layer}_t - {layer}_0|| / ||{layer}_0||", source=ZAIN)
    col(f"{layer}_stable_rank", "weights", f"Stable rank of {layer}.", formula="||W||_F^2 / sigma_max^2",
        source=ZAIN)
    col(f"{layer}_eff_rank", "weights", f"Effective rank of {layer}, the exponential of the entropy of its "
        "singular values normalised to sum to one.", formula="exp(-sum q log q), q = sigma / sum sigma", source=ZAIN)
for a, b in zip(LAYERS[:-1], LAYERS[1:]):
    col(f"layer_imbalance_{a[1:]}{b[1:]}", "weights", f"Difference of the squared norms of {a} and {b} over the "
        "squared total norm. Gradient flow without decay keeps ||W_l||^2 - ||W_(l+1)||^2 fixed for a bias-free "
        "ReLU network. With decay it shrinks by (1 - eta_kappa)^2 per step.",
        formula=f"(||{a}||^2 - ||{b}||^2) / ||theta||^2", source=ZAIN)
for h in HIDDEN:
    col(f"dead_frac_{h}", "weights", f"Fraction of the units of hidden layer {h[1:]} that are inactive on every "
        "training image.", source=ZAIN)
    col(f"active_frac_{h}", "weights", f"Mean over training images of the fraction of the units of hidden layer "
        f"{h[1:]} that are active.", source=ZAIN)

col("S_t", "kernel_raw", "Scale term on the raw probe kernel.", formula="log(||K_t||_F / ||K_0||_F)", source=HIST)
col("R_t", "kernel_raw", "Rotation term on the raw probe kernel.", formula="1 - <K_t / ||K_t||, K_0 / ||K_0||>",
    source=HIST)
col("K_norm", "kernel_raw", "Frobenius norm of the raw probe kernel, the sum-over-outputs kernel.", source=HIST)
col("yKy", "kernel_raw", "y^T K^+ y on the probe set, summed over the label columns.", source=HIST)
col("S_c", "kernel_centred", "Scale term on the centred probe kernel.", formula="log(||Kc_t||_F / ||Kc_0||_F)",
    source=REPORT + "; " + HIST)
col("R_c", "kernel_centred", "Rotation term on the centred probe kernel.", formula="1 - <k_t, k_0>",
    source=REPORT + "; " + HIST)
col("A_t", "kernel_centred", "Centred kernel-target alignment on the probe.",
    formula="<Kc_t, G> / (||Kc_t|| ||G||), G = H Y Y^T H",
    source="Cortes, Mohri and Rostamizadeh (2012), Definition 4; " + REPORT + "; " + HIST)
col("A_u", "kernel_centred", "Uncentred alignment on the probe.", formula="<K_t, Y Y^T> / (||K_t|| ||Y Y^T||)",
    source="ntk_lib.uncentred_alignment; " + HIST)
for part in SPLITS:
    for term, word in (("S_c", "scale term"), ("R_c", "rotation term"), ("A_t", "centred alignment"),
                       ("A_u", "uncentred alignment")):
        col(f"{term}_{part}", "kernel_centred", f"The {word} on the {part} part of the probe, with the kernel "
            "restricted to those images.", source=REPORT + "; " + HIST)
col("K_norm_centred", "kernel_centred", "Frobenius norm of the centred probe kernel.", formula="||Kc_t||_F",
    source=CMR)
col("inner_K0", "kernel_centred", "Inner product of the centred kernel with the centred kernel at step 0.",
    formula="<Kc_t, Kc_0>", source=ZAIN)
col("inner_G", "kernel_centred", "Inner product of the centred kernel with the centred target Gram matrix.",
    formula="<Kc_t, G>", source=ZAIN)
col("D", "kernel_centred", "The usual kernel movement statistic on the centred kernel.",
    formula="||Kc_t - Kc_0||_F / ||Kc_0||_F", source="docs/report.tex, Equation eq:decomp")
col("exp2S_term", "kernel_centred", "First piece of Equation eq:decomp. D^2 = exp2S_term + 1 - cross_term.",
    formula="e^(2 S_c)", source="docs/report.tex, Equation eq:decomp, the group's own algebra")
col("cross_term", "kernel_centred", "Last piece of Equation eq:decomp.", formula="2 e^S_c (1 - R_c)",
    source="docs/report.tex, Equation eq:decomp, the group's own algebra")
col("gamma", "kernel_centred", "The aim: the cosine between the part of k_t orthogonal to k_0 and the part of "
    "the unit target orthogonal to k_0. NaN at step 0.",
    formula="<v_t, g_perp> / ||v_t||, v_t = k_t - <k_t, k_0> k_0", source=ZAIN)
col("share_k0", "kernel_centred", "Share of the squared change of the unit kernel along k_0. It equals R_c / 2.",
    formula="<d_t, k_0>^2 / ||d_t||^2, d_t = k_t - k_0", source=ZAIN)
col("share_gperp", "kernel_centred", "Share of the squared change of the unit kernel along g_perp.",
    formula="<d_t, g_perp>^2 / ||d_t||^2", source=ZAIN)
col("share_residual", "kernel_centred", "Share of the squared change of the unit kernel orthogonal to both k_0 "
    "and g_perp. The three shares sum to one.", source=ZAIN)
col("kernel_degenerate", "kernel_centred", f"True if the centred probe kernel norm is below {DEGENERATE}. The "
    "unit-kernel columns are then NaN.", source=ZAIN)

for layer in LAYERS:
    col(f"S_{layer}", "layer_kernels", f"Scale term of the centred kernel term from the gradient of {layer} alone.",
        formula=f"log(||Kc_{layer},t|| / ||Kc_{layer},0||)", source="kernel_terms; " + OWN)
    col(f"R_{layer}", "layer_kernels", f"Rotation term of the centred kernel term from {layer} alone.",
        source="kernel_terms; " + OWN)
    col(f"A_{layer}", "layer_kernels", f"Centred alignment of the kernel term from {layer} alone.",
        source="kernel_terms; " + OWN)
    col(f"{layer}_kernel_share", "layer_kernels", f"Share of the trace of the centred probe kernel that comes from "
        f"{layer}.", formula=f"tr(Kc_{layer}) / tr(Kc)", source="kernel_terms; " + OWN)
for term, word in (("S", "scale term"), ("R", "rotation term"), ("A", "centred alignment")):
    col(f"{term}_first_logit", "first_logit", f"The {word} of the centred kernel of the first output alone. The "
        "report names this kernel as a robustness check.", source=REPORT + "; kernel_terms")

for term, word in (("S", "Scale, the plain ratio of Frobenius norms to step 0"),
                   ("R", "Shape change, one minus the uncentred cosine to step 0"),
                   ("D", "Variation, the Frobenius norm of the change from step 0 over the norm at step 0"),
                   ("A", "Centred alignment with Y Y^T")):
    col(f"{term}_sum", "report_kernel", f"{word}, of the kernel of the sum of the logits divided by sqrt(C), on a "
        f"fixed draw of {REPORT_SIZE} test images. These are the S_t, R_t, D_t and A_t of the golden report.",
        source=GOLDEN + "; kernel_terms")

col("eig_trace", "spectrum", "Trace of the centred probe kernel.", source=ZAIN)
col("eig_top1", "spectrum", "Largest eigenvalue of the centred probe kernel.", source=ZAIN)
col("eig_top1_share", "spectrum", "Largest eigenvalue over the trace.", source=ZAIN)
col("eig_eff_rank", "spectrum", "Exponential of the entropy of the eigenvalues normalised to sum to one. The "
    "zero eigenvalue that centring creates is left out, and so is every column below.", source=ZAIN)
col("eig_participation", "spectrum", "Participation ratio of the eigenvalues.",
    formula="(sum lambda)^2 / sum lambda^2", source=ZAIN)
col("eig_n90", "spectrum", "Number of the largest eigenvalues that hold 90 percent of the trace.", source=ZAIN)
col("eig_gap_label", "spectrum", f"Ratio of eigenvalue {LABEL_DIM} to eigenvalue {LABEL_DIM + 1}, counting from "
    f"one. The centred labels span {LABEL_DIM} dimensions.", source=OWN)
col(f"eig_gap_top{TOP_K}", "spectrum", f"Ratio of eigenvalue {TOP_K} to eigenvalue {TOP_K + 1}, the edge of the "
    "eigenvectors that the tracking columns follow.", source=OWN)
col("probe_energy_label", "subspaces", "Share of the trace of the centred probe kernel inside the centred label "
    "span on the probe.", formula="tr(Q^T Kc Q) / tr(Kc), Q an orthonormal basis of H Y", source=ZAIN)
for m in CAPTURE:
    col(f"label_capture_{m}", "eigvec_tracking", f"Share of the centred label matrix inside the top {m} "
        "eigenvectors of the centred probe kernel.", formula=f"||V_{m}^T H Y||_F^2 / ||H Y||_F^2", source=ZAIN)
col("topk_drift", "eigvec_tracking", f"Mean squared sine of the principal angles between the span of the top "
    f"{TOP_K} eigenvectors and the same span at step 0.", source=ZAIN)
col("overlap_zero_median", "eigvec_tracking", f"Median over the top {TOP_K} eigenvectors of the largest absolute "
    f"overlap with any of the top {TOP_K} eigenvectors at step 0. dataset_features.py follows a chain of matches "
    "back to step 0 instead.", source=OWN)
col("krr_probe_test_acc", "probe_krr", "Accuracy of kernel ridge regression with the current probe kernel from the "
    f"training part of the probe to its test part. The ridge is {KRR_RIDGE} times the mean diagonal of the training "
    "block.", formula="argmax K_te,tr (K_tr,tr + r I)^-1 Y_tr", source=ZAIN)
col("krr_probe_test_mse", "probe_krr", "Mean squared error of the same kernel ridge regression on the test part.",
    source=ZAIN)

col("dS_step", "instant_rates", "Change of the centred scale term over one virtual full-batch gradient step with "
    "decay at the current learning rate, in float64.", formula="S(K_{t+1}) - S(K_t)", source=ZAIN)
col("dA_step", "instant_rates", "Change of the centred alignment over the same virtual step.",
    formula="A(K_{t+1}) - A(K_t)", source=ZAIN)
col("R_step", "instant_rates", "Rotation of the centred unit kernel over the same virtual step.",
    formula="1 - <k_{t+1}, k_t>", source=ZAIN)

COLUMN_NAMES = [c["name"] for c in COLUMNS]
HISTORY_COPIED = ["train_loss", "test_loss", "train_acc", "test_acc", "S_t", "R_t", "K_norm", "yKy", "S_c", "R_c",
                  "A_t", "A_u", "weight_norm", "param_dist"] + L.PART_KEYS


# =====================================================================
# Helpers
# =====================================================================
def centre(K):
    """Return H K H for a float64 numpy matrix, Cortes et al. (2012), Lemma 1."""
    K = np.asarray(K, dtype=np.float64)
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()


def kernel_terms(weights, factors, x, U):
    """Per-layer terms of the kernel of the outputs U^T f of a bias-free ReLU network, in float64.

    weights are W_1 to W_L, factors c_1 to c_L, and U is a C by m matrix of
    output directions. The kernel is sum_j <grad (U^T f)_j(x), grad (U^T f)_j(x')>.
    As in ntk_lib.closed_form_terms_deep, the term of layer l is
    c_l^2 <a_{l-1}(x), a_{l-1}(x')> <D_l(x), D_l(x')>_F, where now D_L = U and
    D_l = M_l * (c_{l+1} W_{l+1}^T D_{l+1}). U = I gives the sum-over-outputs
    kernel of closed_form_terms_deep, U = e_0 the first-logit kernel, and
    U = 1 / sqrt(C) the kernel of the sum of the logits over sqrt(C).
    """
    X = x.detach().double()
    Ws = [W.detach().double() for W in weights]
    U = torch.as_tensor(U, dtype=torch.float64, device=X.device)
    n_layers = len(Ws)
    acts, masks = [X], []
    for l in range(n_layers - 1):
        h = factors[l] * (acts[-1] @ Ws[l].T)
        masks.append((h > 0).double())
        acts.append(torch.relu(h))
    terms = [None] * n_layers
    terms[-1] = factors[-1] ** 2 * float((U ** 2).sum()) * (acts[-1] @ acts[-1].T)
    D = None
    for l in range(n_layers - 2, -1, -1):
        back = factors[l + 1] * Ws[l + 1].T                       # width_l by width_{l+1}
        if D is None:
            D = masks[l][:, :, None] * (back @ U)[None]          # n by width_l by m
        else:
            D = masks[l][:, :, None] * torch.einsum("ik,nkm->nim", back, D)
        Dflat = D.reshape(D.shape[0], -1)
        terms[l] = factors[l] ** 2 * (acts[l] @ acts[l].T) * (Dflat @ Dflat.T)
    return terms


def report_statistics(K_t, K_0, Y):
    """S_t, R_t, D_t and A_t of golden report Equation 6, as ntk_lib.report_statistics on the zain branch."""
    n_t, n_0 = np.linalg.norm(K_t), np.linalg.norm(K_0)
    Kc, Gc = centre(K_t), centre(Y @ Y.T)
    return dict(S=n_t / n_0, R=1 - np.sum(K_t * K_0) / (n_t * n_0), D=np.linalg.norm(K_t - K_0) / n_0,
                A=np.sum(Kc * Gc) / (np.linalg.norm(Kc) * np.linalg.norm(Gc)))


def orthonormal_basis(M, rtol=1e-10):
    """An orthonormal basis of the column span of M."""
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > rtol * s[0]]


def krr(K, Y, train, test, ridge=KRR_RIDGE):
    """Kernel ridge regression from the train rows to the test rows. Returns accuracy and mean squared error."""
    Ktr = K[np.ix_(train, train)]
    r = ridge * np.trace(Ktr) / len(train)
    if not np.isfinite(r) or r <= 0:
        return np.nan, np.nan
    coef = np.linalg.solve(Ktr + r * np.eye(len(train)), Y[train])
    pred = K[np.ix_(test, train)] @ coef
    return float(np.mean(pred.argmax(1) == Y[test].argmax(1))), float(np.mean((pred - Y[test]) ** 2))


def effective_rank(values):
    """Exponential of the entropy of nonnegative values normalised to sum to one."""
    v = np.clip(np.asarray(values, float), 0.0, None)
    total = v.sum()
    if total <= 0:
        return np.nan
    q = v[v > 0] / total
    return float(np.exp(-np.sum(q * np.log(q))))


# =====================================================================
# The logger
# =====================================================================
class FeatureLogger:
    """The on_checkpoint callback that builds one feature row per checkpoint.

    Construct it with the full configuration of the run. train_run calls it
    once per checkpoint, starting at step 0 and in step order. After the run,
    rows holds the rows, weights holds the saved weight matrices keyed by
    step, and check_values holds the terms that identity_checks compares.

    The callback must not change the model or the random state of training.
    Every gradient here is taken with torch.autograd.grad, which does not touch
    .grad, and the power iteration uses its own random generator.
    """

    def __init__(self, cfg, run_id, save_steps=(), linear_interval=2000, log_steps=(), device=None):
        if (cfg["parameterisation"] != "deep_standard" or cfg["depth"] != DEPTH or cfg["centred"]
                or cfg["alpha"] != 1.0 or cfg["dataset"] != "mnist"):
            raise ValueError("the MNIST feature logger needs the uncentred bias-free network of depth 3 on MNIST")
        self.cfg, self.run_id = cfg, run_id
        self.eta_0, self.eta_kappa = cfg["eta_0"], cfg["eta_kappa"]
        self.save_steps = set(int(s) for s in save_steps)
        self.linear_interval, self.log_steps = linear_interval, set(int(s) for s in log_steps)
        self.ids = dict(run_id=run_id, init_scale=cfg["init_scale"], width=cfg["hidden_dim"],
                        eta_kappa=cfg["eta_kappa"], seed=cfg["seed"], data_seed=cfg["data_seed"])

        (X_train, y_train), (X_test, y_test) = L.make_dataset(cfg)
        self.X = dict(train=X_train, test=X_test)
        self.y = dict(train=y_train, test=y_test)
        self.labels = dict(train=y_train.argmax(1), test=y_test.argmax(1))
        report = np.random.default_rng(REPORT_SEED).choice(len(X_test), REPORT_SIZE, replace=False)
        self.X_report, self.Y_report = X_test[report], y_test[report].double().numpy()
        self.K_sum_0 = None
        self.U = dict(trace=np.eye(N_CLASSES), first=np.eye(N_CLASSES)[:, :1],
                      sum=np.ones((N_CLASSES, 1)) / math.sqrt(N_CLASSES))

        # The probe, exactly as train_run takes it.
        self.x_probe, y_probe, is_test = L.select_probe(X_train, y_train, X_test, y_test, cfg["probe"],
                                                        cfg["probe_size"])
        self.Y_probe = y_probe.numpy().astype(np.float64)
        self.probe_train = np.nonzero(~is_test.numpy())[0]
        self.probe_test = np.nonzero(is_test.numpy())[0]
        self.HY = self.Y_probe - self.Y_probe.mean(0, keepdims=True)
        G = centre(self.Y_probe @ self.Y_probe.T)
        self.G = G
        self.G_unit = G / np.linalg.norm(G)
        self.label_basis = orthonormal_basis(self.HY)
        # The logger evaluates the model on the model's device, which device names when it is not the CPU.
        if device is not None:
            self.X = {s: t.to(device) for s, t in self.X.items()}
            self.y = {s: t.to(device) for s, t in self.y.items()}
            self.labels = {s: t.to(device) for s, t in self.labels.items()}
            self.X_report, self.x_probe = self.X_report.to(device), self.x_probe.to(device)

        self.refs = {}
        self.weights_0 = None
        self.rows, self.weights = [], {}
        self.check_values = {k: [] for k in ("step", "S", "R", "A", "D", "exp2S_term", "cross_term", "gamma",
                                             "share_k0", "share_gperp", "share_residual", "S_c", "R_c", "A_t")}
        self.A_0 = np.nan
        self.last_step, self.last_weights = None, None
        self.sharp_vec = None
        self.rng = torch.Generator().manual_seed(12345)
        self.t_start = time.time()
        self.callback_seconds = 0.0

    # -----------------------------------------------------------------
    def terms(self, name, Kc, G_unit):
        """S, R and A of a centred kernel against its step 0 reference under name, or NaNs if it is zero."""
        if name not in self.refs:
            norm = np.linalg.norm(Kc)
            self.refs[name] = (Kc / norm, norm)
        k0, norm0 = self.refs[name]
        norm = np.linalg.norm(Kc)
        if norm < DEGENERATE:
            return np.nan, np.nan, np.nan
        k = Kc / norm
        return float(np.log(norm / norm0)), float(1.0 - np.sum(k * k0)), float(np.sum(k * G_unit))

    # -----------------------------------------------------------------
    def __call__(self, step, model, K_t, row):
        t_call = time.time()
        Ws = model.weights()
        if self.weights_0 is None:
            if step != 0:
                raise ValueError(f"the first checkpoint must be step 0, not {step}")
            self.weights_0 = [W.detach().double().clone() for W in Ws]
        out = dict(self.ids)
        out.update(step=int(step), log10_step1=math.log10(step + 1),
                   dt_prev=np.nan if self.last_step is None else float(step - self.last_step),
                   on_linear_grid=bool(step % self.linear_interval == 0), on_log_grid=bool(step in self.log_steps),
                   wall_seconds=t_call - self.t_start)
        out.update({k: row[k] for k in HISTORY_COPIED})
        lr = row.get("eta", self.eta_0)
        out.update(lr=lr, lr_factor=lr / self.eta_0, lam=row.get("lam", np.nan))
        out.update(self.performance(model))
        grads = self.optimisation(model, lr, out)
        out.update(self.weight_features(model))
        out.update(self.kernel_features(step, K_t, row, model, lr, grads))
        missing, extra = set(COLUMN_NAMES) - set(out), set(out) - set(COLUMN_NAMES)
        if missing or extra:
            raise KeyError(f"feature row does not match COLUMNS: missing {sorted(missing)}, extra {sorted(extra)}")
        self.rows.append({name: out[name] for name in COLUMN_NAMES})

        current = [W.detach().cpu().numpy().astype(np.float32).copy() for W in Ws]
        if step in self.save_steps:
            self.weights[int(step)] = current
        self.last_step, self.last_weights = int(step), current
        self.callback_seconds += time.time() - t_call

    # -----------------------------------------------------------------
    def performance(self, model):
        out = {}
        with torch.no_grad():
            for s in SPLITS:
                f = model(self.X[s]).double()
                lab = self.labels[s]
                idx = torch.arange(len(lab), device=lab.device)
                correct = f[idx, lab]
                others = f.clone()
                others[idx, lab] = -torch.inf
                margin = correct - others.max(1).values
                wrong = f.clone()
                wrong[idx, lab] = 0.0
                C = f.shape[1]
                out[f"{s}_margin_mean"] = margin.mean().item()
                out[f"{s}_margin_min"] = margin.min().item()
                out[f"{s}_correct_out_mean"] = correct.mean().item()
                out[f"{s}_wrong_out_rms"] = math.sqrt((wrong ** 2).sum().item() / (len(lab) * (C - 1)))
                out[f"{s}_out_rms"] = math.sqrt((f ** 2).mean().item())
                out[f"{s}_out_max_mean"] = f.max(1).values.mean().item()
        return out

    def optimisation(self, model, lr, out):
        """Gradient columns and sharpness. Returns the loss gradient per layer for the virtual step."""
        params = model.weights()
        loss = nn.functional.mse_loss(model(self.X["train"]), self.y["train"])
        grads = torch.autograd.grad(loss, params, create_graph=True)
        g = torch.cat([q.reshape(-1) for q in grads]).detach().double()
        theta = torch.cat([q.detach().reshape(-1) for q in params]).double()
        gn, tn = torch.linalg.norm(g).item(), torch.linalg.norm(theta).item()
        wd = self.eta_kappa / lr
        pull = wd * tn
        out["grad_norm"] = gn
        for layer, q in zip(LAYERS, grads):
            out[f"grad_norm_{layer}"] = torch.linalg.norm(q.detach().double()).item()
        step = torch.linalg.norm(g + wd * theta).item()
        out["update_norm"] = lr * step
        out["grad_weight_cos"] = (g @ theta).item() / (gn * tn) if gn > 0 and tn > 0 else np.nan
        out["grad_decay_ratio"] = gn / pull if pull > 0 else np.nan
        out["decay_residual"] = step / (gn + pull) if gn + pull > 0 else np.nan
        out["stationary"] = bool(wd > 0 and out["decay_residual"] < STATIONARY)

        # Power iteration on the Hessian of the loss, warm started.
        v = [torch.randn(q.shape, generator=self.rng).to(q.device) for q in params] if self.sharp_vec is None else self.sharp_vec
        lam = np.nan
        for _ in range(SHARPNESS_ITERS):
            vn = math.sqrt(sum((x ** 2).sum().item() for x in v))
            if vn == 0 or not math.isfinite(vn):
                break
            v = [x / vn for x in v]
            Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
            lam = sum((a * b).sum().item() for a, b in zip(v, Hv))
            v = [h.detach() for h in Hv]
        vn = math.sqrt(sum((x ** 2).sum().item() for x in v))
        self.sharp_vec = [x / vn for x in v] if vn > 0 and math.isfinite(vn) else None
        out["sharpness"] = lam
        out["lr_sharpness"] = lr * lam
        return [q.detach() for q in grads]

    def weight_features(self, model):
        out = {}
        Ws = [W.detach().double() for W in model.weights()]
        norms = [torch.linalg.norm(W).item() for W in Ws]
        total2 = sum(n ** 2 for n in norms)
        n0 = math.sqrt(sum(torch.linalg.norm(W).item() ** 2 for W in self.weights_0))
        out["log_weight_norm_ratio"] = math.log(math.sqrt(total2) / n0) if total2 > 0 else np.nan
        for layer, W, W0, n in zip(LAYERS, Ws, self.weights_0, norms):
            sv = torch.linalg.svdvals(W).cpu().numpy()
            out[f"{layer}_norm"] = n
            out[f"{layer}_dist"] = torch.linalg.norm(W - W0).item() / torch.linalg.norm(W0).item()
            out[f"{layer}_stable_rank"] = float(n ** 2 / sv[0] ** 2) if sv[0] > 0 else np.nan
            out[f"{layer}_eff_rank"] = effective_rank(sv)
        for i in range(DEPTH - 1):
            out[f"layer_imbalance_{i + 1}{i + 2}"] = (norms[i] ** 2 - norms[i + 1] ** 2) / total2 \
                if total2 > 0 else np.nan
        with torch.no_grad():
            a = self.X["train"].double()
            for h_name, W, c in zip(HIDDEN, Ws, model.factors):
                h = c * (a @ W.T)
                active = h > 0
                out[f"dead_frac_{h_name}"] = (~active.any(0)).double().mean().item()
                out[f"active_frac_{h_name}"] = active.double().mean().item()
                a = torch.relu(h)
        return out

    def centred_terms(self, step, K):
        """The centred-kernel columns from the float32 probe kernel, as ntk_trace.Tracer.measure defines them."""
        Kc = centre(K)
        norm = np.linalg.norm(Kc)
        if step == 0:
            self.Kc0, self.norm_0 = Kc, norm
            self.k_0 = Kc / norm
            g_perp = self.G_unit - np.sum(self.G_unit * self.k_0) * self.k_0
            self.g_perp = g_perp / np.linalg.norm(g_perp)
            self.A_0 = float(np.sum(self.k_0 * self.G_unit))
        k_t = Kc / norm
        c = float(np.sum(k_t * self.k_0))
        S, R, A = float(np.log(norm / self.norm_0)), 1.0 - c, float(np.sum(k_t * self.G_unit))
        out = dict(K_norm_centred=float(norm), inner_K0=float(np.sum(Kc * self.Kc0)),
                   inner_G=float(np.sum(Kc * self.G)), D=float(np.linalg.norm(Kc - self.Kc0) / self.norm_0),
                   exp2S_term=float(np.exp(2 * S)), cross_term=float(2 * np.exp(S) * c))
        v = k_t - c * self.k_0
        vn = np.linalg.norm(v)
        out["gamma"] = float(np.sum(v * self.g_perp) / vn) if step > 0 and vn > 0 else np.nan
        d = k_t - self.k_0
        dn2 = np.sum(d * d)
        if step > 0 and dn2 > 0:
            out["share_k0"] = float(np.sum(d * self.k_0) ** 2 / dn2)
            out["share_gperp"] = float(np.sum(d * self.g_perp) ** 2 / dn2)
            out["share_residual"] = 1.0 - out["share_k0"] - out["share_gperp"]
        else:
            out["share_k0"] = out["share_gperp"] = out["share_residual"] = np.nan
        return out, dict(S=S, R=R, A=A), Kc

    def kernel_features(self, step, K_t, row, model, lr, grads):
        out = {}
        K64 = K_t.detach().cpu().numpy().astype(np.float64)
        Kc = centre(K64)
        degenerate = np.linalg.norm(Kc) < DEGENERATE
        out["kernel_degenerate"] = bool(degenerate)
        if degenerate and step == 0:
            raise ValueError("the centred probe kernel is zero at step 0")
        names = ["K_norm_centred", "inner_K0", "inner_G", "D", "exp2S_term", "cross_term", "gamma", "share_k0",
                 "share_gperp", "share_residual"]
        if not degenerate:
            m, sra, Kc = self.centred_terms(step, K64)
            out.update(m)
            values = {**sra, **m}
        else:
            out.update({k: np.nan for k in names})
            values = {k: np.nan for k in ("S", "R", "A", *names)}
        self.check_values["step"].append(int(step))
        for k in ("S", "R", "A", "D", "exp2S_term", "cross_term", "gamma", "share_k0", "share_gperp",
                  "share_residual"):
            self.check_values[k].append(float(values[k]))
        for k in ("S_c", "R_c", "A_t"):
            self.check_values[k].append(float(row[k]))

        # Spectrum of the centred kernel without the zero that centring creates.
        spectral = ["eig_trace", "eig_top1", "eig_top1_share", "eig_eff_rank", "eig_participation", "eig_n90",
                    "eig_gap_label", f"eig_gap_top{TOP_K}", *[f"label_capture_{m}" for m in CAPTURE],
                    "topk_drift", "overlap_zero_median", "probe_energy_label"]
        if not degenerate:
            eig, V = np.linalg.eigh(Kc)
            eig, V = eig[::-1][:-1], V[:, ::-1][:, :-1]
            pos = np.clip(eig, 0.0, None)
            tr = pos.sum()
            out["eig_trace"] = float(tr)
            out["eig_top1"] = float(eig[0])
            out["eig_top1_share"] = float(eig[0] / tr) if tr > 0 else np.nan
            out["eig_eff_rank"] = effective_rank(pos)
            out["eig_participation"] = float(tr ** 2 / np.sum(pos ** 2)) if tr > 0 else np.nan
            out["eig_n90"] = int(np.searchsorted(np.cumsum(pos) / tr, 0.9) + 1) if tr > 0 else np.nan
            out["eig_gap_label"] = float(eig[LABEL_DIM - 1] / eig[LABEL_DIM]) if eig[LABEL_DIM] > 0 else np.nan
            out[f"eig_gap_top{TOP_K}"] = float(eig[TOP_K - 1] / eig[TOP_K]) if eig[TOP_K] > 0 else np.nan
            hy2 = np.sum(self.HY ** 2)
            for m in CAPTURE:
                out[f"label_capture_{m}"] = float(np.sum((V[:, :m].T @ self.HY) ** 2) / hy2)
            Vk = V[:, :TOP_K]
            if step == 0:
                self.V_0 = Vk
            cosines = np.clip(np.linalg.svd(Vk.T @ self.V_0, compute_uv=False), 0.0, 1.0)
            out["topk_drift"] = float(np.mean(1.0 - cosines ** 2))
            out["overlap_zero_median"] = float(np.median(np.abs(Vk.T @ self.V_0).max(1)))
            Q = self.label_basis
            out["probe_energy_label"] = float(np.trace(Q.T @ Kc @ Q) / np.trace(Kc)) if np.trace(Kc) > 0 else np.nan
        else:
            out.update({k: np.nan for k in spectral})

        # The three layer terms and the first-logit kernel, in float64.
        Ws, factors = model.weights(), model.factors
        terms = [T.cpu().numpy() for T in kernel_terms(Ws, factors, self.x_probe, self.U["trace"])]
        centred = [centre(T) for T in terms]
        traces = [np.trace(T) for T in centred]
        for layer, T, t in zip(LAYERS, centred, traces):
            out[f"S_{layer}"], out[f"R_{layer}"], out[f"A_{layer}"] = self.terms(layer, T, self.G_unit)
            out[f"{layer}_kernel_share"] = float(t / sum(traces)) if sum(traces) > 0 else np.nan
        first = sum(kernel_terms(Ws, factors, self.x_probe, self.U["first"])).cpu().numpy()
        out["S_first_logit"], out["R_first_logit"], out["A_first_logit"] = self.terms(
            "first_logit", centre(first), self.G_unit)

        # The kernel of the golden report, Equations 5 and 6, on the report images.
        K_sum = sum(kernel_terms(Ws, factors, self.X_report, self.U["sum"])).cpu().numpy()
        if self.K_sum_0 is None:
            self.K_sum_0 = K_sum
        for k, v in report_statistics(K_sum, self.K_sum_0, self.Y_report).items():
            out[f"{k}_sum"] = float(v)

        out["krr_probe_test_acc"], out["krr_probe_test_mse"] = krr(K64, self.Y_probe, self.probe_train,
                                                                   self.probe_test)

        # One virtual full-batch step of gradient descent with decay, as torch.optim.SGD takes it.
        wd = self.eta_kappa / lr
        with torch.no_grad():
            nxt = [W.detach() - lr * (g + wd * W.detach()) for W, g in zip(Ws, grads)]
        Kc_now = sum(centred)
        Kc_next = centre(sum(kernel_terms(nxt, factors, self.x_probe, self.U["trace"])).cpu().numpy())
        n_now, n_next = np.linalg.norm(Kc_now), np.linalg.norm(Kc_next)
        if n_now < DEGENERATE or n_next < DEGENERATE:
            out["dS_step"] = out["dA_step"] = out["R_step"] = np.nan
        else:
            k_now, k_next = Kc_now / n_now, Kc_next / n_next
            out["dS_step"] = float(np.log(n_next / n_now))
            out["dA_step"] = float(np.sum(k_next * self.G_unit) - np.sum(k_now * self.G_unit))
            out["R_step"] = float(1.0 - np.sum(k_next * k_now))
        return out

    # -----------------------------------------------------------------
    def check_arrays(self):
        """The terms that identity_checks compares, as arrays, with A_0 at step 0."""
        a = {k: np.asarray(v, dtype=float) for k, v in self.check_values.items()}
        a["A_0"] = np.array(self.A_0)
        return a


def identity_checks(a):
    """Largest differences that should sit at rounding level, as dataset_features.identity_checks computes them.

    a is the output of FeatureLogger.check_arrays. The history fields S_c, R_c
    and A_t are the float32 values train_run records, so their differences
    from the float64 terms here sit at float32 rounding.
    """
    R, gamma, A_0 = a["R"], a["gamma"], float(a["A_0"])
    moved = np.arange(len(R)) > 0
    identity = A_0 * (1 - R) + gamma * np.sqrt(1 - A_0 ** 2) * np.sqrt(np.clip(R * (2 - R), 0, None))

    def worst(x):
        x = np.asarray(x, float)
        return float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else float("nan")

    return {
        "S against the history field S_c": worst(a["S"] - a["S_c"]),
        "R against the history field R_c": worst(a["R"] - a["R_c"]),
        "A against the history field A_t": worst(a["A"] - a["A_t"]),
        "D^2 against exp2S_term + 1 - cross_term": worst(a["D"] ** 2 - (a["exp2S_term"] + 1 - a["cross_term"])),
        "the three shares against a sum of one": worst(a["share_k0"] + a["share_gperp"] + a["share_residual"] - 1),
        "share_k0 against R / 2": worst(a["share_k0"] - R / 2),
        "A against the aim identity": worst((a["A"] - identity)[moved]),
    }
