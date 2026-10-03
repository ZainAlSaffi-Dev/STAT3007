"""The features that the dataset sweep records at every checkpoint of a run.

The dataset sweep in dataset_sweep.py trains every cell of the crossed design
and writes one row per checkpoint. This module computes that row. It is the
on_checkpoint callback of ntk_lib.train_run, in the same way as
ntk_trace.Tracer, so the training loop is not forked. The kernel terms that
the tier x study already defined come from ntk_trace.Tracer.measure and are
not computed again here.

Every column of the row is listed in COLUMNS with its group, its meaning, its
formula and its source. dataset_sweep.py writes the data dictionary from that
list and refuses to assemble a table with a column the list does not name.

Sources. Where a column comes from a paper, its entry in COLUMNS names the
paper and the place in it.
- Cortes, Mohri, and Rostamizadeh (2012), Journal of Machine Learning
  Research 13, pages 795 to 828. Lemma 1 gives the centred kernel H K H.
- Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12. The two layer
  network there, with a quadratic activation, has an exact solution in which
  each hidden unit is a cosine in a, a cosine in b and a cosine in the output
  class, all with one frequency, and the three phases satisfy
  phi_1 + phi_2 = phi_3. The Fourier columns measure how close the weights
  of this ReLU network come to that form. The paper does not predict that a
  ReLU network reaches it. Read on 29 September 2026.
- The closed-form kernel, its two layer terms, the aim, the three shares, the
  instantaneous rates, the kernel ridge regression and the stationarity flag
  are the group's own definitions. No paper states them.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

import ntk_lib as L
import ntk_trace as T
from RepoducedCode import make_modular_addition_dataset

# Raise this when a column is added, removed or computed differently. The
# runner compares it with the version saved next to each run, so a changed
# feature set is never mixed silently with an old one.
FEATURE_VERSION = 2
# The ridge of the kernel regressions, as a fraction of the mean diagonal of
# the training block of the kernel.
KRR_RIDGE = 1e-3
# The number of power iterations for the sharpness, warm started from the
# vector found at the previous checkpoint.
SHARPNESS_ITERS = 20
# A centred kernel whose Frobenius norm is below this is treated as zero.
DEGENERATE = 1e-30
# A run with weight decay is called stationary when the gradient and the
# decay pull cancel to this relative size.
STATIONARY = 1e-3
# The parts of the data on which the predictor columns are computed.
SPLITS = ("train", "test")
# The weight matrices whose Fourier spectrum is measured: the a-half and the
# b-half of W1 along the token axis, and W2 along the class axis.
FOURIER_MATS = ("a", "b", "c")
FOURIER_SOURCES = ("W", "dW")


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
PLAN = "docs/zain_tierx_plan.md"
GROMOV = "Gromov (2023), arXiv:2301.02679v1, Equations 6, 7 and 12, as motivation"

col("run_id", "ids", "The name of the run, the grid cell name of ntk_lib.cell_name.", source="ntk_lib.cell_name")
col("alpha", "ids", "The laziness knob alpha of the rescaled predictor alpha (f - f_0).",
    source="Kumar et al. (2024), Appendix 8.1, Equation 7, arXiv v3")
col("width", "ids", "The hidden width N.")
col("eta_kappa", "ids", "The weight decay as the product of learning rate and decay, eta lambda in the report.",
    source="docs/report.tex, The laziness and weight-decay knobs")
col("seed", "ids", "The model seed. The data seed is 42 for every run.")
col("step", "grid", "The checkpoint step.", units="steps")
col("log10_step1", "grid", "Base 10 logarithm of the step plus one.", formula="log10(step + 1)")
col("dt_prev", "grid", "Steps since the previous checkpoint. NaN at step 0.", units="steps")
col("on_linear_grid", "grid", "True if the step is a multiple of the linear checkpoint interval of 1,000 steps.")
col("on_log_grid", "grid", "True if the step is one of the log-spaced checkpoints.")
col("wall_seconds", "grid", "Wall-clock seconds since the run started. It depends on the machine and the load.",
    units="s")

for split in SPLITS:
    col(f"{split}_loss", "performance", f"Mean squared error of the predictor on the {split} set.", source=HIST)
    col(f"{split}_acc", "performance", f"Accuracy of the predictor on the {split} set, by the largest output.",
        source=HIST)
for split in SPLITS:
    col(f"{split}_margin_mean", "performance",
        f"Mean over the {split} set of the correct output minus the largest other output.",
        formula="mean_i (f_i[y_i] - max_{c != y_i} f_i[c])", source=OWN)
    col(f"{split}_margin_min", "performance", f"Smallest margin over the {split} set.", source=OWN)
    col(f"{split}_correct_out_mean", "performance",
        f"Mean output at the correct class on the {split} set. The MSE target there is one.", source=OWN)
    col(f"{split}_wrong_out_rms", "performance",
        f"Root mean square output at the wrong classes on the {split} set. The MSE target there is zero.",
        source=OWN)
    col(f"{split}_out_rms", "performance", f"Root mean square of all outputs on the {split} set, the output scale.",
        source=OWN)
    col(f"{split}_out_max_mean", "performance", f"Mean over the {split} set of the largest output.", source=OWN)

col("grad_norm", "optimisation", "Norm of the gradient of the training loss, without the decay term.",
    formula="||dL/dtheta||", source=OWN)
col("grad_norm_W1", "optimisation", "Norm of the gradient of the training loss with respect to W1.", source=OWN)
col("grad_norm_W2", "optimisation", "Norm of the gradient of the training loss with respect to W2.", source=OWN)
col("update_norm", "optimisation", "Norm of the next gradient-descent step, decay included.",
    formula="lr ||g + wd theta|| with lr = eta_0 / alpha^2 and wd = eta_kappa / lr", source="ntk_lib.train_run")
col("grad_weight_cos", "optimisation", "Cosine between the loss gradient and the weights. Near minus one at a "
    "point where decay and gradient cancel.", formula="<g, theta> / (||g|| ||theta||)", source=OWN)
col("grad_decay_ratio", "optimisation", "Size of the loss gradient over the size of the decay pull. NaN "
    "without decay.", formula="||g|| / (wd ||theta||)", source=OWN)
col("decay_residual", "optimisation", "How far the gradient and the decay pull are from cancelling. Zero at a "
    "fixed point of gradient descent with decay, one when the two are orthogonal.",
    formula="||g + wd theta|| / (||g|| + wd ||theta||)", source=OWN)
col("stationary", "optimisation", f"True if the run has decay and decay_residual is below {STATIONARY}.",
    source=OWN)
col("sharpness", "optimisation", "Largest eigenvalue of the Hessian of the training loss, without the decay "
    f"term, by {SHARPNESS_ITERS} power iterations with Hessian-vector products, warm started.",
    formula="lambda_max(d^2 L / d theta^2)", source=OWN)
col("lr_sharpness", "optimisation", "Learning rate times sharpness. Gradient descent on a quadratic with this "
    "curvature is stable when it is below 2.", formula="lr lambda_max", source=OWN)

col("weight_norm", "weights", "Norm of all the parameters.", source=HIST)
col("param_dist", "weights", "Relative parameter movement.", formula="||theta_t - theta_0|| / ||theta_0||",
    source=HIST)
col("log_weight_norm_ratio", "weights", "Log of the weight norm over its value at step 0. The kernel of a "
    "2-homogeneous network rescaled by c has norm c^2 times larger, so twice this value is the scale term a "
    "uniform rescaling of the weights would give.", formula="log(||theta_t|| / ||theta_0||)", source=OWN)
for layer in ("W1", "W2"):
    col(f"{layer}_norm", "weights", f"Frobenius norm of {layer}.", source=OWN)
    col(f"{layer}_dist", "weights", f"Relative movement of {layer}.", formula=f"||{layer}_t - {layer}_0|| / "
        f"||{layer}_0||", source=OWN)
    col(f"{layer}_stable_rank", "weights", f"Stable rank of {layer}.", formula="||W||_F^2 / sigma_max^2",
        source=OWN)
    col(f"{layer}_eff_rank", "weights", f"Effective rank of {layer}, the exponential of the entropy of its "
        "singular values normalised to sum to one.", formula="exp(-sum q log q), q = sigma / sum sigma",
        source=OWN)
col("layer_imbalance", "weights", "Difference of the squared layer norms over the squared total norm. Gradient "
    "flow without decay keeps ||W1||^2 - ||W2||^2 fixed for this bias-free ReLU network. With decay it shrinks "
    "by (1 - eta_kappa)^2 per step, so it mostly records decay.", formula="(||W1||^2 - ||W2||^2) / ||theta||^2",
    source=OWN)
col("dead_frac", "weights", "Fraction of hidden units that are inactive on every training input.", source=OWN)
col("active_frac", "weights", "Mean over training inputs of the fraction of hidden units that are active.",
    source=OWN)

for src in FOURIER_SOURCES:
    what = "the weights" if src == "W" else "the change of the weights since step 0"
    for mat in FOURIER_MATS:
        part = {"a": "the a-half of W1 along the token axis", "b": "the b-half of W1 along the token axis",
                "c": "W2 along the class axis"}[mat]
        col(f"fourier_{src}_{mat}_topshare", "fourier", f"Mean over hidden units, weighted by the unit's norm, of "
            f"the share of nonzero-frequency power at the unit's largest frequency, for {part}, in {what}. "
            "Frequencies k and p - k are folded together.", formula="sum_i w_i max_k P_ik / sum_k P_ik / sum_i w_i",
            source=GROMOV)
        col(f"fourier_{src}_{mat}_neff", "fourier", f"Effective number of nonzero frequencies in the pooled power "
            f"of {part}, in {what}.", formula="exp(entropy of sum_i P_ik / sum_ik P_ik)", source=GROMOV)
        col(f"fourier_{src}_{mat}_dcshare", "fourier", f"Share of the power at frequency zero for {part}, in "
            f"{what}.", formula="sum_i P_i0 / sum_ik P_ik over all k", source=OWN)
    col(f"fourier_{src}_match", "fourier", f"Weighted fraction of hidden units whose largest frequency is the "
        f"same for a, b and the output class, in {what}.", source=GROMOV)
    col(f"fourier_{src}_phase", "fourier", f"Weighted mean of cos(phi_a + phi_b - phi_c) over the matched units, "
        f"with the phases of the DFT at the shared frequency, in {what}. Equation 12 of Gromov (2023) makes this "
        "one for the exact solution. NaN if no unit matches.", source=GROMOV)

col("S_t", "kernel_raw", "Scale term on the raw probe kernel.", formula="log(||K_t||_F / ||K_0||_F)", source=HIST)
col("R_t", "kernel_raw", "Rotation term on the raw probe kernel.", formula="1 - <K_t / ||K_t||, K_0 / ||K_0||>",
    source=HIST)
col("K_norm", "kernel_raw", "Frobenius norm of the raw probe kernel. K is the kernel of f, not of the rescaled "
    "predictor. With lr = eta_0 / alpha^2 the step in function space is eta_0 K at every alpha, so this can be "
    "compared across alpha.", source=HIST)
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
            f"restricted to those pairs.", source=REPORT + "; " + HIST)
col("K_norm_centred", "kernel_centred", "Frobenius norm of the centred probe kernel.", formula="||Kc_t||_F",
    source=CMR + "; ntk_trace.Tracer.measure")
col("inner_K0", "kernel_centred", "Inner product of the centred kernel with the centred kernel at step 0.",
    formula="<Kc_t, Kc_0>", source="ntk_trace.Tracer.measure")
col("inner_G", "kernel_centred", "Inner product of the centred kernel with the centred target Gram matrix.",
    formula="<Kc_t, G>", source="ntk_trace.Tracer.measure")
col("D", "kernel_centred", "The usual kernel movement statistic on the centred kernel.",
    formula="||Kc_t - Kc_0||_F / ||Kc_0||_F", source="docs/report.tex, Equation eq:decomp")
col("exp2S_term", "kernel_centred", "First piece of Equation eq:decomp. D^2 = exp2S_term + 1 - cross_term.",
    formula="e^(2 S_c)", source="docs/report.tex, Equation eq:decomp, the group's own algebra")
col("cross_term", "kernel_centred", "Last piece of Equation eq:decomp.", formula="2 e^S_c (1 - R_c)",
    source="docs/report.tex, Equation eq:decomp, the group's own algebra")
col("gamma", "kernel_centred", "The aim: the cosine between the part of k_t orthogonal to k_0 and the part of "
    "the unit target orthogonal to k_0. NaN at step 0.", formula="<v_t, g_perp> / ||v_t||, v_t = k_t - <k_t, k_0> k_0",
    source="term_dependence.py docstring, the group's own algebra; ntk_trace.Tracer.measure")
col("share_k0", "kernel_centred", "Share of the squared change of the unit kernel along k_0. It equals R_c / 2.",
    formula="<d_t, k_0>^2 / ||d_t||^2, d_t = k_t - k_0", source=PLAN + ", E3; ntk_trace.Tracer.measure")
col("share_gperp", "kernel_centred", "Share of the squared change of the unit kernel along g_perp.",
    formula="<d_t, g_perp>^2 / ||d_t||^2", source=PLAN + ", E3; ntk_trace.Tracer.measure")
col("share_residual", "kernel_centred", "Share of the squared change of the unit kernel orthogonal to both k_0 "
    "and g_perp. The three shares sum to one.", source=PLAN + ", E3; ntk_trace.Tracer.measure")
col("kernel_degenerate", "kernel_centred", f"True if the centred probe kernel norm is below {DEGENERATE}. The "
    "unit-kernel columns are then NaN.", source=OWN)

for layer in ("W1", "W2"):
    col(f"S_{layer}", "layer_kernels", f"Scale term of the centred kernel term from the gradient of {layer} alone.",
        formula=f"log(||Kc_{layer},t|| / ||Kc_{layer},0||)", source="ntk_lib.entk_closed_form, " + OWN)
    col(f"R_{layer}", "layer_kernels", f"Rotation term of the centred kernel term from {layer} alone.",
        source="ntk_lib.entk_closed_form, " + OWN)
    col(f"A_{layer}", "layer_kernels", f"Centred alignment of the kernel term from {layer} alone.",
        source="ntk_lib.entk_closed_form, " + OWN)
col("W1_kernel_share", "layer_kernels", "Share of the trace of the centred probe kernel that comes from W1.",
    formula="tr(Kc_W1) / (tr(Kc_W1) + tr(Kc_W2))", source="ntk_lib.entk_closed_form, " + OWN)
for term, word in (("S", "scale term"), ("R", "rotation term"), ("A", "centred alignment")):
    col(f"{term}_first_logit", "first_logit", f"The {word} of the centred kernel of the first output alone. The "
        "report names this kernel as a robustness check.", source=REPORT + "; ntk_lib.entk_closed_form")

GOLDEN = "docs/report_golden.pdf, Equations 5 and 6"
for term, word in (("S", "Scale, the plain ratio of Frobenius norms to step 0"),
                   ("R", "Shape change, one minus the uncentred cosine to step 0"),
                   ("D", "Variation, the Frobenius norm of the change from step 0 over the norm at step 0"),
                   ("A", "Centred alignment with Y Y^T")):
    col(f"{term}_sum", "report_kernel", f"{word}, of the kernel of the sum of the logits divided by sqrt(p), "
        "on the test pairs of the run. These are the S_t, R_t, D_t and A_t of the golden report.",
        source=GOLDEN + "; ntk_lib.sum_kernel")

col("eig_trace", "spectrum", "Trace of the centred probe kernel.", source=OWN)
col("eig_top1", "spectrum", "Largest eigenvalue of the centred probe kernel.", source=OWN)
col("eig_top1_share", "spectrum", "Largest eigenvalue over the trace.", source=OWN)
col("eig_eff_rank", "spectrum", "Exponential of the entropy of the eigenvalues normalised to sum to one. The "
    "zero eigenvalue that centring creates is left out, and so is every column below.", source=OWN)
col("eig_participation", "spectrum", "Participation ratio of the eigenvalues.",
    formula="(sum lambda)^2 / sum lambda^2", source=OWN)
col("eig_n90", "spectrum", "Number of the largest eigenvalues that hold 90 percent of the trace.", source=OWN)
col("eig_gap_ab", "spectrum", "Ratio of eigenvalue 2(p - 1) to eigenvalue 2(p - 1) + 1, counting from one. In "
    "the tier x traces the functions of a alone and of b alone filled the top 2(p - 1) places.",
    source=PLAN + ", E4")
col("eig_gap_top88", "spectrum", "Ratio of eigenvalue 4(p - 1) to eigenvalue 4(p - 1) + 1, the edge of the "
    "eigenvectors that the tracking columns follow.", source=PLAN + ", E4")
for s, word in (("sum", "(a + b) mod p"), ("diff", "(a - b) mod p"), ("a", "a alone"), ("b", "b alone")):
    col(f"probe_energy_{s}", "subspaces", f"Share of the trace of the centred probe kernel inside the centred "
        f"functions of {word} on the probe. The subspaces are not orthogonal on the probe, so the four shares need "
        "not sum to one." + (" This is the label subspace." if s == "sum" else ""),
        formula="tr(Q_s^T Kc Q_s) / tr(Kc)", source=PLAN + ", E3")
for m in (22, 44, 88):
    col(f"label_capture_{m}", "eigvec_tracking", f"Share of the centred label matrix inside the top {m} "
        "eigenvectors of the centred probe kernel.", formula=f"||V_{m}^T H Y||_F^2 / ||H Y||_F^2", source=OWN)
col("topk_drift", "eigvec_tracking", "Mean squared sine of the principal angles between the top 4(p - 1) "
    "eigenvectors and the same span at step 0. At large width the cut falls in a nearly degenerate bulk, so this "
    "is noisy there.", source=PLAN + ", E4; ntk_trace.Tracer.measure")
col("overlap_zero_median", "eigvec_tracking", "Median over the tracked top eigenvectors of the absolute overlap "
    "with the step 0 vector their chain of matches leads back to. Noisy at large width for the same reason.",
    source=PLAN + ", E4; ntk_trace.Tracer.measure")

col("S_full", "full_grid", "Scale term of the centred kernel on all p^2 pairs.", source="ntk_lib.entk_closed_form")
col("R_full", "full_grid", "Rotation term of the centred kernel on all p^2 pairs.", source="ntk_lib.entk_closed_form")
col("A_full", "full_grid", "Centred alignment on all p^2 pairs, with the labels of every pair.",
    source="ntk_lib.entk_closed_form")
for s, word in (("sum", "(a + b) mod p"), ("diff", "(a - b) mod p"), ("a", "a alone"), ("b", "b alone")):
    col(f"full_energy_{s}", "full_grid", f"Share of the trace of the centred full-grid kernel inside the centred "
        f"functions of {word}. On the full grid the four subspaces are orthogonal.",
        formula="tr(Q_s^T Kc Q_s) / tr(Kc)", source=OWN)
col("full_energy_rest", "full_grid", "Share of the trace of the centred full-grid kernel outside the four "
    "subspaces.", formula="1 - sum of the four shares", source=OWN)
for where, word in (("full", "the real training pairs to the real test pairs"),
                    ("probe", "the training part of the probe to its test part")):
    col(f"krr_{where}_test_acc", f"{where}_krr" if where == "probe" else "full_grid",
        f"Accuracy of kernel ridge regression with the current kernel from {word}, starting from a zero function "
        f"as the predictor does. The ridge is {KRR_RIDGE} times the mean diagonal of the training block.",
        formula="argmax K_te,tr (K_tr,tr + r I)^-1 Y_tr", source=OWN)
    col(f"krr_{where}_test_mse", f"{where}_krr" if where == "probe" else "full_grid",
        f"Mean squared error of the same kernel ridge regression on {word.split(' to ')[1]}.", source=OWN)

col("dS_step", "instant_rates", "Change of the centred scale term over the next gradient-descent step, from a "
    "virtual step with the current gradient and decay, in float64.", formula="S(K_{t+1}) - S(K_t)", source=OWN)
col("dA_step", "instant_rates", "Change of the centred alignment over the next step, from the same virtual step.",
    formula="A(K_{t+1}) - A(K_t)", source=OWN)
col("R_step", "instant_rates", "Rotation of the centred unit kernel over the next step, from the same virtual "
    "step.", formula="1 - <k_{t+1}, k_t>", source=OWN)

COLUMN_NAMES = [c["name"] for c in COLUMNS]
HISTORY_COPIED = ["train_loss", "test_loss", "train_acc", "test_acc", "S_t", "R_t", "K_norm", "yKy", "S_c", "R_c",
                  "A_t", "A_u", "weight_norm", "param_dist"] + L.PART_KEYS


# =====================================================================
# Helpers
# =====================================================================
def one_hot_pairs(a, b, p):
    """Return the concatenated one-hot inputs of the pairs (a, b) as a float32 tensor."""
    X = torch.zeros((len(a), 2 * p), dtype=torch.float32)
    X[torch.arange(len(a)), torch.as_tensor(a)] = 1.0
    X[torch.arange(len(a)), p + torch.as_tensor(b)] = 1.0
    return X


def centre(K):
    """Return H K H for a float64 numpy matrix, Cortes et al. (2012), Lemma 1."""
    return T.kernels.centre(K)


def unit_terms(Kc, Kc0_unit, norm0, G_unit):
    """Return S, R and A of a centred kernel against a unit reference and a unit target, or NaNs if it is zero."""
    norm = np.linalg.norm(Kc)
    if norm < DEGENERATE:
        return np.nan, np.nan, np.nan
    k = Kc / norm
    return float(np.log(norm / norm0)), float(1.0 - np.sum(k * Kc0_unit)), float(np.sum(k * G_unit))


def subspace_energy(Kc, bases):
    """Return tr(Q^T Kc Q) / tr(Kc) for each orthonormal basis Q."""
    tr = np.trace(Kc)
    return [float(np.trace(Q.T @ Kc @ Q) / tr) if tr > 0 else np.nan for Q in bases]


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


def fourier_power(M, p):
    """Return the folded power (units, (p - 1) / 2 + 1) and the phases (units, (p - 1) / 2 + 1) of the rows of M.

    Column 0 is frequency zero. For k from 1 to (p - 1) / 2 the power of k and
    p - k are added, which for a real row is twice |F_k|^2. The phase is the
    angle of F_k, so a row cos(2 pi k j / p + phi) has phase phi at k.
    """
    F = np.fft.fft(M, axis=1)[:, : (p - 1) // 2 + 1]
    P = np.abs(F) ** 2
    P[:, 1:] *= 2.0
    return P, np.angle(F)


def fourier_features(W1, W2, p):
    """The Fourier columns of one source: a dict keyed as in COLUMNS without the source prefix.

    W1 has shape (N, 2p) and W2 shape (p, N). The rows of the three
    matrices are the a-half of W1, the b-half of W1, and the columns of W2.
    Each unit is weighted by ||W1 row|| ||W2 column||.
    """
    mats = dict(a=W1[:, :p], b=W1[:, p:], c=W2.T)
    weight = np.linalg.norm(W1, axis=1) * np.linalg.norm(W2, axis=0)
    out, dominant, phases = {}, {}, {}
    for name, M in mats.items():
        P, phi = fourier_power(M, p)
        nonzero = P[:, 1:]
        per_unit = nonzero.sum(1)
        ok = per_unit > 0
        w = weight * ok
        top = np.where(ok, nonzero.max(1) / np.where(ok, per_unit, 1.0), 0.0)
        out[f"{name}_topshare"] = float(np.sum(w * top) / np.sum(w)) if np.sum(w) > 0 else np.nan
        out[f"{name}_neff"] = effective_rank(nonzero.sum(0))
        total = P.sum()
        out[f"{name}_dcshare"] = float(P[:, 0].sum() / total) if total > 0 else np.nan
        dominant[name] = nonzero.argmax(1) + 1
        phases[name] = phi
    match = (dominant["a"] == dominant["b"]) & (dominant["b"] == dominant["c"]) & (weight > 0)
    total_w = weight.sum()
    out["match"] = float(weight[match].sum() / total_w) if total_w > 0 else np.nan
    if match.any():
        idx = np.nonzero(match)[0]
        k = dominant["a"][idx]
        coherence = np.cos(phases["a"][idx, k] + phases["b"][idx, k] - phases["c"][idx, k])
        out["phase"] = float(np.sum(weight[idx] * coherence) / weight[idx].sum())
    else:
        out["phase"] = np.nan
    return out


# =====================================================================
# The logger
# =====================================================================
class FeatureLogger:
    """The on_checkpoint callback that builds one feature row per checkpoint.

    Construct it with the full configuration of the run, as
    ntk_lib.normalise_checkpoints returns it. train_run calls it once per
    checkpoint, starting at step 0 and in step order. After the run, rows
    holds the rows, weights holds the saved weight matrices keyed by step, and
    check_values holds the terms that ntk_trace.checks compares.

    The callback must not change the model. Every gradient here is taken with
    torch.autograd.grad, which does not touch .grad, and the power iteration
    uses its own random generator, so the global random state is untouched.
    """

    def __init__(self, cfg, run_id, save_steps=(), linear_interval=1000, log_steps=()):
        if cfg["parameterisation"] not in ("ntk", "mean_field") or cfg["activation"] != "relu":
            raise ValueError("the feature logger needs a bias-free ReLU model")
        p = cfg["p"]
        self.cfg, self.run_id, self.p = cfg, run_id, p
        self.alpha = cfg["alpha"]
        self.lr = cfg["eta_0"] / self.alpha ** 2
        self.wd = cfg["eta_kappa"] / self.lr
        self.save_steps = set(int(s) for s in save_steps)
        self.linear_interval, self.log_steps = linear_interval, set(int(s) for s in log_steps)
        self.ids = dict(run_id=run_id, alpha=self.alpha, width=cfg["hidden_dim"], eta_kappa=cfg["eta_kappa"],
                        seed=cfg["seed"])

        (X_train, y_train), (X_test, y_test) = make_modular_addition_dataset(
            p=p, train_fraction=cfg["train_fraction"], seed=cfg["data_seed"])
        self.X = dict(train=X_train, test=X_test)
        self.y = dict(train=y_train, test=y_test)
        self.labels = dict(train=y_train.argmax(1), test=y_test.argmax(1))
        self.X_test64, self.Y_test64 = X_test.double().numpy(), y_test.double().numpy()
        self.K_sum_0 = None

        # The probe, exactly as train_run takes it, and the tier x tracer on it.
        self.x_probe, y_probe, is_test = L.select_probe(X_train, y_train, X_test, y_test, cfg["probe"],
                                                        cfg["probe_size"])
        self.tracer = T.Tracer(cfg)
        self.Y_probe = y_probe.numpy().astype(np.float64)
        self.probe_train = np.nonzero(~is_test.numpy())[0]
        self.probe_test = np.nonzero(is_test.numpy())[0]
        n = len(self.Y_probe)
        H = np.eye(n) - 1.0 / n
        self.HY = H @ self.Y_probe
        G = centre(self.Y_probe @ self.Y_probe.T)
        self.G_unit = G / np.linalg.norm(G)

        # The full grid of p^2 pairs in the order a * p + b, with the real split.
        a_all, b_all = np.divmod(np.arange(p * p), p)
        self.X_all = one_hot_pairs(a_all, b_all, p)
        self.Y_all = np.eye(p)[(a_all + b_all) % p]
        train_idx = (X_train[:, :p].argmax(1) * p + X_train[:, p:].argmax(1)).numpy()
        is_train = np.zeros(p * p, dtype=bool)
        is_train[train_idx] = True
        self.full_train, self.full_test = np.nonzero(is_train)[0], np.nonzero(~is_train)[0]
        G_full = centre(self.Y_all @ self.Y_all.T)
        self.G_full_unit = G_full / np.linalg.norm(G_full)
        H_full = np.eye(p * p) - 1.0 / (p * p)
        codes = [(a_all + b_all) % p, (a_all - b_all) % p, a_all, b_all]
        self.full_bases = [T.orthonormal_basis(H_full @ np.eye(p)[c]) for c in codes]

        self.model_0 = None
        self.rows, self.weights = [], {}
        self.check_values = {k: [] for k in ("step", "S", "R", "A", "A_uncentred", "D", "exp2S_term",
                                             "cross_term", "gamma", "share_k0", "share_gperp",
                                             "share_residual", "S_c", "R_c", "A_t", "A_u")}
        self.last_step, self.last_weights = None, None
        self.sharp_vec = None
        self.rng = torch.Generator().manual_seed(12345)
        self.t_start = time.time()
        self.callback_seconds = 0.0

    # -----------------------------------------------------------------
    def reference(self, name, Kc):
        """Keep the unit centred kernel and the norm at step 0 under a name, and return them."""
        if not hasattr(self, "refs"):
            self.refs = {}
        if name not in self.refs:
            norm = np.linalg.norm(Kc)
            self.refs[name] = (Kc / norm, norm)
        return self.refs[name]

    def terms(self, name, Kc, G_unit):
        """S, R and A of a centred kernel against its step 0 reference under name."""
        k0, norm0 = self.reference(name, Kc)
        return unit_terms(Kc, k0, norm0, G_unit)

    # -----------------------------------------------------------------
    def __call__(self, step, model, K_t, row):
        t_call = time.time()
        if self.model_0 is None:
            if step != 0:
                raise ValueError(f"the first checkpoint must be step 0, not {step}")
            self.model_0 = copy.deepcopy(model)
            for q in self.model_0.parameters():
                q.requires_grad = False
            with torch.no_grad():
                self.f0 = {s: self.model_0(self.X[s]) for s in SPLITS}
            self.W1_0 = model.W1.detach().double().clone()
            self.W2_0 = model.W2.detach().double().clone()
        out = dict(self.ids)
        out.update(step=int(step), log10_step1=math.log10(step + 1),
                   dt_prev=np.nan if self.last_step is None else float(step - self.last_step),
                   on_linear_grid=bool(step % self.linear_interval == 0), on_log_grid=bool(step in self.log_steps),
                   wall_seconds=t_call - self.t_start)
        out.update({k: row[k] for k in HISTORY_COPIED})
        out.update(self.performance(model))
        grads = self.optimisation(model, out)
        out.update(self.weight_features(model))
        out.update(self.kernel_features(step, K_t, row, model, grads))
        missing = set(COLUMN_NAMES) - set(out)
        extra = set(out) - set(COLUMN_NAMES)
        if missing or extra:
            raise KeyError(f"feature row does not match COLUMNS: missing {sorted(missing)}, extra {sorted(extra)}")
        self.rows.append({name: out[name] for name in COLUMN_NAMES})

        W1 = model.W1.detach().cpu().numpy().astype(np.float32).copy()
        W2 = model.W2.detach().cpu().numpy().astype(np.float32).copy()
        if step in self.save_steps:
            self.weights[int(step)] = (W1, W2)
        self.last_step, self.last_weights = int(step), (W1, W2)
        self.callback_seconds += time.time() - t_call

    # -----------------------------------------------------------------
    def performance(self, model):
        out = {}
        with torch.no_grad():
            for s in SPLITS:
                f = (self.alpha * (model(self.X[s]) - self.f0[s])).double()
                lab = self.labels[s]
                idx = torch.arange(len(lab))
                correct = f[idx, lab]
                others = f.clone()
                others[idx, lab] = -torch.inf
                margin = correct - others.max(1).values
                wrong = f.clone()
                wrong[idx, lab] = 0.0
                p = f.shape[1]
                out[f"{s}_margin_mean"] = margin.mean().item()
                out[f"{s}_margin_min"] = margin.min().item()
                out[f"{s}_correct_out_mean"] = correct.mean().item()
                out[f"{s}_wrong_out_rms"] = math.sqrt((wrong ** 2).sum().item() / (len(lab) * (p - 1)))
                out[f"{s}_out_rms"] = math.sqrt((f ** 2).mean().item())
                out[f"{s}_out_max_mean"] = f.max(1).values.mean().item()
        return out

    def optimisation(self, model, out):
        """Gradient columns and sharpness. Returns the loss gradient (g1, g2) in float32 for the virtual step."""
        params = [model.W1, model.W2]
        f = self.alpha * (model(self.X["train"]) - self.f0["train"])
        loss = nn.functional.mse_loss(f, self.y["train"])
        grads = torch.autograd.grad(loss, params, create_graph=True)
        g = torch.cat([q.reshape(-1) for q in grads]).detach().double()
        theta = torch.cat([q.detach().reshape(-1) for q in params]).double()
        gn, tn = torch.linalg.norm(g).item(), torch.linalg.norm(theta).item()
        pull = self.wd * tn
        out["grad_norm"] = gn
        out["grad_norm_W1"] = torch.linalg.norm(grads[0].detach().double()).item()
        out["grad_norm_W2"] = torch.linalg.norm(grads[1].detach().double()).item()
        out["update_norm"] = self.lr * torch.linalg.norm(g + self.wd * theta).item()
        out["grad_weight_cos"] = (g @ theta).item() / (gn * tn) if gn > 0 and tn > 0 else np.nan
        out["grad_decay_ratio"] = gn / pull if pull > 0 else np.nan
        denom = gn + pull
        out["decay_residual"] = torch.linalg.norm(g + self.wd * theta).item() / denom if denom > 0 else np.nan
        out["stationary"] = bool(self.wd > 0 and out["decay_residual"] < STATIONARY)

        # Power iteration on the Hessian of the loss, warm started.
        if self.sharp_vec is None:
            v = [torch.randn(q.shape, generator=self.rng) for q in params]
        else:
            v = self.sharp_vec
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
        out["lr_sharpness"] = self.lr * lam
        return grads[0].detach(), grads[1].detach()

    def weight_features(self, model):
        out = {}
        W1, W2 = model.W1.detach().double(), model.W2.detach().double()
        n1, n2 = torch.linalg.norm(W1).item(), torch.linalg.norm(W2).item()
        n0 = math.sqrt(torch.linalg.norm(self.W1_0).item() ** 2 + torch.linalg.norm(self.W2_0).item() ** 2)
        total2 = n1 ** 2 + n2 ** 2
        out["log_weight_norm_ratio"] = math.log(math.sqrt(total2) / n0) if total2 > 0 else np.nan
        for name, W, W0, n in (("W1", W1, self.W1_0, n1), ("W2", W2, self.W2_0, n2)):
            sv = torch.linalg.svdvals(W).numpy()
            out[f"{name}_norm"] = n
            out[f"{name}_dist"] = torch.linalg.norm(W - W0).item() / torch.linalg.norm(W0).item()
            out[f"{name}_stable_rank"] = float(n ** 2 / sv[0] ** 2) if sv[0] > 0 else np.nan
            out[f"{name}_eff_rank"] = effective_rank(sv)
        out["layer_imbalance"] = (n1 ** 2 - n2 ** 2) / total2 if total2 > 0 else np.nan
        c1, _ = L.closed_form_scales(model)
        with torch.no_grad():
            active = (c1 * (self.X["train"].double() @ W1.T)) > 0
        out["dead_frac"] = (~active.any(0)).double().mean().item()
        out["active_frac"] = active.double().mean().item()
        W1n, W2n = W1.numpy(), W2.numpy()
        for src, (A, B) in (("W", (W1n, W2n)), ("dW", (W1n - self.W1_0.numpy(), W2n - self.W2_0.numpy()))):
            for k, v in fourier_features(A, B, self.p).items():
                out[f"fourier_{src}_{k}"] = v
        return out

    def kernel_features(self, step, K_t, row, model, grads):
        out = {}
        K = K_t.detach().cpu().numpy().astype(np.float32)
        K64 = K.astype(np.float64)
        Kc = centre(K64)
        norm = np.linalg.norm(Kc)
        degenerate = norm < DEGENERATE
        out["kernel_degenerate"] = bool(degenerate)
        if degenerate and step == 0:
            raise ValueError("the centred probe kernel is zero at step 0")

        names = ["K_norm_centred", "inner_K0", "inner_G", "D", "exp2S_term", "cross_term", "gamma", "share_k0",
                 "share_gperp", "share_residual"]
        if not degenerate:
            m = self.tracer.measure(step, K)
            for k in names:
                out[k] = float(m[k])
            for k in ("S", "R", "A", "A_uncentred", "D", "exp2S_term", "cross_term", "gamma", "share_k0",
                      "share_gperp", "share_residual"):
                self.check_values[k].append(float(m[k]))
            eig = np.asarray(m["eigvals"], dtype=np.float64)[:-1]
            V = np.asarray(m["eigvecs_topk"], dtype=np.float64)
            angles_zero = np.asarray(m["principal_angles_zero"])
            overlap_zero = np.asarray(m["overlap_zero"])
        else:
            for k in names:
                out[k] = np.nan
            for k in ("S", "R", "A", "A_uncentred", "D", "exp2S_term", "cross_term", "gamma", "share_k0",
                      "share_gperp", "share_residual"):
                self.check_values[k].append(np.nan)
            eig = V = angles_zero = overlap_zero = None
        self.check_values["step"].append(int(step))
        for k in ("S_c", "R_c", "A_t", "A_u"):
            self.check_values[k].append(float(row[k]))

        # Spectrum of the centred kernel without the zero that centring creates.
        p = self.p
        if eig is not None:
            pos = np.clip(eig, 0.0, None)
            tr = pos.sum()
            out["eig_trace"] = float(tr)
            out["eig_top1"] = float(eig[0])
            out["eig_top1_share"] = float(eig[0] / tr) if tr > 0 else np.nan
            out["eig_eff_rank"] = effective_rank(pos)
            out["eig_participation"] = float(tr ** 2 / np.sum(pos ** 2)) if tr > 0 else np.nan
            out["eig_n90"] = int(np.searchsorted(np.cumsum(pos) / tr, 0.9) + 1) if tr > 0 else np.nan
            i_ab, i_88 = 2 * (p - 1), 4 * (p - 1)
            out["eig_gap_ab"] = float(eig[i_ab - 1] / eig[i_ab]) if eig[i_ab] > 0 else np.nan
            out["eig_gap_top88"] = float(eig[i_88 - 1] / eig[i_88]) if eig[i_88] > 0 else np.nan
            hy2 = np.sum(self.HY ** 2)
            for mm in (22, 44, 88):
                out[f"label_capture_{mm}"] = float(np.sum((V[:, :mm].T @ self.HY) ** 2) / hy2)
            out["topk_drift"] = float(np.mean(np.sin(angles_zero) ** 2))
            out["overlap_zero_median"] = float(np.nanmedian(overlap_zero)) if np.isfinite(overlap_zero).any() \
                else np.nan
        else:
            for k in ("eig_trace", "eig_top1", "eig_top1_share", "eig_eff_rank", "eig_participation", "eig_n90",
                      "eig_gap_ab", "eig_gap_top88", "label_capture_22", "label_capture_44", "label_capture_88",
                      "topk_drift", "overlap_zero_median"):
                out[k] = np.nan
        for s, e in zip(("sum", "diff", "a", "b"), subspace_energy(Kc, self.tracer.subspace_bases)):
            out[f"probe_energy_{s}"] = e

        # The two layer terms and the first-logit kernel, in float64.
        c1, c2 = L.closed_form_scales(model)
        K_W1, K_W2 = L.closed_form_terms(model.W1, model.W2, self.x_probe, c1, c2)
        Kc_W1, Kc_W2 = centre(K_W1.numpy()), centre(K_W2.numpy())
        for name, M in (("W1", Kc_W1), ("W2", Kc_W2)):
            out[f"S_{name}"], out[f"R_{name}"], out[f"A_{name}"] = self.terms(name, M, self.G_unit)
        t1, t2 = np.trace(Kc_W1), np.trace(Kc_W2)
        out["W1_kernel_share"] = float(t1 / (t1 + t2)) if t1 + t2 > 0 else np.nan
        F1, F2 = L.closed_form_terms(model.W1, model.W2, self.x_probe, c1, c2, output=0)
        out["S_first_logit"], out["R_first_logit"], out["A_first_logit"] = self.terms(
            "first_logit", centre((F1 + F2).numpy()), self.G_unit)

        # The kernel of the golden report, Equations 5 and 6, on the test pairs.
        K_sum = L.sum_kernel(model.W1.detach().double().cpu().numpy(), model.W2.detach().double().cpu().numpy(),
                             self.X_test64)
        if self.K_sum_0 is None:
            self.K_sum_0 = K_sum
        for k, v in L.report_statistics(K_sum, self.K_sum_0, self.Y_test64).items():
            out[f"{k[0]}_sum"] = float(v)

        # The full grid of p^2 pairs.
        A1, A2 = L.closed_form_terms(model.W1, model.W2, self.X_all, c1, c2)
        K_full = (A1 + A2).numpy()
        Kc_full = centre(K_full)
        out["S_full"], out["R_full"], out["A_full"] = self.terms("full", Kc_full, self.G_full_unit)
        energies = subspace_energy(Kc_full, self.full_bases)
        for s, e in zip(("sum", "diff", "a", "b"), energies):
            out[f"full_energy_{s}"] = e
        out["full_energy_rest"] = float(1.0 - np.sum(energies))
        out["krr_full_test_acc"], out["krr_full_test_mse"] = krr(K_full, self.Y_all, self.full_train,
                                                                 self.full_test)
        out["krr_probe_test_acc"], out["krr_probe_test_mse"] = krr(K64, self.Y_probe, self.probe_train,
                                                                   self.probe_test)

        # One virtual step of gradient descent with decay, as torch.optim.SGD takes it.
        g1, g2 = grads
        with torch.no_grad():
            W1n = model.W1.detach() - self.lr * (g1 + self.wd * model.W1.detach())
            W2n = model.W2.detach() - self.lr * (g2 + self.wd * model.W2.detach())
        B1, B2 = L.closed_form_terms(W1n, W2n, self.x_probe, c1, c2)
        Kc_now = Kc_W1 + Kc_W2
        Kc_next = centre((B1 + B2).numpy())
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
        """The terms that ntk_trace.checks compares, as arrays, with A_0 from the tracer."""
        a = {k: np.asarray(v, dtype=float) for k, v in self.check_values.items()}
        a["A_0"] = np.array(getattr(self.tracer, "A_0", np.nan))
        return a


def identity_checks(a):
    """Largest differences that should sit at rounding level, as ntk_trace.checks computes them.

    a is the output of FeatureLogger.check_arrays. The history fields S_c, R_c,
    A_t and A_u are the float32 values train_run records, so their differences
    from the float64 tracer terms sit at float32 rounding.
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
        "A_uncentred against the history field A_u": worst(a["A_uncentred"] - a["A_u"]),
        "D^2 against exp2S_term + 1 - cross_term": worst(a["D"] ** 2 - (a["exp2S_term"] + 1 - a["cross_term"])),
        "the three shares against a sum of one": worst(a["share_k0"] + a["share_gperp"] + a["share_residual"] - 1),
        "share_k0 against R / 2": worst(a["share_k0"] - R / 2),
        "A against the aim identity": worst((a["A"] - identity)[moved]),
    }
