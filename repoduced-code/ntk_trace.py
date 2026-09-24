"""Trace a run at every checkpoint, for the tier x study and the animations.

The tier x plan in docs/zain_tierx_plan.md asks whether the scale, rotation
and alignment terms carry separate information, and where the kernel goes
when it turns. The saved runs keep the history and, for the dense runs, the
kernel every 250 steps. This module keeps what a later analysis or scene
could need at every checkpoint. That is the kernel, the weights, the
predictor on the probe set, every intermediate of the three terms, the three
shares of the kernel change, the top eigenvectors matched across checkpoints,
and torus averages. Section 6 of the plan lists the arrays. The schema file
written next to each trace repeats the list with the formula and the source
of every array.

The module does not fork the training loop. It builds the checkpoint grid,
passes a Tracer to ntk_lib.train_run as its on_checkpoint callback, and
writes the trace when the run ends.

Checkpoints. The grid is the union of step 0, every 250 steps, and about 60
log-spaced steps from 1 to the run length. The report's Kernel metrics
section asks for a log-spaced grid because grokking spans three to five
orders of magnitude in step. The every-250 part makes every checkpoint of the
dense runs and of the Tier 1 grid runs a checkpoint here too, so the
histories can be compared.

Names. A traced run is called trace_N{width}_a{alpha}_wd{eta kappa}_s{seed}.
It writes three files to results/. The run JSON is the one train_run writes.
The arrays go to name_trace.npz, which git ignores. The schema goes to
name_trace.json, which is committed. The dense runs of kernel_snapshots.py
keep their own names and files, because four scenes read their histories
row by row and expect a checkpoint every 250 steps.

Sources.
- Cortes, Mohri, and Rostamizadeh (2012), Journal of Machine Learning
  Research 13, pages 795 to 828. Lemma 1 gives the centred kernel H K H and
  Definition 4 the centred alignment.
- The decomposition of the movement statistic, Equation eq:decomp of the
  report, and the aim identity in the docstring of term_dependence.py are the
  group's own algebra.
- The three shares, the four subspaces and the eigenvector matching follow
  experiments E3 and E4 in Section 5 of the plan. They are definitions made
  for this study. No paper states them.

Run it from repoduced-code with the environment at the repository root.

    uv run python ntk_trace.py          # all seven cells
    uv run python ntk_trace.py 1        # the three cells with alpha 1
    uv run python ntk_trace.py 0.5 2    # the four new cells
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import kernel_snapshots as KS
import ntk_lib as L

HERE = Path(__file__).resolve().parent
# The numpy kernel helpers of the animations need no torch, so the tracer
# uses the same centring, target Gram matrix and torus average as the scenes.
sys.path.insert(0, str(HERE.parent / "animations"))
from common import kernels  # noqa: E402

WIDTH, SEED, BASE_RATE = 100, 0, 100.0
INTERVAL, N_LOG = 250, 60
# The number of top eigenvectors that are matched and compared. In the alpha 1
# traces of 24 September 2026, the leading 2(p - 1) = 44 eigenvectors of the
# centred kernel were functions of a alone and of b alone. The largest
# relative gap in the spectrum came right after them. During training the
# functions of the sum and of the difference rose out of the bulk to indices
# 44 to 87. So k = 4(p - 1) = 88 holds both. The plan first chose 32, which cut
# the first block in the middle.
TOP_K = 88
# An eigenvector whose assigned overlap with the previous checkpoint is below
# this value is treated as a new arrival.
NEW_ARRIVAL = 0.5
SUBSPACES = ("sum", "difference", "a", "b")

# The seven traced cells as (alpha, eta kappa, steps). Section 4 of the plan
# gives the reason for each one and for the longer alpha 2 runs.
CELLS = [
    (1.0, 0.0, 30000),
    (1.0, 3e-4, 30000),
    (1.0, 1e-3, 30000),
    (0.5, 0.0, 30000),
    (0.5, 3e-4, 30000),
    (2.0, 7.5e-5, 60000),
    (2.0, 1e-4, 60000),
]


def run_name(alpha, eta_kappa):
    """Name of a traced run. It is the grid cell name with the prefix trace."""
    return "trace" + L.cell_name(WIDTH, alpha, eta_kappa, SEED)[len("ntk"):]


def checkpoint_grid(steps, interval=INTERVAL, n_log=N_LOG):
    """Return step 0, every interval steps, and n_log log-spaced steps rounded to integers.

    Rounding merges some of the smallest log-spaced steps, so the grid holds
    slightly fewer than n_log of them.
    """
    log_steps = np.rint(np.logspace(0, np.log10(steps), n_log)).astype(int)
    return sorted({0, *range(0, steps + 1, interval), *log_steps.tolist()})


def run_config(alpha, eta_kappa, steps):
    """The configuration of a traced run, as keyword arguments for ntk_lib.train_run.

    It is the configuration ntk_lib.load_cell uses for the grid, apart from
    the run length and the checkpoints. The kernel file that train_run
    writes keeps only the first and the last kernel, because the trace holds
    all of them.
    """
    return dict(parameterisation="ntk", hidden_dim=WIDTH, alpha=alpha, eta_0=BASE_RATE,
                eta_kappa=eta_kappa, seed=SEED, steps=steps, eval_interval=INTERVAL,
                kernel_save_interval=steps, checkpoint_steps=checkpoint_grid(steps), probe="mixed")


# =====================================================================
# Linear algebra helpers
# =====================================================================
def inner(A, B):
    """Frobenius inner product of two matrices."""
    return float(np.sum(A * B))


def orthonormal_basis(M, rtol=1e-10):
    """Orthonormal basis of the column space of M, from its singular value decomposition."""
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > rtol * s[0]]


def principal_angles(Q1, Q2):
    """Principal angles in radians between the spans of two orthonormal bases, smallest first."""
    cosines = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def fix_sign(v):
    """Return v with its largest entry in absolute value made positive."""
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


# =====================================================================
# The tracer
# =====================================================================
class Tracer:
    """The on_checkpoint callback that collects the trace of one run.

    train_run calls it once per checkpoint, starting at step 0. After the run,
    arrays() returns the finished trace. Everything is computed in float64
    from the float32 kernel that train_run passes in, so the terms here agree
    with the history fields S_c, R_c and A_t to float32 rounding.
    """

    def __init__(self, cfg, k=TOP_K):
        p = cfg["p"]
        self.p, self.k, self.alpha = p, k, cfg["alpha"]
        self.probe_a, self.probe_b, self.probe_is_test = L.probe_pairs(cfg)
        n = len(self.probe_a)
        # The probe inputs are the one-hot codes of the probe pairs, the same
        # rows that train_run takes from the dataset.
        X = np.zeros((n, 2 * p), dtype=np.float32)
        X[np.arange(n), self.probe_a] = 1.0
        X[np.arange(n), p + self.probe_b] = 1.0
        self.x_probe = torch.from_numpy(X).to(L.DEVICE)

        labels = (self.probe_a + self.probe_b) % p
        Y = np.eye(p)[labels]
        H = np.eye(n) - 1.0 / n
        self.YY = Y @ Y.T
        self.G = kernels.target_gram(labels, p)
        self.G_norm = np.linalg.norm(self.G)
        self.label_basis = orthonormal_basis(H @ Y)
        # The centred functions of each of the four codes, in the order of SUBSPACES.
        codes = [labels, (self.probe_a - self.probe_b) % p, self.probe_a, self.probe_b]
        self.subspace_bases = [orthonormal_basis(H @ np.eye(p)[c]) for c in codes]
        self.torus = kernels.Torus(self.probe_a, self.probe_b, p)
        self.rows = []
        self.model_0 = None

    def start(self, step, model, Kc, norm):
        """Keep what every later checkpoint is measured against."""
        if step != 0:
            raise ValueError(f"the first checkpoint must be step 0, not {step}")
        self.model_0 = copy.deepcopy(model)
        self.K0c, self.norm_0 = Kc, norm
        self.k_0 = Kc / norm
        g = self.G / self.G_norm
        g_perp = g - inner(g, self.k_0) * self.k_0
        self.g_perp = g_perp / np.linalg.norm(g_perp)
        self.A_0 = inner(Kc, self.G) / (norm * self.G_norm)
        self.V_prev = self.V_0 = self.origin_prev = None

    def match(self, V):
        """Match the top eigenvectors to the previous checkpoint and fix their signs.

        The assignment maximises the sum of absolute overlaps. A vector whose
        assigned overlap is below NEW_ARRIVAL is a new arrival. It gets match
        index minus one and has its largest entry made positive. Every other
        vector gets the sign that makes its inner product with its predecessor
        positive.
        """
        k = V.shape[1]
        if self.V_prev is None:
            return np.stack([fix_sign(v) for v in V.T], axis=1), np.full(k, -1), np.full(k, np.nan)
        overlap = np.abs(V.T @ self.V_prev)
        rows, cols = linear_sum_assignment(-overlap)
        assigned = np.empty(k, dtype=int)
        assigned[rows] = cols
        overlap_prev = overlap[np.arange(k), assigned]
        match = np.where(overlap_prev >= NEW_ARRIVAL, assigned, -1)
        V = V.copy()
        for j in range(k):
            if match[j] < 0:
                V[:, j] = fix_sign(V[:, j])
            elif V[:, j] @ self.V_prev[:, match[j]] < 0:
                V[:, j] = -V[:, j]
        return V, match, overlap_prev

    def __call__(self, step, model, K_t, row):
        K = K_t.detach().cpu().numpy().astype(np.float32)
        K64 = K.astype(np.float64)
        Kc = kernels.centre(K64)
        norm = np.linalg.norm(Kc)
        if self.model_0 is None:
            self.start(step, model, Kc, norm)
        k_t, k_0, g_perp = Kc / norm, self.k_0, self.g_perp

        # The three terms, and the pieces of Equation eq:decomp on the centred kernel.
        S = np.log(norm / self.norm_0)
        R = 1.0 - inner(k_t, k_0)
        A = inner(Kc, self.G) / (norm * self.G_norm)

        # The change of the unit kernel, split along k_0, along g_perp, and the rest.
        change = k_t - k_0
        along_k0, along_g = inner(change, k_0), inner(change, g_perp)
        residual = change - along_k0 * k_0 - along_g * g_perp
        length2 = inner(change, change)
        if length2 > 0:
            shares = (along_k0 ** 2 / length2, along_g ** 2 / length2, inner(residual, residual) / length2)
            v = k_t - inner(k_t, k_0) * k_0
            gamma = inner(v, g_perp) / np.linalg.norm(v)
        else:
            shares, gamma = (np.nan, np.nan, np.nan), np.nan

        # Eigenvectors of the centred kernel, largest eigenvalue first.
        eigvals, U = np.linalg.eigh(Kc)
        eigvals, U = eigvals[::-1], U[:, ::-1]
        V, match, overlap_prev = self.match(U[:, : self.k])
        if self.V_0 is None:
            self.V_0, origin = V, np.arange(self.k)
        else:
            origin = np.where(match >= 0, self.origin_prev[np.maximum(match, 0)], -1)
        overlap_zero = np.full(self.k, np.nan)
        traced = origin >= 0
        overlap_zero[traced] = np.abs(np.sum(V[:, traced] * self.V_0[:, origin[traced]], axis=0))
        self.V_prev, self.origin_prev = V, origin

        with torch.no_grad():
            f_probe = self.alpha * (model(self.x_probe) - self.model_0(self.x_probe))

        self.rows.append(dict(
            step=step,
            # The parameters change in place during training, so they are copied.
            W1=model.W1.detach().cpu().numpy().copy(), W2=model.W2.detach().cpu().numpy().copy(),
            K=K, f_probe=f_probe.cpu().numpy().astype(np.float32),
            **{key: row[key] for key in ("train_loss", "test_loss", "train_acc", "test_acc",
                                         "param_dist", "weight_norm", "yKy")},
            K_norm_centred=norm, inner_K0=inner(Kc, self.K0c), inner_G=inner(Kc, self.G),
            S=S, R=R, A=A, A_uncentred=inner(K64, self.YY) / (np.linalg.norm(K64) * np.linalg.norm(self.YY)),
            D=np.linalg.norm(Kc - self.K0c) / self.norm_0,
            exp2S_term=np.exp(2 * S), cross_term=2 * np.exp(S) * (1 - R),
            gamma=gamma, share_k0=shares[0], share_gperp=shares[1], share_residual=shares[2],
            eigvals=eigvals, eigvecs_topk=V.astype(np.float32), match_index=match,
            overlap_prev=overlap_prev, origin_index=origin, overlap_zero=overlap_zero,
            principal_angles_zero=principal_angles(V, self.V_0),
            principal_angles_label=principal_angles(V, self.label_basis),
            subspace_energy=np.stack([np.sum((Q.T @ V) ** 2, axis=0) for Q in self.subspace_bases], axis=1),
            torus_K_unit=self.torus(k_t), torus_change=self.torus(change), torus_residual=self.torus(residual),
        ))

    def arrays(self):
        """Return the trace as a dictionary of arrays, with a leading checkpoint axis where one applies."""
        out = {name: np.stack([np.asarray(r[name]) for r in self.rows]) for name in self.rows[0]}
        out.update(probe_a=self.probe_a, probe_b=self.probe_b, probe_is_test=self.probe_is_test,
                   G_norm=np.array(self.G_norm),
                   A_0=np.array(self.A_0))
        return out


# =====================================================================
# The schema
# =====================================================================
NOTATION = (
    "H = I - (1/n) 1 1^T. <A, B> is the Frobenius inner product. K_t is the raw probe kernel and "
    "Kc_t = H K_t H. k_t = Kc_t / ||Kc_t||_F is the unit centred kernel. Y is the one-hot probe labels, "
    "G = H Y Y^T H and g = G / ||G||_F. g_perp is the part of g orthogonal to k_0, scaled to unit length. "
    "d_t = k_t - k_0 is the change of the unit kernel."
)

CONVENTIONS = dict(
    notation=NOTATION,
    kernel=("The sum-over-logits kernel of RepoducedCode.compute_entk in mode trace. "
            "K_ij is the sum over the p outputs of the inner product of the parameter gradients of that "
            "output at probe points i and j."),
    centring=("Every term, share and eigenvector uses the centred kernel H K H, following Cortes, Mohri and "
              "Rostamizadeh (2012), Lemma 1, and the report's Kernel metrics section."),
    probe=("The mixed probe of the report's Kernel metrics section, as ntk_lib.select_probe takes it. At p = 23 "
           "and a training fraction of 0.9 it holds 203 training pairs followed by all 53 test pairs."),
    precision=("torch computes the kernel in float32. Everything derived from it here is computed in float64 "
               "from that float32 kernel."),
    top_k=TOP_K,
    eigenvalue_order="Descending.",
    matching=(f"A linear assignment on minus the absolute overlaps with the previous checkpoint's top {TOP_K} "
              f"eigenvectors, by scipy.optimize.linear_sum_assignment. An assigned overlap below "
              f"{NEW_ARRIVAL} marks a new arrival."),
    sign=("Fixed by continuity, so that each matched eigenvector has a positive inner product with its "
          "predecessor. A vector with no predecessor has its largest entry in absolute value made positive."),
    label_subspace="The column space of H Y. It has dimension p - 1.",
    subspaces=("The centred functions on the probe points of (a + b) mod p, (a - b) mod p, a, and b, in that "
               "order. Each has dimension p - 1."),
    checkpoints=(f"Step 0, every {INTERVAL} steps, and {N_LOG} log-spaced steps from 1 to the run length, "
                 f"rounded to integers. Rounding merges some of the smallest ones."),
)

PLAN = "docs/zain_tierx_plan.md"
REPORT_METRICS = "docs/report.tex, Kernel metrics section"
COPIED = dict(formula="Copied from the history row at the checkpoint.", source="ntk_lib.train_run")

ARRAYS = dict(
    step=dict(meaning="The checkpoint steps.", formula="See conventions.checkpoints.",
              source=REPORT_METRICS + ", for the log-spaced grid"),
    probe_a=dict(meaning="The first number a of each probe pair. Stored once.", formula="-",
                 source="ntk_lib.probe_pairs"),
    probe_b=dict(meaning="The second number b of each probe pair. The label of probe point i is "
                         "(a_i + b_i) mod p. Stored once.", formula="-", source="ntk_lib.probe_pairs"),
    probe_is_test=dict(meaning="True at the probe pairs that are test pairs. Stored once.", formula="-",
                       source="ntk_lib.probe_pairs"),
    W1=dict(meaning="The hidden layer weights, shape (N, 2p).", formula="h = x W1^T / sqrt(2p)",
            source="ntk_lib.NTKMLP"),
    W2=dict(meaning="The output weights, shape (p, N).", formula="f = relu(h) W2^T / sqrt(N)",
            source="ntk_lib.NTKMLP"),
    K=dict(meaning="The raw probe kernel, in float32 as torch computed it.", formula="See conventions.kernel.",
           source="RepoducedCode.compute_entk through ntk_lib.entk"),
    f_probe=dict(meaning="The centred and rescaled predictor on the probe set.",
                 formula="alpha (f_t(x) - f_0(x))",
                 source="ntk_lib.train_run, which follows Kumar et al. (2024), Appendix 8.1, Equation 7"),
    train_loss=dict(meaning="Training loss.", **COPIED),
    test_loss=dict(meaning="Test loss.", **COPIED),
    train_acc=dict(meaning="Training accuracy.", **COPIED),
    test_acc=dict(meaning="Test accuracy.", **COPIED),
    param_dist=dict(meaning="Relative parameter movement ||w_t - w_0|| / ||w_0||.", **COPIED),
    weight_norm=dict(meaning="Norm of all the parameters.", **COPIED),
    yKy=dict(meaning="y^T K^+ y on the probe set, summed over the label columns.", **COPIED),
    K_norm_centred=dict(meaning="The Frobenius norm of the centred kernel.", formula="||Kc_t||_F",
                        source="Cortes, Mohri and Rostamizadeh (2012), Lemma 1, for H K H"),
    inner_K0=dict(meaning="Inner product of the centred kernel with the centred kernel at step 0.",
                  formula="<Kc_t, Kc_0>", source=PLAN + ", Section 6"),
    inner_G=dict(meaning="Inner product of the centred kernel with the centred target Gram matrix.",
                 formula="<Kc_t, G>", source=PLAN + ", Section 6"),
    G_norm=dict(meaning="The norm of the centred target Gram matrix. Stored once.", formula="||G||_F",
                source=REPORT_METRICS),
    A_0=dict(meaning="The centred alignment at step 0. Stored once.", formula="A at step 0",
             source=REPORT_METRICS),
    S=dict(meaning="The scale term on the centred kernel. It matches the history field S_c.",
           formula="log(||Kc_t||_F / ||Kc_0||_F)", source=REPORT_METRICS),
    R=dict(meaning="The rotation term on the centred kernel. It matches the history field R_c.",
           formula="1 - <k_t, k_0>", source=REPORT_METRICS),
    A=dict(meaning="The centred alignment. It matches the history field A_t.",
           formula="<Kc_t, G> / (||Kc_t||_F ||G||_F)",
           source="Cortes, Mohri and Rostamizadeh (2012), Definition 4; " + REPORT_METRICS),
    A_uncentred=dict(meaning="The alignment without centring. It matches the history field A_u.",
                     formula="<K_t, Y Y^T> / (||K_t||_F ||Y Y^T||_F)", source="ntk_lib.uncentred_alignment"),
    D=dict(meaning="The movement statistic on the centred kernel.", formula="||Kc_t - Kc_0||_F / ||Kc_0||_F",
           source="docs/report.tex, Equation eq:decomp, applied to the centred kernel"),
    exp2S_term=dict(meaning="The first piece of Equation eq:decomp. D^2 = exp2S_term + 1 - cross_term.",
                    formula="e^(2 S)", source="docs/report.tex, Equation eq:decomp, the group's own algebra"),
    cross_term=dict(meaning="The last piece of Equation eq:decomp.", formula="2 e^S (1 - R)",
                    source="docs/report.tex, Equation eq:decomp, the group's own algebra"),
    gamma=dict(meaning="The aim, the cosine between the part of k_t orthogonal to k_0 and g_perp. "
                       "NaN at step 0, where the kernel has not moved.",
               formula="<v_t, g_perp> / ||v_t||_F with v_t = k_t - <k_t, k_0> k_0",
               source="term_dependence.py docstring, the group's own algebra"),
    share_k0=dict(meaning="The share of ||d_t||^2 along k_0. NaN at step 0.",
                  formula="<d_t, k_0>^2 / ||d_t||^2, which equals R / 2", source=PLAN + ", E3"),
    share_gperp=dict(meaning="The share of ||d_t||^2 along g_perp. NaN at step 0.",
                     formula="<d_t, g_perp>^2 / ||d_t||^2, which equals gamma^2 (1 - R / 2)", source=PLAN + ", E3"),
    share_residual=dict(meaning="The share of ||d_t||^2 orthogonal to both k_0 and g_perp. The three shares "
                                "sum to one. NaN at step 0.",
                        formula="||d_t - <d_t, k_0> k_0 - <d_t, g_perp> g_perp||^2 / ||d_t||^2, which equals "
                                "(1 - gamma^2) (1 - R / 2)", source=PLAN + ", E3"),
    eigvals=dict(meaning="All eigenvalues of the centred kernel, largest first.", formula="eigenvalues of Kc_t",
                 source=PLAN + ", E4"),
    eigvecs_topk=dict(meaning="The eigenvectors of the top_k largest eigenvalues, as columns in eigenvalue order "
                              "at each checkpoint, with signs fixed as conventions.sign says.",
                      formula="eigenvectors of Kc_t", source=PLAN + ", E4"),
    match_index=dict(meaning="The column at the previous checkpoint that each column continues. Minus one for a "
                             "new arrival and at step 0.", formula="See conventions.matching.",
                     source=PLAN + ", E4"),
    overlap_prev=dict(meaning="The absolute inner product with the assigned column at the previous checkpoint. "
                              "It is kept for new arrivals too, where it is below the threshold. NaN at step 0.",
                      formula="|<u_t, u_(t-1)>|", source=PLAN + ", E4"),
    origin_index=dict(meaning="The column at step 0 that the chain of matches leads back to. Minus one if the "
                              "chain passes through a new arrival. This array is an addition to the plan.",
                      formula="match_index followed back to step 0", source=PLAN + ", E4"),
    overlap_zero=dict(meaning="The absolute inner product with the step 0 column that origin_index names. NaN "
                              "where origin_index is minus one.", formula="|<u_t, u_0>|", source=PLAN + ", E4"),
    principal_angles_zero=dict(meaning="Principal angles in radians between the span of the top eigenvectors "
                                       "and the same span at step 0, smallest first.",
                               formula="arccos of the singular values of V_t^T V_0", source=PLAN + ", E4"),
    principal_angles_label=dict(meaning="Principal angles in radians between the span of the top eigenvectors "
                                        "and the label subspace, smallest first.",
                                formula="arccos of the singular values of V_t^T Q with Q a basis of H Y",
                                source=PLAN + ", E4"),
    subspace_energy=dict(meaning="For each top eigenvector, the share of its squared length in each of the four "
                                 "subspaces of conventions.subspaces. The projection is by least squares, "
                                 "because the probe set is a subset of the pairs and the indicator functions are "
                                 "not orthogonal on it. The four shares need not sum to one.",
                         formula="||Q_s^T u||^2 with Q_s an orthonormal basis of subspace s",
                         source=PLAN + ", E3 and E4"),
    torus_K_unit=dict(meaning="The torus average of the unit centred kernel.", formula="Torus(k_t)",
                      source="animations/common/kernels.Torus"),
    torus_change=dict(meaning="The torus average of the change of the unit centred kernel since step 0.",
                      formula="Torus(d_t)", source="animations/common/kernels.Torus"),
    torus_residual=dict(meaning="The torus average of the part of d_t that share_residual measures. It is not "
                                "divided by its length.",
                        formula="Torus(d_t - <d_t, k_0> k_0 - <d_t, g_perp> g_perp)",
                        source="animations/common/kernels.Torus"),
)


def checks(run, a):
    """Largest differences that should sit at rounding level. They are printed and saved in the schema."""
    h = run["history"]
    if list(h["step"]) != a["step"].tolist():
        raise ValueError("the history steps and the trace steps differ")
    R, gamma, A_0 = a["R"], a["gamma"], float(a["A_0"])
    moved = np.arange(len(R)) > 0
    identity = A_0 * (1 - R) + gamma * np.sqrt(1 - A_0 ** 2) * np.sqrt(np.clip(R * (2 - R), 0, None))

    def worst(x):
        return float(np.nanmax(np.abs(x)))

    return {
        "S against the history field S_c": worst(a["S"] - np.asarray(h["S_c"])),
        "R against the history field R_c": worst(R - np.asarray(h["R_c"])),
        "A against the history field A_t": worst(a["A"] - np.asarray(h["A_t"])),
        "A_uncentred against the history field A_u": worst(a["A_uncentred"] - np.asarray(h["A_u"])),
        "D^2 against exp2S_term + 1 - cross_term": worst(a["D"] ** 2 - (a["exp2S_term"] + 1 - a["cross_term"])),
        "the three shares against a sum of one": worst(a["share_k0"] + a["share_gperp"] + a["share_residual"] - 1),
        "share_k0 against R / 2": worst(a["share_k0"] - R / 2),
        "share_gperp against gamma^2 (1 - R / 2)": worst(a["share_gperp"] - gamma ** 2 * (1 - R / 2)),
        "A against the aim identity": worst((a["A"] - identity)[moved]),
    }


def write(name, run, tracer):
    """Write the trace arrays and the schema of a run to results/, and return the checks."""
    arrays = tracer.arrays()
    np.savez_compressed(L.RESULTS / f"{name}_trace.npz", **arrays)
    found = checks(run, arrays)
    c = run["config"]
    schema = dict(
        run=name, run_json=f"{name}.json",
        cell=dict(alpha=c["alpha"], eta_kappa=c["eta_kappa"], steps=c["steps"], width=c["hidden_dim"],
                  seed=c["seed"], checkpoints=len(arrays["step"])),
        conventions=CONVENTIONS, checks=found,
        arrays={key: dict(shape=list(v.shape), dtype=str(v.dtype), **ARRAYS[key]) for key, v in arrays.items()},
    )
    (L.RESULTS / f"{name}_trace.json").write_text(json.dumps(schema, indent=1) + "\n")
    return found


def load(name):
    """Return the trace arrays of a run as a dictionary, and its schema."""
    path = L.RESULTS / f"{name}_trace.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run repoduced-code/ntk_trace.py to write it.")
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads((L.RESULTS / f"{name}_trace.json").read_text())


# =====================================================================
# Running and comparing
# =====================================================================
def trace(alpha, eta_kappa, steps, force=False):
    """Train one cell with the tracer attached and write its trace.

    If the trace exists and the saved run matches the configuration, nothing
    is trained. run_or_load calls the tracer only when it trains, so the
    trace is written exactly when the run was trained here.
    """
    name = run_name(alpha, eta_kappa)
    cfg = run_config(alpha, eta_kappa, steps)
    tracer = Tracer({**L.DEFAULTS, **cfg})
    missing = not (L.RESULTS / f"{name}_trace.npz").exists()
    run = L.run_or_load(name, rerun=force or missing, on_checkpoint=tracer, **cfg)
    if tracer.rows:
        found = write(name, run, tracer)
        print(f"{name}: wrote results/{name}_trace.npz with {len(tracer.rows)} checkpoints")
    else:
        found = load(name)[1]["checks"]
        print(f"{name}: the trace exists and matches the configuration")
    for check, value in found.items():
        print(f"  {check:>45}: largest difference {value:.1e}")
    return run


TRAINING_FIELDS = ("train_loss", "test_loss", "train_acc", "test_acc", "param_dist", "weight_norm")


def compare_with_saved(run, alpha, eta_kappa):
    """Print how far the history is from the saved grid run and dense run at the checkpoints they share.

    Training is deterministic, so the fields that depend only on the weights
    agree exactly with a run made from the same configuration. That holds
    across machines too: on 24 September 2026 the grid runs, which a teammate
    committed, agreed exactly in these fields. The kernel fields depend on
    float32 sums inside the kernel computation, which can round differently on
    another machine. y^T K^+ y magnifies that rounding through the
    pseudo-inverse. Each kernel difference is therefore printed as a fraction
    of the largest absolute value of that field in the saved run. The kernel
    fields are compared only when the saved run used the same probe. The
    dense runs keep the probe of training pairs, so for them only the
    training fields are compared.

    Returns a dictionary with the largest differences for each saved run
    that exists. The kernel entry is None when the probes differ.
    """
    mine = run["history"]
    index = {s: i for i, s in enumerate(mine["step"])}
    found = {}
    for kind, prefix in [("grid", "ntk"), ("dense", "dense")]:
        name = prefix + L.cell_name(WIDTH, alpha, eta_kappa, SEED)[len("ntk"):]
        path = L.RESULTS / f"{name}.json"
        if not path.exists():
            print(f"  no saved {kind} run {name} to compare with")
            continue
        saved_run = json.loads(path.read_text())
        saved = saved_run["history"]
        same_probe = {**L.LEGACY_VALUES, **saved_run["config"]}["probe"] == run["config"]["probe"]
        shared = [(index[s], j) for j, s in enumerate(saved["step"]) if s in index]
        training, kernel = {}, {}
        for key in [k for k in L.HISTORY_KEYS[1:] if k in saved]:
            if key not in TRAINING_FIELDS and not same_probe:
                continue
            pairs = np.array([(mine[key][i], saved[key][j]) for i, j in shared], dtype=float)
            both = ~np.isnan(pairs).any(axis=1)
            diff = float(np.max(np.abs(pairs[both, 0] - pairs[both, 1]))) if both.any() else 0.0
            if key in TRAINING_FIELDS:
                training[key] = diff
            else:
                kernel[key] = diff / max(float(np.nanmax(np.abs(saved[key]))), 1e-30)
        found[kind] = dict(training=max(training.values()), kernel=max(kernel.values()) if kernel else None)
        text = (f"  {kind} run {name}: {len(shared)} shared checkpoints. Training fields: largest difference "
                f"{found[kind]['training']:.1e}.")
        if kernel:
            worst = max(kernel, key=kernel.get)
            text += f" Kernel fields: largest difference {kernel[worst]:.1e} of the field's largest value, in {worst}."
        else:
            text += " Kernel fields not compared, because the saved run uses another probe."
        print(text)
    return found


def main(alphas):
    for alpha, eta_kappa, steps in CELLS:
        if alphas and alpha not in alphas:
            continue
        run = trace(alpha, eta_kappa, steps)
        compare_with_saved(run, alpha, eta_kappa)


if __name__ == "__main__":
    main([float(v) for v in sys.argv[1:]])
