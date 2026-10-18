"""
This module collects the training loop, the grokking time, and the plot
helpers that the notebooks in this folder use. The dataset, the standard
parameterisation model, the empirical kernel, the raw scale and rotation
terms, and the centred alignment come from RepoducedCode.py and are imported
from there.

Sources. Every claim that comes from a paper names the paper and, where
possible, the section or equation.

- Kumar et al. (2024), arXiv:2310.06110v3. Appendix 8.3 gives the modular
  addition baseline. Appendix 8.1, Equation 7, gives the centred and
  rescaled predictor and the learning rate scaling. Appendix 8.2 states that
  the initialisation scale sigma and alpha act through the product
  sigma squared times alpha. Read on 14 September 2026.
- Gromov (2023), arXiv:2301.02679. Equations 1 and 2 define the two layer
  network in mean-field parameterisation with standard normal weights.
  Read on 14 September 2026.
- Cortes, Mohri, and Rostamizadeh (2012), Journal of Machine Learning
  Research 13, pages 795 to 828. Lemma 1 gives the centred kernel matrix
  H K H with H = I - (1/n) 1 1^T.
"""

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, grad, stack_module_state

from RepoducedCode import (
    make_modular_addition_dataset,
    ModularMLP,
    compute_entk,
    compute_scale_and_rotation,
    compute_task_alignment,
)

DEVICE = "cpu"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

# Colours in a fixed categorical order. Train is always blue and test is
# always orange. The ramp is for ordered sweeps such as alpha or sigma.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })


# =====================================================================
# Models
# =====================================================================
class MeanFieldMLP(nn.Module):
    """One hidden layer network in mean-field parameterisation.

    This follows Gromov (2023), Equations 1 and 2. The hidden preactivation
    is W1 x divided by the input size D, and the output is W2 phi(h) divided
    by the width N. Both weight matrices are drawn from a normal distribution
    with standard deviation init_scale, which is the sigma of Kumar et al.
    (2024), Appendix 8.2. Gromov uses a quadratic activation. ReLU is the
    default here because the proposal's homogeneity argument is stated for a
    bias-free ReLU network. There are no biases, so the network is
    2-homogeneous in its parameters.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu", init_scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.W1 = nn.Parameter(init_scale * torch.randn(hidden_dim, input_dim))
        self.W2 = nn.Parameter(init_scale * torch.randn(output_dim, hidden_dim))

    def forward(self, x):
        h = x @ self.W1.T / self.input_dim
        if self.activation == "relu":
            z = torch.relu(h)
        elif self.activation == "quadratic":
            z = h * h
        else:
            raise ValueError(self.activation)
        return z @ self.W2.T / self.hidden_dim


class NTKMLP(nn.Module):
    """One hidden layer network in NTK parameterisation, as stated in the report.

    f(x) = (1 / sqrt(N)) sum_i a_i phi(w_i . x / sqrt(D)) with a_i and w_ij
    drawn from a normal distribution of standard deviation init_scale. This is
    the parameterisation of Jacot, Gabriel, and Hongler (2018). There are no
    biases, so the network is 2-homogeneous. It differs from MeanFieldMLP by
    a factor of sqrt(N) on the output and sqrt(D) on the hidden preactivation.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu", init_scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.W1 = nn.Parameter(init_scale * torch.randn(hidden_dim, input_dim))
        self.W2 = nn.Parameter(init_scale * torch.randn(output_dim, hidden_dim))

    def forward(self, x):
        h = x @ self.W1.T / math.sqrt(self.input_dim)
        z = torch.relu(h) if self.activation == "relu" else h * h
        return z @ self.W2.T / math.sqrt(self.hidden_dim)


class OmnigrokMLP(nn.Module):
    """The MNIST network of Liu, Michaud and Tegmark (2023), Appendix A.

    It is a ReLU perceptron with three linear layers, each of width
    hidden_dim except the last, with biases and PyTorch's default
    initialisation. Section 4 of that paper multiplies the initial weights
    by a constant, and build_model does this to every parameter, so that
    the ratio of the parameter norm to its default value is init_scale. The
    biases make the network inhomogeneous, so the closed-form kernel does
    not apply and the kernel is computed by autograd.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class DeepBiasFreeMLP(nn.Module):
    """A bias-free ReLU network with depth linear layers, for the MNIST study.

    Every hidden layer has width hidden_dim. Layer l computes
    h_l = c_l a_{l-1} W_l^T with a_0 = x, and a_l = relu(h_l) for every
    layer but the last, whose h_L is the output. There are no biases, so the
    network is depth-homogeneous in its weights.

    scaling chooses the parameterisation. Under "standard" each W_l is drawn
    uniformly from (-1 / sqrt(fan_in), 1 / sqrt(fan_in)), the draw that
    PyTorch's nn.Linear makes for its weight, and c_l = 1. With depth 3 and
    width 200 this is OmnigrokMLP without its biases. Under "ntk" each W_l
    is drawn from N(0, 1) and c_l = 1 / sqrt(fan_in), which is NTKMLP when
    depth is 2. Every weight is multiplied by init_scale after the draw.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, depth=3, scaling="standard", init_scale=1.0):
        super().__init__()
        if depth < 2:
            raise ValueError("DeepBiasFreeMLP needs at least two layers")
        self.input_dim, self.hidden_dim, self.depth, self.scaling = input_dim, hidden_dim, depth, scaling
        self.activation = "relu"
        sizes = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
        self.factors = []
        for l in range(depth):
            fan_in, fan_out = sizes[l], sizes[l + 1]
            if scaling == "standard":
                bound = 1.0 / math.sqrt(fan_in)
                W, c = torch.empty(fan_out, fan_in).uniform_(-bound, bound), 1.0
            elif scaling == "ntk":
                W, c = torch.randn(fan_out, fan_in), 1.0 / math.sqrt(fan_in)
            else:
                raise ValueError(f"unknown scaling {scaling!r}")
            setattr(self, f"W{l + 1}", nn.Parameter(init_scale * W))
            self.factors.append(c)

    def weights(self):
        """Return the weight matrices W_1 to W_L in order."""
        return [getattr(self, f"W{l + 1}") for l in range(self.depth)]

    def forward(self, x):
        a = x
        for l, (W, c) in enumerate(zip(self.weights(), self.factors)):
            h = c * (a @ W.T)
            a = torch.relu(h) if l < self.depth - 1 else h
        return a


def build_model(parameterisation, input_dim, hidden_dim, output_dim, activation="relu", init_scale=1.0,
                depth=None):
    """Return a fresh model. The caller sets the seed before calling this.

    "deep_standard" and "deep_ntk" build a DeepBiasFreeMLP with depth layers
    in the standard or NTK parameterisation. depth is used by these two only.
    """
    if parameterisation in ("deep_standard", "deep_ntk"):
        if depth is None:
            raise ValueError(f"parameterisation {parameterisation!r} needs depth")
        if activation != "relu":
            raise ValueError("DeepBiasFreeMLP uses relu only")
        return DeepBiasFreeMLP(input_dim, hidden_dim, output_dim, depth=depth,
                               scaling=parameterisation[len("deep_"):], init_scale=init_scale)
    if parameterisation == "omnigrok":
        model = OmnigrokMLP(input_dim, hidden_dim, output_dim)
        with torch.no_grad():
            for q in model.parameters():
                q.mul_(init_scale)
        return model
    if parameterisation == "ntk":
        return NTKMLP(input_dim, hidden_dim, output_dim, activation, init_scale)
    if parameterisation == "standard":
        model = ModularMLP(input_dim, hidden_dim, output_dim)
        if init_scale != 1.0:
            with torch.no_grad():
                for q in model.parameters():
                    q.mul_(init_scale)
        return model
    if parameterisation == "mean_field":
        return MeanFieldMLP(input_dim, hidden_dim, output_dim, activation, init_scale)
    raise ValueError(parameterisation)


# =====================================================================
# Kernel computation for wide networks
# =====================================================================
def compute_entk_by_logit(model, params, x_probe):
    """The sum-over-logits empirical kernel, built one output at a time.

    compute_entk in RepoducedCode.py builds the full Jacobian of shape
    (probe points, outputs, parameters) in one call. At width 1600 that is
    about 2.6 GB, so this version takes the gradient of one output at a time
    with torch.func.grad inside vmap and accumulates the Gram matrix. The
    result is the same kernel up to floating point rounding.
    """
    def output_c(c):
        def f(p_dict, x_single):
            return functional_call(model, p_dict, x_single.unsqueeze(0)).squeeze(0)[c]
        return f

    n_out = functional_call(model, params, x_probe[:1]).shape[1]
    K = torch.zeros((x_probe.shape[0], x_probe.shape[0]), device=x_probe.device)
    for c in range(n_out):
        grads = vmap(grad(output_c(c)), (None, 0))(params, x_probe)
        G = torch.cat([g.flatten(start_dim=1) for g in grads.values()], dim=1)
        K += G @ G.T
    return K


def entk(model, x_probe, chunk_above=2_000_000):
    """Return the probe kernel, choosing the memory-safe path for large networks."""
    params = dict(model.named_parameters())
    n_params = sum(q.numel() for q in params.values())
    if n_params * x_probe.shape[0] > chunk_above:
        return compute_entk_by_logit(model, params, x_probe).detach()
    return compute_entk(model, params, x_probe, mode="trace").detach()


def closed_form_scales(model):
    """Return the factors c1 on the hidden preactivation and c2 on the output of a bias-free model."""
    if isinstance(model, NTKMLP):
        return 1.0 / math.sqrt(model.input_dim), 1.0 / math.sqrt(model.hidden_dim)
    if isinstance(model, MeanFieldMLP):
        return 1.0 / model.input_dim, 1.0 / model.hidden_dim
    raise TypeError(f"the closed-form kernel needs NTKMLP or MeanFieldMLP, not {type(model).__name__}")


def entk_closed_form(model, x_probe, parts=False, output=None):
    """The sum-over-logits kernel of a bias-free one hidden layer ReLU network, from its weights.

    Write h = c1 X W1^T for the hidden preactivations, Z = relu(h), M for the
    mask of positive entries of h, and s_i for the sum over outputs of W2[c, i]
    squared. The factors are c1 = 1 / sqrt(2p) and c2 = 1 / sqrt(N) for
    NTKMLP, and 1 / (2p) and 1 / N for MeanFieldMLP. The gradient of output c
    with respect to W2[c, i] is c2 Z_i, and with respect to row i of W1 it is
    c2 W2[c, i] M_i c1 x. Summing the inner products over the outputs gives

        K = p c2^2 Z Z^T + c1^2 c2^2 (X X^T) * (M diag(s) M^T),

    where the star is the elementwise product. The first term comes from W2
    and the second from W1. This is the group's own derivation from the
    definition in RepoducedCode.compute_entk in mode "trace". No paper
    states it. It costs O(n^2 N) instead of a Jacobian of n p rows.

    The sums run in float64. With parts=False the result is returned in
    float32, the precision train_run works in. With parts=True the two terms
    are returned in float64 as (K_W1, K_W2). With output=c the kernel is that
    of output c alone, the first-logit kernel of the report when c = 0.
    """
    if model.activation != "relu":
        raise ValueError("the closed-form kernel is derived for the relu activation only")
    c1, c2 = closed_form_scales(model)
    K_W1, K_W2 = closed_form_terms(model.W1, model.W2, x_probe, c1, c2, output)
    if parts:
        return K_W1, K_W2
    return (K_W1 + K_W2).float()


def closed_form_terms(W1, W2, x, c1, c2, output=None):
    """Return the two terms (K_W1, K_W2) of entk_closed_form in float64, from the weight matrices."""
    X = x.detach().double()
    W1, W2 = W1.detach().double(), W2.detach().double()
    h = c1 * (X @ W1.T)
    Z = torch.relu(h)
    M = (h > 0).double()
    if output is None:
        s, n_out = (W2 ** 2).sum(0), W2.shape[0]
    else:
        s, n_out = W2[output] ** 2, 1
    K_W2 = n_out * c2 ** 2 * (Z @ Z.T)
    K_W1 = c1 ** 2 * c2 ** 2 * (X @ X.T) * ((M * s) @ M.T)
    return K_W1, K_W2


def closed_form_terms_deep(weights, factors, x):
    """Return the per-layer terms of the sum-over-outputs kernel of a bias-free ReLU network, in float64.

    weights are W_1 to W_L and factors are c_1 to c_L, with h_l = c_l a_{l-1} W_l^T,
    a_0 = x, a_l = relu(h_l) for l < L and f = h_L. M_l is the mask of positive
    entries of h_l. The gradient of output c with respect to W_l is
    c_l d_{l,c} a_{l-1}^T, where d_{L,c} = e_c and
    d_{l,c} = M_l * (c_{l+1} W_{l+1}^T d_{l+1,c}) for l < L. Summing the inner
    products of these gradients over the outputs gives

        K(x, x') = sum_l c_l^2 <a_{l-1}(x), a_{l-1}(x')> <D_l(x), D_l(x')>_F,

    where D_l(x) is the width_l by C matrix whose column c is d_{l,c}(x). For
    the last layer D_L is the C by C identity, so the second factor is C. The
    list returned holds the L terms, layer 1 first; their sum is the kernel.
    For L = 2 the terms are (K_W1, K_W2) of closed_form_terms. This is the
    group's own derivation from the definition in RepoducedCode.compute_entk
    in mode "trace". No paper states it.
    """
    X = x.detach().double()
    Ws = [W.detach().double() for W in weights]
    L = len(Ws)
    acts, masks = [X], []
    for l in range(L - 1):
        h = factors[l] * (acts[-1] @ Ws[l].T)
        masks.append((h > 0).double())
        acts.append(torch.relu(h))
    C = Ws[-1].shape[0]
    terms = [None] * L
    terms[L - 1] = factors[L - 1] ** 2 * C * (acts[L - 1] @ acts[L - 1].T)
    D = None
    for l in range(L - 2, -1, -1):
        back = factors[l + 1] * Ws[l + 1].T                 # width_l by width_{l+1}
        if D is None:
            D = masks[l][:, :, None] * back[None]          # n by width_l by C
        else:
            D = masks[l][:, :, None] * torch.einsum("ik,nkc->nic", back, D)
        Dflat = D.reshape(D.shape[0], -1)
        terms[l] = factors[l] ** 2 * (acts[l] @ acts[l].T) * (Dflat @ Dflat.T)
    return terms


def entk_closed_form_deep(model, x_probe, parts=False):
    """The sum-over-outputs kernel of a bias-free ReLU network of any depth, from its weights.

    The model is a DeepBiasFreeMLP, an NTKMLP or a MeanFieldMLP.
    closed_form_terms_deep gives the formula. With parts=True the per-layer
    terms are returned in float64, layer 1 first. Otherwise their sum is
    returned in float32, as entk_closed_form does.
    """
    if isinstance(model, DeepBiasFreeMLP):
        weights, factors = model.weights(), model.factors
    else:
        if model.activation != "relu":
            raise ValueError("the closed-form kernel is derived for the relu activation only")
        weights, factors = [model.W1, model.W2], list(closed_form_scales(model))
    terms = closed_form_terms_deep(weights, factors, x_probe)
    if parts:
        return terms
    return sum(terms).float()


def probe_kernel(model, x_probe, method):
    """Return the probe kernel by the chosen method, "autograd" or "closed_form".

    "closed_form" uses entk_closed_form_deep for a DeepBiasFreeMLP and
    entk_closed_form for the two layer models.
    """
    if method == "autograd":
        return entk(model, x_probe)
    if method == "closed_form":
        if isinstance(model, DeepBiasFreeMLP):
            return entk_closed_form_deep(model, x_probe)
        return entk_closed_form(model, x_probe)
    raise ValueError(f"unknown kernel method {method!r}")


def gauss_newton_top_eigenvalue(model, X, alpha=1.0, f0=None, iters=50, seed=0):
    """Largest eigenvalue of (2 / (n C)) J^T J at the model's current weights, by power iteration.

    J is the Jacobian of the predictor alpha f on X, or alpha (f - f0) if f0
    is given, with respect to the parameters. n C is the number of entries
    MSELoss averages over. This is top_gauss_newton_eigenvalue of mnist/exploration/mnist_pilot.py
    applied to a given model. The start vector comes from a generator seeded
    with seed, so the global random state is not touched.

    The weights stay fixed while the iteration runs, so the Jacobian-vector
    product comes from torch.func.linearize, which its documentation gives
    for a jvp computed many times at the same primals, and the
    vector-Jacobian product from one torch.func.vjp, whose function is
    called with each new cotangent.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}

    def predict(p):
        out = functional_call(model, p, X)
        return alpha * (out - f0) if f0 is not None else alpha * out

    gen = torch.Generator().manual_seed(seed)
    v = {k: torch.randn(t.shape, generator=gen, dtype=t.dtype).to(t.device) for k, t in params.items()}
    _, jvp_fn = torch.func.linearize(predict, params)
    _, pullback = torch.func.vjp(predict, params)
    lam = 0.0
    for _ in range(iters):
        norm = torch.sqrt(sum((t ** 2).sum() for t in v.values()))
        v = {k: t / norm for k, t in v.items()}
        Jv = jvp_fn(v)
        (Gv,) = pullback(2.0 * Jv / Jv.numel())
        lam = sum((Gv[k] * v[k]).sum() for k in v).item()
        v = Gv
    return lam


# =====================================================================
# Centred scale and rotation
# =====================================================================
def centre_kernel(K):
    """Return H K H with H = I - (1/n) 1 1^T, Cortes et al. (2012), Lemma 1."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones((n, n), device=K.device, dtype=K.dtype) / n
    return H @ K @ H


def batched_centre(K):
    """centre_kernel for a stack of kernels with any leading dimensions."""
    n = K.shape[-1]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones((n, n), device=K.device, dtype=K.dtype) / n
    return H @ K @ H


def batched_scale_and_rotation(K_0, K_t):
    """compute_scale_and_rotation for stacks of kernels, returning tensors with the leading dimensions."""
    norm_0, norm_t = torch.linalg.matrix_norm(K_0), torch.linalg.matrix_norm(K_t)
    overlap = ((K_t / norm_t[..., None, None]) * (K_0 / norm_0[..., None, None])).sum((-2, -1))
    return torch.log(norm_t / norm_0), 1.0 - overlap


def batched_task_alignment(K_t, y_probe):
    """compute_task_alignment for stacks of kernels and labels, zero where a centred matrix vanishes."""
    n = K_t.shape[-1]
    H = (torch.eye(n, device=K_t.device, dtype=K_t.dtype)
         - (1.0 / n) * torch.ones((n, n), device=K_t.device, dtype=K_t.dtype))
    HKH = H @ K_t @ H
    HYH = H @ (y_probe @ y_probe.mT) @ H
    norm_HKH, norm_HYH = torch.linalg.matrix_norm(HKH), torch.linalg.matrix_norm(HYH)
    alignment = (HKH * HYH).sum((-2, -1)) / (norm_HKH * norm_HYH)
    return torch.where((norm_HKH == 0) | (norm_HYH == 0), torch.zeros_like(alignment), alignment)


def batched_uncentred_alignment(K, y):
    """uncentred_alignment for stacks of kernels and labels."""
    YY = y @ y.mT
    return (K * YY).sum((-2, -1)) / (torch.linalg.matrix_norm(K) * torch.linalg.matrix_norm(YY))


def uncentred_alignment(K, y):
    """The alignment without centring, sum(y^T K y) / (||K||_F ||y y^T||_F), summed over the label columns.

    This is the form printed in Kumar et al. (2024), Section 5, up to their
    choice of evaluating it on the test set at initialisation only. It is
    recorded for comparability; the centred version is primary.
    """
    YY = y @ y.T
    return (torch.sum(K * YY) / (torch.linalg.norm(K, ord="fro") * torch.linalg.norm(YY, ord="fro"))).item()


def compute_centred_scale_and_rotation(K_0, K_t):
    """The scale and rotation terms of the proposal computed on the centred kernels.

    The formulas are the same as in compute_scale_and_rotation, applied to
    H K_0 H and H K_t H. Centring removes the component of the kernel that
    is constant across probe points. With one-hot inputs and ReLU units
    that component is large, and it can hide movement of the eigenbasis
    when the terms are computed on the raw kernel.
    """
    return compute_scale_and_rotation(centre_kernel(K_0), centre_kernel(K_t))


def sum_kernel(W1, W2, X):
    """Kernel of g = sum_c f_c / sqrt(p) for f = W2 relu(W1 x / sqrt(D)) / sqrt(N), in closed form.

    This is the kernel of golden report Equation 5. g has one output whose
    second-layer weight is u = W2.sum(0) / sqrt(p). Its gradient with respect
    to W2[c, i] is relu(h_i) / sqrt(N p) for every c, which sums to Z Z^T / N
    over c. Its gradient with respect to row i of W1 is u_i 1[h_i > 0] x / sqrt(D N).
    Takes and returns float64 numpy arrays.
    """
    D, N = X.shape[1], W1.shape[0]
    h = X @ W1.T / np.sqrt(D)
    Z, M = np.maximum(h, 0), (h > 0).astype(float)
    u = W2.sum(0) / np.sqrt(W2.shape[0])
    return Z @ Z.T / N + (X @ X.T) * ((M * u ** 2) @ M.T) / (D * N)


def report_statistics(K_t, K_0, Y):
    """S_t, R_t, D_t and A_t of golden report Equation 6 for one pair of kernels on the test pairs."""
    r = len(K_t)
    C = np.eye(r) - 1 / r
    n_t, n_0 = np.linalg.norm(K_t), np.linalg.norm(K_0)
    Kc, Gc = C @ K_t @ C, C @ Y @ Y.T @ C
    return {"S_t": n_t / n_0,
            "R_t": 1 - np.sum(K_t * K_0) / (n_t * n_0),
            "D_t": np.linalg.norm(K_t - K_0) / n_0,
            "A_t": np.sum(Kc * Gc) / (np.linalg.norm(Kc) * np.linalg.norm(Gc))}


# =====================================================================
# Training
# =====================================================================
DEFAULTS = dict(parameterisation="mean_field", p=23, hidden_dim=100, alpha=1.0,
                eta_0=100.0, eta_kappa=0.0, steps=60000, eval_interval=250,
                probe_size=256, seed=0, data_seed=42, train_fraction=0.9,
                activation="relu", init_scale=1.0, kernel_save_interval=5000, probe="train",
                checkpoint_steps=None, kernel_method="autograd",
                dataset="modular_addition", n_train=None, centred=True, optimizer="sgd",
                batch_size=None, momentum=0.0,
                depth=None, lr_schedule=None, lr_update_interval=10000, lr_norm_power=4.0, lr_cap=1000.0,
                dtype="float32", fp32_precision="ieee")

# Configuration keys added after some runs were saved, with the value those
# runs were trained with. A saved run that lacks one of them is compared as if
# it had that value.
LEGACY_VALUES = dict(probe="train", checkpoint_steps=None, kernel_method="autograd",
                     dataset="modular_addition", n_train=None, centred=True, optimizer="sgd",
                     batch_size=None, momentum=0.0,
                     depth=None, lr_schedule=None, lr_update_interval=10000, lr_norm_power=4.0, lr_cap=1000.0,
                     dtype="float32", fp32_precision="ieee")

# The centred terms and the two alignments on the training part and on the
# test part of the probe. A part with fewer than two pairs gives NaN.
PART_KEYS = [f"{term}_{part}" for part in ("train", "test") for term in ("S_c", "R_c", "A_t", "A_u")]

HISTORY_KEYS = ["step", "train_loss", "test_loss", "train_acc", "test_acc",
                "S_t", "R_t", "S_c", "R_c", "A_t", "A_u", "param_dist", "weight_norm",
                "yKy", "K_norm"] + PART_KEYS


def select_probe(X_train, y_train, X_test, y_test, probe, probe_size):
    """Return the probe inputs, the probe labels, and a mask that is True at the test pairs.

    "mixed" is the probe of the report's Kernel metrics section. It takes
    test pairs, up to half of probe_size, and fills the rest with training
    pairs, training pairs first. At p = 23 and a training fraction of 0.9
    there are only 53 test pairs, so the probe holds all 53 and 203 training
    pairs. "train" takes the first probe_size training pairs. Every run saved
    before 24 September 2026 used it. "test" takes the first probe_size test
    pairs. The first pairs of each set are a random draw, because the split
    is a random permutation fixed by data_seed, and they are the same for
    every run with the same data_seed.
    """
    if probe == "train":
        n_train, n_test = min(probe_size, len(X_train)), 0
    elif probe == "test":
        n_train, n_test = 0, min(probe_size, len(X_test))
    elif probe == "mixed":
        n_test = min(probe_size // 2, len(X_test))
        n_train = min(probe_size - n_test, len(X_train))
    else:
        raise ValueError(f"unknown probe {probe!r}")
    x = torch.cat([X_train[:n_train], X_test[:n_test]])
    y = torch.cat([y_train[:n_train], y_test[:n_test]])
    is_test = torch.cat([torch.zeros(n_train, dtype=torch.bool, device=x.device),
                         torch.ones(n_test, dtype=torch.bool, device=x.device)])
    return x, y, is_test


def make_dataset(cfg):
    """Return ((X_train, y_train), (X_test, y_test)) for the dataset a configuration names.

    "modular_addition" is the task of the report and uses p, train_fraction
    and data_seed. "mnist" takes n_train training images drawn at random by
    data_seed and the full test set, as mnist_data.load_mnist describes. The
    targets are one-hot in both cases, so the model's input and output sizes
    are read from the data.
    """
    if cfg["dataset"] == "modular_addition":
        return make_modular_addition_dataset(p=cfg["p"], train_fraction=cfg["train_fraction"],
                                             seed=cfg["data_seed"])
    if cfg["dataset"] == "mnist":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mnist"))
        from mnist_data import load_mnist
        if cfg["n_train"] is None:
            raise ValueError("an MNIST run needs n_train")
        return load_mnist(n_train=cfg["n_train"], seed=cfg["data_seed"])
    raise ValueError(f"unknown dataset {cfg['dataset']!r}")


def probe_pairs(cfg):
    """Return the probe pairs (a, b) of a configuration and the mask of test pairs, as numpy arrays."""
    if cfg.get("dataset", "modular_addition") != "modular_addition":
        raise ValueError("probe pairs exist only for modular addition")
    p = cfg["p"]
    (X_train, y_train), (X_test, y_test) = make_modular_addition_dataset(
        p=p, train_fraction=cfg["train_fraction"], seed=cfg["data_seed"])
    x, _, is_test = select_probe(X_train, y_train, X_test, y_test, cfg["probe"], cfg["probe_size"])
    x = x.numpy()
    return x[:, :p].argmax(1), x[:, p:].argmax(1), is_test.numpy()


def normalise_checkpoints(cfg):
    """Return the configuration with checkpoint_steps as a sorted list of distinct integers.

    Step 0 is always a checkpoint, so it is added if missing. The list form
    is what train_run saves, so run_or_load can compare a requested list
    with a saved one whatever sequence type the caller passed. None, the
    default, is returned unchanged and means every eval_interval steps.
    """
    given = cfg["checkpoint_steps"]
    if given is None:
        return cfg
    if any(s != int(s) for s in given):
        raise ValueError("checkpoint steps must be whole numbers")
    steps = sorted({0, *(int(s) for s in given)})
    if steps[0] < 0 or steps[-1] > cfg["steps"]:
        raise ValueError(f"checkpoint steps must lie between 0 and the run length {cfg['steps']}")
    return {**cfg, "checkpoint_steps": steps}


def train_run(name, verbose=True, on_checkpoint=None, results_dir=None, **overrides):
    """Train one run and return a dictionary with the configuration and history.

    The predictor is the centred and rescaled function of Kumar et al.
    (2024), Appendix 8.1, Equation 7, with learning rate eta_0 over alpha
    squared. Weight decay enters as a factor of one minus eta times kappa per
    step. The model seed is separate from the data seed. The train loss is
    recorded after the update, at the same point as the test loss. The
    initial function f_0 on the training and test sets is computed once,
    since it never changes. This gives the same numbers bit for bit as
    computing it at every step, and it saves a forward pass per step.

    At each checkpoint the history records the losses, the accuracies, the
    raw scale and rotation terms S_t and R_t, the centred versions S_c and
    R_c, the centred alignment A_t, the relative parameter movement, the
    weight norm, y^T K^+ y on the probe set, and the kernel norm. It also
    records S_c, R_c, A_t and A_u on the training part and on the test part
    of the probe, as the report's Kernel metrics section asks. The probe
    kernel itself is saved every kernel_save_interval steps to a compressed
    file next to the JSON so later analyses can use it.

    The checkpoints are every eval_interval steps. If checkpoint_steps is
    given, they are exactly those steps instead, with step 0 added. Either
    way the kernel is saved only at checkpoints that are multiples of
    kernel_save_interval.

    If on_checkpoint is given, it is called at every checkpoint as
    on_checkpoint(step, model, K_t, row). K_t is the probe kernel and row is
    a dictionary of the history values just recorded, extra keys included. Training continues
    from the same model afterwards, so the callback must not change it. The
    callback is not part of the configuration and is not saved.

    kernel_method chooses how the probe kernel is computed: "autograd" with
    torch.func, or "closed_form" with entk_closed_form, which gives the same
    kernel to float32 rounding for the bias-free ReLU models. results_dir
    chooses where the files are written. It defaults to RESULTS and is not
    part of the configuration.

    dataset chooses the task, as make_dataset describes. centred=False drops
    the subtraction of f_0, so the predictor is alpha f. optimizer is "sgd",
    plain gradient descent, or "adamw", the optimiser of Liu, Michaud and
    Tegmark (2023), Appendix A. The tangent kernel describes the
    function-space step only under "sgd" without momentum. momentum is the
    heavy-ball coefficient of "sgd" and is 0 by default. batch_size=None
    trains on the full training set at every step. The defaults are the modular
    addition task, the centred predictor and "sgd", which every run saved
    before 11 October 2026 used.

    If the training loss at a checkpoint is not finite, the run stops there
    without recording that checkpoint. The returned dictionary then has
    diverged set to True and diverged_step set to that step.

    The keys added on 11 October 2026 change nothing when left at their
    defaults. parameterisation "deep_standard" or "deep_ntk" with depth set
    trains a DeepBiasFreeMLP. With kernel_method "closed_form" such a run
    also records, for each layer l, the centred alignment A_Wl of that
    layer's kernel term, its log norm ratio S_Wl to its value at step 0, and
    its share share_Wl of the trace of the kernel. lr_schedule chooses a
    learning rate that changes during training, under "sgd" only.
    "curvature" recomputes the top Gauss-Newton eigenvalue lam_t of the loss
    on the full training set every lr_update_interval steps and sets
    eta_t = eta (lam_0 / lam_t). "norm" sets
    eta_t = eta (||theta_0|| / ||theta_t||)^lr_norm_power after every step.
    Both cap the factor on eta at lr_cap. The per-step decay eta_kappa is
    kept fixed when eta changes. With a schedule the history records eta,
    the rate for the steps after the checkpoint, and under "curvature" also
    lam, the latest eigenvalue. These extra history keys appear only for
    the runs they describe, so the history of every other run is unchanged.

    dtype, added on 18 October 2026, names the torch floating point type of
    the data and the model. Its default "float32" is the type every earlier
    run was trained in.
    """
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown configuration keys: {sorted(unknown)}")
    cfg = normalise_checkpoints({**DEFAULTS, **overrides})
    out_dir = Path(results_dir) if results_dir is not None else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.fp32_precision = cfg["fp32_precision"]
    (X_train, y_train), (X_test, y_test) = make_dataset(cfg)
    data = [t.to(DEVICE, getattr(torch, cfg["dtype"])) for t in (X_train, y_train, X_test, y_test)]

    torch.manual_seed(cfg["seed"])
    model = new_model(cfg, data[0].shape[1], data[1].shape[1]).to(DEVICE, getattr(torch, cfg["dtype"]))
    run = Run(name, cfg, model, data, on_checkpoint)

    lr = run.lr
    weight_decay = cfg["eta_kappa"] / lr
    # AdamW multiplies the weights by 1 - lr weight_decay each step, so eta_kappa
    # is the per-step decay under both optimisers.
    if cfg["optimizer"] == "sgd":
        run.optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                        momentum=cfg["momentum"])
    elif cfg["optimizer"] == "adamw":
        run.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer {cfg['optimizer']!r}")
    optimizer = run.optimizer
    loss_fn = nn.MSELoss()
    schedule = cfg["lr_schedule"]

    # With batch_size set, each step takes the next batch_size training points
    # of a random order that is drawn again once fewer remain. One step is
    # then one minibatch, not one epoch.
    batch_size = cfg["batch_size"]
    batch_gen = torch.Generator().manual_seed(cfg["seed"])
    order = []

    t0 = time.time()
    diverged_step = None
    if not run.evaluate(0):
        diverged_step = 0
    for step in range(1, cfg["steps"] + 1):
        if diverged_step is not None:
            break
        optimizer.zero_grad()
        if batch_size is None:
            loss = loss_fn(run.predict(run.X_train, run.f0_train), run.y_train)
        else:
            if len(order) < batch_size:
                order = torch.randperm(len(run.X_train), generator=batch_gen).tolist()
            idx, order = order[:batch_size], order[batch_size:]
            loss = loss_fn(run.predict(run.X_train[idx], run.f0_train[idx]), run.y_train[idx])
        loss.backward()
        optimizer.step()
        if schedule == "norm":
            run.set_lr((run.norm_0 / run.current_norm()) ** cfg["lr_norm_power"])
        elif schedule == "curvature" and step % cfg["lr_update_interval"] == 0:
            run.lr_state["lam"] = run.curvature_now(step)
            run.set_lr(run.lr_state["lam_0"] / run.lr_state["lam"] if run.lr_state["lam"] > 0 else float("inf"))
        if run.is_checkpoint(step):
            if not run.evaluate(step):
                diverged_step = step
                break
            if verbose and step % (cfg["eval_interval"] * 40) == 0:
                h = run.history
                print(f"{name}: step {step:6d}  train {h['train_loss'][-1]:.2e}  "
                      f"test {h['test_loss'][-1]:.2e}  S {h['S_t'][-1]:+.2f}  "
                      f"R {h['R_t'][-1]:.3f}  Rc {h['R_c'][-1]:.3f}  A {h['A_t'][-1]:.3f}")
    return run.finish(time.time() - t0, diverged_step, out_dir, verbose)


def new_model(cfg, input_dim, output_dim):
    """Return the fresh model of a configuration. The caller sets the seed before calling this."""
    if cfg["depth"] is None:
        return build_model(cfg["parameterisation"], input_dim, cfg["hidden_dim"], output_dim,
                           cfg["activation"], cfg["init_scale"])
    return build_model(cfg["parameterisation"], input_dim, cfg["hidden_dim"], output_dim,
                       cfg["activation"], cfg["init_scale"], depth=cfg["depth"])


class Run:
    """The record of one run: its probe, its values at step 0, its history and its learning rate.

    train_run trains one run and train_fused trains several at once. Both
    keep one Run for each run, and Run.evaluate records a checkpoint with
    the code train_run has always used, so the history of a run does not
    depend on how its gradient steps were taken. The model is the run's own
    module. Under train_fused its parameters are views into the stacked
    parameters of all the runs.
    """

    def __init__(self, name, cfg, model, data, on_checkpoint=None, curvature=True):
        self.name, self.cfg, self.model, self.on_checkpoint = name, cfg, model, on_checkpoint
        self.X_train, self.y_train, self.X_test, self.y_test = data
        self.alpha = cfg["alpha"]
        self.optimizer = None

        # The probe set is fixed for the run. select_probe says which pairs each choice of probe takes.
        self.x_probe, self.y_probe, probe_is_test = select_probe(self.X_train, self.y_train, self.X_test,
                                                                 self.y_test, cfg["probe"], cfg["probe_size"])
        self.parts = [~probe_is_test, probe_is_test]

        self.model_0 = copy.deepcopy(model)
        for q in self.model_0.parameters():
            q.requires_grad = False
        self.lr = cfg["eta_0"] / self.alpha ** 2
        self.loss_fn = nn.MSELoss()
        with torch.no_grad():
            self.f0_train, self.f0_test = self.model_0(self.X_train), self.model_0(self.X_test)

        self.params_0 = torch.cat([q.detach().flatten() for q in model.parameters()]).clone()
        self.norm_0 = torch.linalg.norm(self.params_0).item()
        # A DeepBiasFreeMLP with the closed form also records per-layer terms.
        self.layer_terms = isinstance(model, DeepBiasFreeMLP) and cfg["kernel_method"] == "closed_form"
        self.K_0, self.layer_0 = self.kernel_now()

        # The learning-rate schedule, off by default. lr_state holds the rate in
        # force and, under "curvature", the eigenvalues.
        schedule = cfg["lr_schedule"]
        self.lr_state = dict(lr=self.lr)
        if schedule is not None:
            if cfg["optimizer"] != "sgd":
                raise ValueError("lr_schedule is implemented for sgd only")
            if schedule not in ("curvature", "norm"):
                raise ValueError(f"unknown lr_schedule {schedule!r}")
        if schedule == "curvature" and curvature:
            self.lr_state["lam_0"] = self.lr_state["lam"] = self.curvature_now(0)

        self.history = {k: [] for k in HISTORY_KEYS}
        self.saved_steps, self.saved_kernels = [], []
        self.checkpoints = None if cfg["checkpoint_steps"] is None else set(cfg["checkpoint_steps"])

    # With centred=False the initial function is kept, as in Liu, Michaud and
    # Tegmark (2023), whose large initialisations start from a large random output.
    def predict(self, X, f0):
        return self.alpha * (self.model(X) - f0) if self.cfg["centred"] else self.alpha * self.model(X)

    def kernel_now(self):
        if self.layer_terms:
            terms = entk_closed_form_deep(self.model, self.x_probe, parts=True)
            return sum(terms).to(self.X_train.dtype), terms
        return probe_kernel(self.model, self.x_probe, self.cfg["kernel_method"]).to(self.X_train.dtype), None

    def current_norm(self):
        with torch.no_grad():
            return math.sqrt(sum((q.detach() ** 2).sum().item() for q in self.model.parameters()))

    def curvature_now(self, step):
        self.model.eval()
        f0 = self.f0_train if self.cfg["centred"] else None
        lam = gauss_newton_top_eigenvalue(self.model, self.X_train, alpha=self.alpha, f0=f0,
                                          seed=self.cfg["seed"] + step)
        self.model.train()
        return lam

    def set_lr(self, factor):
        new_lr = self.lr * min(factor, self.cfg["lr_cap"])
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = new_lr
                g["weight_decay"] = self.cfg["eta_kappa"] / new_lr
        self.lr_state["lr"] = new_lr

    def is_checkpoint(self, step):
        if self.checkpoints is None:
            return step % self.cfg["eval_interval"] == 0
        return step in self.checkpoints

    def evaluate(self, step):
        """Record one checkpoint. Return False, and record nothing, if the training loss is not finite."""
        model, history, cfg = self.model, self.history, self.cfg
        model.eval()
        with torch.no_grad():
            tr, te = self.predict(self.X_train, self.f0_train), self.predict(self.X_test, self.f0_test)
            train_loss, test_loss = self.loss_fn(tr, self.y_train).item(), self.loss_fn(te, self.y_test).item()
            if not math.isfinite(train_loss):
                model.train()
                return False
            train_acc = (tr.argmax(1) == self.y_train.argmax(1)).float().mean().item()
            test_acc = (te.argmax(1) == self.y_test.argmax(1)).float().mean().item()
            params_t = torch.cat([q.flatten() for q in model.parameters()])
            weight_norm = torch.linalg.norm(params_t).item()
            param_dist = (torch.linalg.norm(params_t - self.params_0) / self.norm_0).item()
        K_t, layer_t = self.kernel_now()
        K_0, y_probe = self.K_0, self.y_probe
        S_t, R_t = compute_scale_and_rotation(K_0, K_t)
        S_c, R_c = compute_centred_scale_and_rotation(K_0, K_t)
        A_t = compute_task_alignment(K_t, y_probe)
        A_u = uncentred_alignment(K_t, y_probe)
        K_norm = torch.linalg.norm(K_t, ord="fro").item()
        if K_norm > 0:
            yKy = torch.sum(y_probe * (torch.linalg.pinv(K_t, rtol=1e-6) @ y_probe)).item()
        else:
            yKy = float("nan")
        # The same terms on the training part and on the test part of the probe, in the order of PART_KEYS.
        part_terms = []
        for mask in self.parts:
            if int(mask.sum()) < 2:
                part_terms += [float("nan")] * 4
                continue
            K_p, K_p0, y_p = K_t[mask][:, mask], K_0[mask][:, mask], y_probe[mask]
            part_terms += [*compute_centred_scale_and_rotation(K_p0, K_p),
                           compute_task_alignment(K_p, y_p), uncentred_alignment(K_p, y_p)]
        for k, v in zip(HISTORY_KEYS, [step, train_loss, test_loss, train_acc, test_acc,
                                       S_t, R_t, S_c, R_c, A_t, A_u, param_dist, weight_norm,
                                       yKy, K_norm, *part_terms]):
            history[k].append(v)
        extra = {}
        if self.layer_terms:
            trace_K = sum(torch.trace(T) for T in layer_t).item()
            for l, (T, T0) in enumerate(zip(layer_t, self.layer_0), start=1):
                extra[f"A_W{l}"] = compute_task_alignment(T.to(y_probe.dtype), y_probe)
                n_t, n_0 = torch.linalg.norm(T).item(), torch.linalg.norm(T0).item()
                extra[f"S_W{l}"] = math.log(n_t / n_0) if n_t > 0 and n_0 > 0 else float("nan")
                extra[f"share_W{l}"] = torch.trace(T).item() / trace_K if trace_K > 0 else float("nan")
        schedule = cfg["lr_schedule"]
        if schedule is not None:
            extra["eta"] = self.lr_state["lr"]
            if schedule == "curvature":
                extra["lam"] = self.lr_state["lam"]
        for k, v in extra.items():
            history.setdefault(k, []).append(v)
        if step % cfg["kernel_save_interval"] == 0:
            self.saved_steps.append(step)
            self.saved_kernels.append(K_t.cpu().numpy().astype(np.float32))
        if self.on_checkpoint is not None:
            # The row holds every history value of this checkpoint, the extra keys of a
            # deep or scheduled run included. For other runs that is HISTORY_KEYS alone.
            self.on_checkpoint(step, model, K_t, {k: v[-1] for k, v in history.items()})
        model.train()
        return True

    def finish(self, seconds, diverged_step, out_dir, verbose):
        """Write the run's JSON and kernels to out_dir and return the run dictionary."""
        name = self.name
        run = dict(config=dict(name=name, **self.cfg), history=self.history, init_weight_norm=self.norm_0,
                   seconds=seconds, diverged=diverged_step is not None, diverged_step=diverged_step)
        (out_dir / f"{name}.json").write_text(json.dumps(run))
        if self.saved_kernels:
            np.savez_compressed(out_dir / f"{name}_kernels.npz",
                                steps=np.asarray(self.saved_steps), K=np.stack(self.saved_kernels))
        if verbose:
            note = f", diverged at step {diverged_step}" if diverged_step is not None else ""
            print(f"{name}: done in {run['seconds']:.0f} s{note}, saved to {out_dir / name}.json")
        return run


# Configuration keys that every run of train_fused must share, because they fix
# the shapes of the fused computation, the steps it takes or the checkpoints.
FUSED_SHARED = ("parameterisation", "p", "hidden_dim", "train_fraction", "activation", "steps", "eval_interval",
                "probe_size", "kernel_save_interval", "probe", "checkpoint_steps", "kernel_method", "dataset",
                "n_train", "centred", "optimizer", "batch_size", "momentum", "depth", "lr_schedule",
                "lr_update_interval", "lr_cap", "dtype", "fp32_precision")


def train_fused(names, configs, on_checkpoints=None, results_dir=None, verbose=True, device=None,
                on_fused_checkpoint=None):
    """Train several runs at once as one horizontally fused model, and return the dictionary of each run.

    names and configs give the runs that train_run would train one at a time,
    and each run returns the dictionary and writes the files that train_run
    returns and writes. on_checkpoints, if given, holds one on_checkpoint
    callback for each run. The runs must agree on every key in
    FUSED_SHARED. They may differ in the seeds, the initial scale, alpha,
    eta_0 and eta_kappa. The optimiser must be plain SGD without momentum,
    and the schedule None or "curvature".

    The parameters of the runs are stacked with
    torch.func.stack_module_state and trained together under vmap. This is
    the model-ensembling pattern of the PyTorch tutorials, and the
    horizontally fused training of Wang et al. (2021), arXiv:2102.02344,
    Section 3, in which the learning rate and the decay of each model enter
    the fused step as one entry of a vector. Each run keeps its own data
    draw, initial weights and minibatch order. A run's minibatch is drawn as
    train_run draws it: the next batch_size points of a permutation from
    the run's own generator, which is drawn again when fewer than batch_size
    points remain. The step is that of torch.optim.SGD with weight decay
    eta_kappa / eta, so the parameters move by eta times the gradient plus
    eta_kappa times the parameters. The curvature schedule takes the
    eigenvalue of every run from one vmapped power iteration,
    fused_gauss_newton_top_eigenvalue.

    The device defaults to the accelerator that
    torch.accelerator.current_accelerator reports, or the CPU. On a CUDA
    device the step is compiled with torch.compile in its default mode and
    captured as one CUDA graph after one warm-up step on a side stream, as
    the CUDA Graphs section of the PyTorch CUDA semantics notes describes
    whole-network capture. A replay then launches the whole step at once.
    The permutations and the learning rates, which change between steps,
    are copied into the static tensors the graph reads.

    Checkpoints are recorded run by run with Run.evaluate, the code of
    train_run. A run whose training loss stops being finite records nothing
    after that checkpoint, as under train_run. Its parameters keep taking
    fused steps. vmap keeps the computation of every run separate, so this
    cannot change the other runs.
    """
    if on_checkpoints is None:
        on_checkpoints = [None] * len(names)
    if not (len(names) == len(configs) == len(on_checkpoints)) or not names:
        raise ValueError("train_fused needs one name, one configuration and one callback for each run")
    cfgs = []
    for overrides in configs:
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown configuration keys: {sorted(unknown)}")
        cfgs.append(normalise_checkpoints({**DEFAULTS, **overrides}))
    for key in FUSED_SHARED:
        if len({json.dumps(c[key]) for c in cfgs}) > 1:
            raise ValueError(f"the runs of train_fused must share {key!r}")
    shared = cfgs[0]
    if shared["optimizer"] != "sgd" or shared["momentum"] != 0.0:
        raise ValueError("train_fused takes plain SGD steps without momentum only")
    schedule = shared["lr_schedule"]
    if schedule not in (None, "curvature"):
        raise ValueError(f"train_fused does not implement lr_schedule {schedule!r}")
    if device is None:
        device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
    device = torch.device(device)
    dtype = getattr(torch, shared["dtype"])
    # fp32_precision is the setting of torch.backends.cuda.matmul.fp32_precision: "ieee", or "tf32" for the
    # TensorFloat32 tensor cores the PyTorch numerical accuracy notes describe.
    torch.backends.cuda.matmul.fp32_precision = shared["fp32_precision"]
    out_dir = Path(results_dir) if results_dir is not None else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, test_set = [], None
    for name, cfg, hook in zip(names, cfgs, on_checkpoints):
        (X_train, y_train), (X_test, y_test) = make_dataset(cfg)
        # Every run of the same dataset has the same test set, so it is held on the device once.
        if test_set is None or not (torch.equal(X_test, test_set[0]) and torch.equal(y_test, test_set[1])):
            test_set = (X_test, y_test, X_test.to(device, dtype), y_test.to(device, dtype))
        data = [X_train.to(device, dtype), y_train.to(device, dtype), test_set[2], test_set[3]]
        torch.manual_seed(cfg["seed"])
        model = new_model(cfg, data[0].shape[1], data[1].shape[1]).to(device, dtype)
        runs.append(Run(name, cfg, model, data, hook, curvature=False))
    M = len(runs)

    params = {k: v.detach().clone() for k, v in stack_module_state([r.model for r in runs])[0].items()}
    for m, run in enumerate(runs):
        for k, v in params.items():
            owner, _, leaf = k.rpartition(".")
            setattr(run.model.get_submodule(owner), leaf, nn.Parameter(v[m]))
    X = torch.stack([r.X_train for r in runs])
    Y = torch.stack([r.y_train for r in runs])
    for m, run in enumerate(runs):
        run.X_train, run.y_train = X[m], Y[m]
    centred = shared["centred"]
    F0 = torch.stack([r.f0_train for r in runs]) if centred else None
    alpha = torch.tensor([r.alpha for r in runs], dtype=X.dtype, device=device)
    eta_kappa = torch.tensor([c["eta_kappa"] for c in cfgs], dtype=X.dtype, device=device)
    lr_v = torch.tensor([r.lr_state["lr"] for r in runs], dtype=X.dtype, device=device)
    wd_v = eta_kappa / lr_v
    base = copy.deepcopy(runs[0].model).to("meta")
    if schedule == "curvature":
        # The eigenvalue at step 0 that Run.curvature_now(0) would give each run, for all runs at once.
        lams = fused_gauss_newton_top_eigenvalue(base, params, X, alpha, f0=F0, seeds=[c["seed"] for c in cfgs])
        for run, lam in zip(runs, lams):
            run.lr_state["lam_0"] = run.lr_state["lam"] = lam

    def fnet(p, x):
        return functional_call(base, (p,), (x,))

    def loss(p, x, y, f0, a):
        out = a * (fnet(p, x) - f0) if f0 is not None else a * fnet(p, x)
        return nn.functional.mse_loss(out, y)

    fused_grad = vmap(grad(loss), in_dims=(0, 0, 0, 0 if centred else None, 0))
    n = X.shape[1]
    batch = n if shared["batch_size"] is None else shared["batch_size"]
    per_epoch = n // batch
    perm = torch.arange(n, device=device).expand(M, n).clone()
    counter = torch.zeros((), dtype=torch.long, device=device)
    offsets = torch.arange(batch, device=device)
    gens = [torch.Generator().manual_seed(c["seed"]) for c in cfgs]

    def advance():
        idx = perm[:, (counter % per_epoch) * batch + offsets]
        xb = torch.take_along_dim(X, idx[..., None], dim=1)
        yb = torch.take_along_dim(Y, idx[..., None], dim=1)
        f0b = torch.take_along_dim(F0, idx[..., None], dim=1) if centred else None
        grads = fused_grad(params, xb, yb, f0b, alpha)
        for k, p in params.items():
            shape = (-1,) + (1,) * (p.dim() - 1)
            p.sub_(lr_v.view(shape) * (grads[k] + wd_v.view(shape) * p))
        counter.add_(1)

    graph, captured = None, False
    if device.type == "cuda":
        advance = torch.compile(advance)
        graph = torch.cuda.CUDAGraph()

    def take_step():
        nonlocal captured
        if graph is None:
            advance()
        elif not captured:
            # The real step warms the workload up on a side stream once, then it is captured.
            side = torch.Stream(device)
            side.wait_stream(torch.accelerator.current_stream())
            with side:
                advance()
            torch.accelerator.current_stream().wait_stream(side)
            with torch.cuda.graph(graph):
                advance()
            captured = True
        else:
            graph.replay()

    live, diverged = [True] * M, [None] * M
    # The closed-form deep kernel lets every run's checkpoint be computed at once, as tensors with a leading
    # run dimension, with one copy to the host per checkpoint. Other models are recorded run by run.
    batched = all(r.layer_terms for r in runs) and all(
        all(torch.equal(a, b) for a, b in zip(r.parts, runs[0].parts)) for r in runs)
    if batched:
        x_probe, y_probe = torch.stack([r.x_probe for r in runs]), torch.stack([r.y_probe for r in runs])
        K_0 = torch.stack([r.K_0 for r in runs])
        layer_0 = [torch.stack(T) for T in zip(*[r.layer_0 for r in runs])]
        theta_0 = torch.stack([r.params_0 for r in runs])
        norm_0 = torch.tensor([r.norm_0 for r in runs], dtype=X.dtype, device=device)
        F0_test = torch.stack([r.f0_test for r in runs]) if centred else None
        X_test, y_test = runs[0].X_test, runs[0].y_test
        factors = runs[0].model.factors
        layer_terms = vmap(lambda Ws, x: closed_form_terms_deep(Ws, factors, x))

    def record(step):
        if not batched:
            for m, run in enumerate(runs):
                if live[m] and not run.evaluate(step):
                    live[m], diverged[m] = False, step
            return
        with torch.no_grad():
            out_train = vmap(fnet)(params, X)
            out_test = vmap(fnet, in_dims=(0, None))(params, X_test)
            if centred:
                out_train, out_test = out_train - F0, out_test - F0_test
            tr, te = alpha[:, None, None] * out_train, alpha[:, None, None] * out_test
            values = dict(train_loss=((tr - Y) ** 2).mean((1, 2)), test_loss=((te - y_test) ** 2).mean((1, 2)),
                          train_acc=(tr.argmax(-1) == Y.argmax(-1)).to(X.dtype).mean(1),
                          test_acc=(te.argmax(-1) == y_test.argmax(-1)).to(X.dtype).mean(1))
            theta = torch.cat([p.flatten(1) for p in params.values()], 1)
            values["weight_norm"] = torch.linalg.vector_norm(theta, dim=1)
            values["param_dist"] = torch.linalg.vector_norm(theta - theta_0, dim=1) / norm_0
        terms = layer_terms(list(base_weights(params)), x_probe)
        K_t = sum(terms).to(X.dtype)
        values["S_t"], values["R_t"] = batched_scale_and_rotation(K_0, K_t)
        values["S_c"], values["R_c"] = batched_scale_and_rotation(batched_centre(K_0), batched_centre(K_t))
        values["A_t"] = batched_task_alignment(K_t, y_probe)
        values["A_u"] = batched_uncentred_alignment(K_t, y_probe)
        values["K_norm"] = torch.linalg.matrix_norm(K_t)
        values["yKy"] = torch.where(values["K_norm"] > 0, (y_probe * (torch.linalg.pinv(K_t, rtol=1e-6) @ y_probe))
                                    .sum((1, 2)), torch.nan)
        for part, mask in zip(("train", "test"), runs[0].parts):
            names = [f"{term}_{part}" for term in ("S_c", "R_c", "A_t", "A_u")]
            if int(mask.sum()) < 2:
                values.update({k: torch.full_like(values["K_norm"], torch.nan) for k in names})
                continue
            K_p, K_p0, y_p = K_t[:, mask][:, :, mask], K_0[:, mask][:, :, mask], y_probe[:, mask]
            S_p, R_p = batched_scale_and_rotation(batched_centre(K_p0), batched_centre(K_p))
            values.update(zip(names, (S_p, R_p, batched_task_alignment(K_p, y_p),
                                      batched_uncentred_alignment(K_p, y_p))))
        traces = [T.diagonal(dim1=-2, dim2=-1).sum(-1) for T in terms]
        trace_K = sum(traces)
        for l, (T, T0, tr_l) in enumerate(zip(terms, layer_0, traces), start=1):
            values[f"A_W{l}"] = batched_task_alignment(T.to(y_probe.dtype), y_probe)
            n_t, n_0 = torch.linalg.matrix_norm(T), torch.linalg.matrix_norm(T0)
            values[f"S_W{l}"] = torch.where((n_t > 0) & (n_0 > 0), torch.log(n_t / n_0), torch.nan)
            values[f"share_W{l}"] = torch.where(trace_K > 0, tr_l / trace_K, torch.nan)
        names = list(values)
        host = torch.stack([values[k].to(torch.float64) for k in names]).cpu().tolist()
        saving = step % shared["kernel_save_interval"] == 0
        K_host = K_t.cpu().numpy().astype(np.float32) if saving else None
        rows = [None] * M
        for m, run in enumerate(runs):
            if not live[m]:
                continue
            row = {k: host[i][m] for i, k in enumerate(names)}
            if not math.isfinite(row["train_loss"]):
                live[m], diverged[m] = False, step
                continue
            row["step"] = step
            if schedule is not None:
                row["eta"] = run.lr_state["lr"]
                if schedule == "curvature":
                    row["lam"] = run.lr_state["lam"]
            for k in HISTORY_KEYS:
                run.history[k].append(row[k])
            for k in [k for k in row if k not in HISTORY_KEYS]:
                run.history.setdefault(k, []).append(row[k])
            if saving:
                run.saved_steps.append(step)
                run.saved_kernels.append(K_host[m])
            rows[m] = {k: v[-1] for k, v in run.history.items()}
            if on_fused_checkpoint is None and run.on_checkpoint is not None:
                run.on_checkpoint(step, run.model, K_t[m], rows[m])
        if on_fused_checkpoint is not None:
            on_fused_checkpoint(step, [m for m in range(M) if live[m]], params, K_t, terms, rows, fnet, factors)

    def base_weights(p):
        return (p[k] for k in params)

    t0 = time.time()
    record(0)
    for step in range(1, shared["steps"] + 1):
        if not any(live):
            break
        if shared["batch_size"] is not None and (step - 1) % per_epoch == 0:
            perm.copy_(torch.stack([torch.randperm(n, generator=g) for g in gens]))
        take_step()
        if schedule == "curvature" and step % shared["lr_update_interval"] == 0:
            lams = fused_gauss_newton_top_eigenvalue(base, params, X, alpha, f0=F0,
                                                     seeds=[c["seed"] + step for c in cfgs])
            for run, lam in zip(runs, lams):
                run.lr_state["lam"] = lam
                run.set_lr(run.lr_state["lam_0"] / lam if lam > 0 else float("inf"))
            lr_v.copy_(torch.tensor([r.lr_state["lr"] for r in runs], dtype=X.dtype))
            wd_v.copy_(eta_kappa / lr_v)
        if runs[0].is_checkpoint(step):
            record(step)
    seconds = time.time() - t0
    return [run.finish(seconds, d, out_dir, verbose) for run, d in zip(runs, diverged)]


def fused_gauss_newton_top_eigenvalue(base, params, X, alpha, f0=None, seeds=None, iters=50):
    """The eigenvalue of gauss_newton_top_eigenvalue for every run of a fused model, from one vmapped power iteration.

    base is a module with the runs' architecture, params their stacked
    parameters, X their stacked inputs and alpha their output scales. f0,
    if given, holds their stacked initial outputs on X. Run m starts from
    the vector that gauss_newton_top_eigenvalue draws from a generator
    seeded with seeds[m], so each run takes the iteration it would take
    alone. The whole iteration of one run is vmapped over the runs, with its
    vjp built once per run. torch.func.linearize has no batching rule under
    vmap in PyTorch 2.14 (aten::_make_dual), so the Jacobian-vector product
    is a torch.func.jvp at each iteration. Returns a list with one float for
    each run.
    """
    M = next(iter(params.values())).shape[0]
    seeds = [0] * M if seeds is None else seeds
    device = next(iter(params.values())).device
    starts = []
    for s in seeds:
        gen = torch.Generator().manual_seed(s)
        starts.append({k: torch.randn(t.shape[1:], generator=gen, dtype=t.dtype) for k, t in params.items()})
    v = {k: torch.stack([st[k] for st in starts]).to(device) for k in params}

    def power_iteration(p, x, a, f, u):
        def predict(q):
            out = functional_call(base, (q,), (x,))
            return a * (out - f) if f is not None else a * out

        _, pullback = torch.func.vjp(predict, p)
        lam = torch.zeros((), dtype=x.dtype, device=x.device)
        for _ in range(iters):
            norm = torch.sqrt(sum((t ** 2).sum() for t in u.values()))
            u = {k: t / norm for k, t in u.items()}
            _, Jv = torch.func.jvp(predict, (p,), (u,))
            (Gv,) = pullback(2.0 * Jv / Jv.numel())
            lam = sum((Gv[k] * u[k]).sum() for k in u)
            u = Gv
        return lam

    return vmap(power_iteration, in_dims=(0, 0, 0, 0 if f0 is not None else None, 0))(
        params, X, alpha, f0, v).tolist()


def run_or_load(name, rerun=False, verbose=True, on_checkpoint=None, results_dir=None, **overrides):
    """Load a saved run if its configuration matches, otherwise train it.

    A saved run that lacks a key in LEGACY_VALUES is compared as if it had
    the value given there, which is the value it was trained with.
    on_checkpoint is passed to train_run, so it is called only when the run
    is trained. Loading a saved run does not call it. results_dir is where
    the run is looked for and written, RESULTS by default.
    """
    out_dir = Path(results_dir) if results_dir is not None else RESULTS
    path = out_dir / f"{name}.json"
    if path.exists() and not rerun:
        run = json.loads(path.read_text())
        saved = {**LEGACY_VALUES, **{k: v for k, v in run["config"].items() if k != "name"}}
        wanted = normalise_checkpoints({**DEFAULTS, **overrides})
        if saved == wanted and "R_c" in run["history"]:
            if verbose:
                print(f"{name}: loaded from results/{name}.json")
            return run
        if verbose:
            print(f"{name}: saved run does not match, training again")
    return train_run(name, verbose=verbose, on_checkpoint=on_checkpoint, results_dir=results_dir, **overrides)


def load_kernels(name, results_dir=None):
    """Return the saved probe kernels of a run as (steps, K) arrays."""
    data = np.load((Path(results_dir) if results_dir is not None else RESULTS) / f"{name}_kernels.npz")
    return data["steps"], data["K"]


# =====================================================================
# Grokking time
# =====================================================================
def first_crossing(steps, values, tau):
    """Return the first recorded step at which values fall below tau, or None."""
    below = np.where(np.asarray(values) < tau)[0]
    return int(steps[below[0]]) if len(below) else None


def grokking_time(run, tau_train, tau_test):
    """Return the crossing steps and the grokking time, with a censoring flag.

    The grokking time is the number of steps between the training loss first
    falling below tau_train and the test loss first falling below tau_test.
    If the test loss never crosses, the run is right censored and the time is
    reported as a lower bound.
    """
    h = run["history"]
    steps = np.asarray(h["step"])
    t_train = first_crossing(steps, h["train_loss"], tau_train)
    t_test = first_crossing(steps, h["test_loss"], tau_test)
    last = int(steps[-1])
    if t_train is None:
        return dict(t_train=None, t_test=t_test, t_grok=None, censored=True, last_step=last)
    if t_test is None:
        return dict(t_train=t_train, t_test=None, t_grok=last - t_train, censored=True, last_step=last)
    return dict(t_train=t_train, t_test=t_test, t_grok=t_test - t_train, censored=False, last_step=last)


def format_time(g):
    """Return the grokking time as a string, with a censored value marked."""
    if g["t_grok"] is None:
        return "-"
    return f">{g['t_grok']}" if g["censored"] else str(g["t_grok"])


def value_at_step(run, key, step):
    """Return the recorded value of key at the last checkpoint at or before step."""
    steps = np.asarray(run["history"]["step"])
    return run["history"][key][np.where(steps <= step)[0][-1]]


def peak_step(run, key):
    """Return the step at which the recorded value of key is largest."""
    h = run["history"]
    return int(h["step"][int(np.argmax(h[key]))])


def sensitivity_table(run, taus=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    """Print the grokking time for a grid of thresholds."""
    print(f"{'tau_train':>10} {'tau_test':>9} {'t_train':>8} {'t_test':>8} {'t_grok':>10}")
    for tau_tr in taus:
        for tau_te in taus:
            g = grokking_time(run, tau_tr, tau_te)
            t_tr = "-" if g["t_train"] is None else g["t_train"]
            t_te = "-" if g["t_test"] is None else g["t_test"]
            print(f"{tau_tr:>10g} {tau_te:>9g} {str(t_tr):>8} {str(t_te):>8} {format_time(g):>10}")


# =====================================================================
# Plot helpers
# =====================================================================
def _last_step(ax):
    return max((line.get_xdata()[-1] for line in ax.get_lines() if len(line.get_xdata())), default=None)


def tidy_axes(ax):
    """Make the tick labels readable.

    Log axes get ticks at 1, 2 and 5 times each power of ten with plain
    number labels, so a decade shows more than one number. Linear axes whose
    range runs into the thousands get labels with a thousands separator.
    """
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    for axis, scale in [(ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())]:
        if scale == "log":
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        elif scale == "linear":
            lo, hi = axis.get_view_interval()
            if max(abs(lo), abs(hi)) >= 1000:
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def log_step_axis(ax):
    """Log step axis that keeps step zero. The linear region covers 0 to 100.

    The right limit is padded by a quarter of the last step so that markers
    at the end of a run do not sit on the frame.
    """
    ax.set_xscale("symlog", linthresh=100, linscale=1.0)
    last = _last_step(ax)
    ax.set_xlim(left=0, right=None if last is None else 1.25 * last)


def linear_step_axis(ax):
    """Linear step axis from zero with a small pad on the right."""
    last = _last_step(ax)
    ax.set_xlim(left=0, right=None if last is None else 1.05 * last)


def plot_times(ax, x, t, censored, colour, flip=False, ms=6, alpha=0.85, label=None):
    """Draw measured times as circles and censored bounds as triangles.

    A censored grokking time is a lower bound, so its triangle points up
    towards the true value. With flip the triangle points down, which is the
    right direction when the plotted value is the reciprocal of a time. This
    matches the convention of the Tier 0 notebook.
    """
    x, t, censored = np.asarray(x, float), np.asarray(t, float), np.asarray(censored, bool)
    ax.plot(x[~censored], t[~censored], "o", color=colour, ms=ms, alpha=alpha, markeredgecolor="white",
            markeredgewidth=0.5, label=label, lw=0)
    ax.plot(x[censored], t[censored], "v" if flip else "^", color=colour, ms=ms + 1, alpha=alpha,
            markeredgecolor="white", markeredgewidth=0.5, lw=0)


def mark_crossings(ax, g):
    """Dotted line at the train crossing and dashed line at the test crossing."""
    if g["t_train"] is not None:
        ax.axvline(g["t_train"], color=GRAY, lw=1, ls=":")
    if g["t_test"] is not None:
        ax.axvline(g["t_test"], color=GRAY, lw=1, ls="--")


def plot_losses(run, g=None, log_x=True, title=None):
    """Train and test loss on the left, accuracy on the right."""
    h = run["history"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(h["step"], h["train_loss"], color=BLUE, label="train")
    ax.plot(h["step"], h["test_loss"], color=ORANGE, label="test")
    ax.axhline(1 / run["config"]["p"], color=GRAY, lw=1, ls="-.", label="chance, 1/p")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("mean squared error")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.plot(h["step"], h["train_acc"], color=BLUE, label="train")
    ax.plot(h["step"], h["test_acc"], color=ORANGE, label="test")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower right")
    for ax in axes:
        if log_x:
            log_step_axis(ax)
        if g is not None:
            mark_crossings(ax, g)
        tidy_axes(ax)
    fig.suptitle(title or run["config"]["name"])
    fig.tight_layout()
    return fig


YLABELS = {"train_loss": "mean squared error", "test_loss": "mean squared error", "train_acc": "accuracy",
           "test_acc": "accuracy", "weight_norm": "weight norm", "S_c": "log of kernel norm ratio",
           "S_t": "log of kernel norm ratio", "R_c": "one minus cosine to initial kernel",
           "R_t": "one minus cosine to initial kernel", "A_t": "centred alignment",
           "param_dist": "relative parameter movement", "yKy": "y' K^+ y", "K_norm": "kernel norm"}


def plot_series(runs, keys, labels, colours, titles, log_y=(), legend_panel=0, suptitle=None,
                crossings=None, figsize=None, log_x=True, mark_train=False):
    """Overlay one history key per panel for several runs.

    runs, labels and colours are parallel lists. keys and titles are
    parallel lists, one per panel. crossings is an optional list of grokking
    time dictionaries parallel to runs; if given, the test crossing of each
    run is marked with a filled dot on every panel, and with mark_train the
    train crossing is marked with an open dot. log_x chooses a log step axis,
    which shows the early dynamics, or a linear one, which shows the late
    crossings.
    """
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=figsize or (4.4 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, key, title in zip(axes, keys, titles):
        for i, (run, label, colour) in enumerate(zip(runs, labels, colours)):
            h = run["history"]
            ax.plot(h["step"], h[key], color=colour, label=label, lw=1.6)
            if crossings is not None:
                t_tr, t_te = crossings[i].get("t_train"), crossings[i].get("t_test")
                if mark_train and t_tr is not None:
                    ax.plot([t_tr], [value_at_step(run, key, t_tr)], "o", ms=7, markerfacecolor="white",
                            markeredgecolor=colour, markeredgewidth=1.6)
                if t_te is not None:
                    ax.plot([t_te], [value_at_step(run, key, t_te)], "o", color=colour, ms=7,
                            markeredgecolor="white", markeredgewidth=1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel(YLABELS.get(key, key))
        if key in log_y:
            ax.set_yscale("log")
        if log_x:
            log_step_axis(ax)
        else:
            linear_step_axis(ax)
        tidy_axes(ax)
    axes[legend_panel].legend()
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


# =====================================================================
# Tier grid: naming, loading and per-cell summaries
# =====================================================================
def cell_name(N, alpha, ek, seed):
    """Name of a grid cell. All grid runs are NTK parameterisation at base rate 100."""
    return f"ntk_N{N}_a{alpha:g}_wd{ek:g}_s{seed}"


def load_cell(N, alpha, ek, seed, steps, eval_interval, verbose=False, probe="mixed"):
    """Load a grid cell from results/, training it first if it is missing.

    The default probe is the mixed probe of the report's Kernel metrics
    section, and select_probe describes it. A cell with another probe has
    the probe in its name, such as ntk_N100_a1_wd0_s0_probetest.
    """
    suffix = "" if probe == "mixed" else f"_probe{probe}"
    return run_or_load(cell_name(N, alpha, ek, seed) + suffix, verbose=verbose, parameterisation="ntk", hidden_dim=N,
                       alpha=alpha, eta_0=100.0, eta_kappa=ek, seed=seed, steps=steps,
                       eval_interval=eval_interval, kernel_save_interval=steps, probe=probe)


def first_step_at_least(run, key, level):
    """First recorded step at which the history value reaches level, or None."""
    h = run["history"]
    idx = np.where(np.asarray(h[key]) >= level)[0]
    return int(h["step"][idx[0]]) if len(idx) else None


def accuracy_crossings(run, test_level=0.95):
    """Crossings defined on accuracy: train accuracy 1 and test accuracy test_level.

    Returns the same keys as grokking_time so the two definitions can be
    swapped. A run whose test accuracy never reaches the level is censored
    and its grokking time is the lower bound given by the run length.
    """
    t_train = first_step_at_least(run, "train_acc", 1.0)
    t_test = first_step_at_least(run, "test_acc", test_level)
    last = int(run["history"]["step"][-1])
    if t_train is None:
        return dict(t_train=None, t_test=t_test, t_grok=None, censored=True, last_step=last)
    if t_test is None:
        return dict(t_train=t_train, t_test=None, t_grok=last - t_train, censored=True, last_step=last)
    return dict(t_train=t_train, t_test=t_test, t_grok=t_test - t_train, censored=False, last_step=last)


def relative_movement(run, step):
    """The usual kernel movement statistic ||K_t - K_0||_F / ||K_0||_F at a step.

    It is recovered from the recorded centred scale and rotation terms
    through Equation 1 of the proposal.
    """
    S, R = value_at_step(run, "S_c", step), value_at_step(run, "R_c", step)
    return float(np.sqrt(max(np.exp(2 * S) + 1 - 2 * np.exp(S) * (1 - R), 0.0)))


def summarise_cell(run, definition="accuracy", tau=1e-2, test_level=0.95):
    """One row of numbers for a cell: the crossings and the kernel terms at the test crossing.

    definition is "accuracy" for the accuracy crossings, with the test
    crossing at test accuracy test_level, or "loss" for the loss thresholds
    tau on both losses. Kernel terms at the crossing are None when the test
    crossing is censored.
    """
    g = accuracy_crossings(run, test_level) if definition == "accuracy" else grokking_time(run, tau, tau)
    h = run["history"]
    t = g["t_test"]
    row = dict(t_train=g["t_train"], t_test=t, t_grok=g["t_grok"], censored=g["censored"],
               last_step=g["last_step"], A_peak=float(max(h["A_t"])), A_peak_step=peak_step(run, "A_t"),
               norm_growth=h["weight_norm"][-1] / run["init_weight_norm"],
               A_at_test=None, S_at_test=None, R_at_test=None, D_at_test=None)
    if t is not None:
        row.update(A_at_test=value_at_step(run, "A_t", t), S_at_test=value_at_step(run, "S_c", t),
                   R_at_test=value_at_step(run, "R_c", t), D_at_test=relative_movement(run, t))
    return row


def fmt(x, digits=4):
    """Format a number for a table, with None shown as a dash."""
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{digits}f}"


# =====================================================================
# Censored regression for the head claim
# =====================================================================
def censored_linear_fit(X, y, censored, side="right", n_boot=300, seed=0, groups=None):
    """Fit y = X beta + normal noise by maximum likelihood when some values of y are censored.

    X is a design matrix with an intercept column, y the response already on
    the scale being fitted, and censored a boolean array. With side "right" a
    censored value is a lower bound and enters through the probability that
    the true value exceeds it, which is the Tobit model the report names.
    With side "left" a censored value is an upper bound and enters through
    the probability that the true value is below it. The reciprocal fit needs
    this, because a lower bound on the grokking time is an upper bound on its
    reciprocal.

    Confidence intervals come from a bootstrap. Without groups, runs are
    resampled from the whole pool. With groups, an array of cell labels, runs
    are resampled within each cell, so every cell appears in every resample.
    Resamples whose design matrix loses rank are skipped. Returns the
    coefficients, their 95 percent intervals, the fitted log standard
    deviation, the number of bootstrap fits used, and the bootstrap draws.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    c = np.asarray(censored, bool)
    if side not in ("right", "left"):
        raise ValueError(side)
    tail = norm.logsf if side == "right" else norm.logcdf

    def nll(params, Xf, yf, cf):
        beta, sigma = params[:-1], np.exp(params[-1])
        mu = Xf @ beta
        return -(norm.logpdf(yf[~cf], mu[~cf], sigma).sum() + tail(yf[cf], mu[cf], sigma).sum())

    def fit(Xf, yf, cf):
        beta0 = np.linalg.lstsq(Xf, yf, rcond=None)[0]
        # A log response starts at sigma 1, as it always has. Other responses, such as a
        # reciprocal time, start at the spread of the least-squares residuals.
        start = np.log(np.std(yf - Xf @ beta0)) if side == "left" else 0.0
        res = minimize(lambda p: nll(p, Xf, yf, cf), np.r_[beta0, start], method="BFGS")
        return res.x

    est = fit(X, y, c)
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is not None:
        members = [np.where(np.asarray(groups) == g)[0] for g in dict.fromkeys(groups)]
    boots = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([m[rng.integers(0, len(m), len(m))] for m in members])
        if np.linalg.matrix_rank(X[idx]) < X.shape[1]:
            continue
        boots.append(fit(X[idx], y[idx], c[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(coef=est[:-1], lo=lo[:-1], hi=hi[:-1], log_sigma=est[-1], n_boot=len(boots),
                boot=boots[:, :-1])


def censored_log_fit(X, t, censored, n_boot=300, seed=0, groups=None):
    """Fit log t = X beta + noise by maximum likelihood with right censoring.

    t holds the grokking times and censored marks the lower bounds. The noise
    is normal on the log scale. This is censored_linear_fit applied to log t,
    and the bootstrap follows the groups argument described there.
    """
    return censored_linear_fit(X, np.log(np.asarray(t, float)), censored, side="right",
                               n_boot=n_boot, seed=seed, groups=groups)


def plot_definitions(run, tau=1e-2, test_level=0.95, title=None):
    """Show the two grokking-time definitions on one run, side by side.

    The left panel draws the training and test losses with the threshold
    tau and marks where each first falls below it. The right panel draws the
    two accuracies with the levels 1 and test_level and marks where each
    first reaches them. On each panel the gap between the memorisation event
    and the generalisation event is shaded; that gap is the grokking time
    under that definition. A missing marker means the event did not happen
    within the run, and the gap is then drawn open to the end of the run.
    """
    h = run["history"]
    steps = np.asarray(h["step"])
    loss = grokking_time(run, tau, tau)
    acc = accuracy_crossings(run, test_level)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    def panel(ax, g, keys, levels, ylabel, log_y):
        for key, colour, label in zip(keys, [BLUE, ORANGE], ["train", "test"]):
            ax.plot(steps, h[key], color=colour, label=label)
        for level in levels:
            ax.axhline(level, color=GRAY, lw=1, ls="-.")
        t0, t1 = g["t_train"], g["t_test"]
        if t0 is not None:
            ax.axvline(t0, color=BLUE, lw=1, ls=":")
            ax.plot([t0], [value_at_step(run, keys[0], t0)], "o", ms=8, markerfacecolor="white", markeredgecolor=BLUE,
                    markeredgewidth=1.6, label="memorisation event")
        if t1 is not None:
            ax.axvline(t1, color=ORANGE, lw=1, ls=":")
            ax.plot([t1], [value_at_step(run, keys[1], t1)], "o", color=ORANGE, ms=8, markeredgecolor="white",
                    label="generalisation event")
        if t0 is not None:
            end = t1 if t1 is not None else steps[-1]
            ax.axvspan(t0, end, color=AQUA, alpha=0.15,
                       label="grokking time" if t1 is not None else "grokking time, open ended")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="best")
        tidy_axes(ax)

    panel(axes[0], loss, ["train_loss", "test_loss"], [tau], "mean squared error", True)
    axes[0].set_title(f"loss definition: both losses crossing {tau:g}", loc="left", fontsize=10)
    panel(axes[1], acc, ["train_acc", "test_acc"], [1.0, test_level], "accuracy", False)
    axes[1].set_title(f"accuracy definition: train accuracy 1, test accuracy {test_level:g}", loc="left", fontsize=10)
    axes[1].set_ylim(0, 1.05)
    fig.suptitle(title or f"How the grokking time is measured on one run, {run['config']['name']}")
    fig.tight_layout()

    def word(t):
        return "never within the run" if t is None else f"step {t}"
    print(f"Loss definition: memorisation {word(loss['t_train'])}, generalisation {word(loss['t_test'])}, "
          f"grokking time {format_time(loss)} steps.")
    print(f"Accuracy definition: memorisation {word(acc['t_train'])}, generalisation {word(acc['t_test'])}, "
          f"grokking time {format_time(acc)} steps.")
    return fig


def plot_sensitivity(run, taus=(5e-2, 3e-2, 2e-2, 1e-2, 5e-3, 3e-3, 1e-3), title=None):
    """Grokking time under the loss definition over a grid of thresholds, as a heat map.

    Rows are the training-loss threshold and columns the test-loss threshold.
    A measured positive grokking time is coloured on a log scale and
    labelled. A negative or zero value, where the test loss crossed no later
    than the training loss, is shown in light grey with its label. A cell
    whose test loss never crosses is a lower bound and is shown in darker
    grey with hatching and a greater-than sign. A cell whose training loss
    never crosses has no grokking time and is blank.
    """
    from matplotlib.colors import LogNorm
    taus = list(taus)
    n = len(taus)
    values = np.full((n, n), np.nan)
    notes = {}
    for i, tau_tr in enumerate(taus):
        for j, tau_te in enumerate(taus):
            g = grokking_time(run, tau_tr, tau_te)
            if g["t_grok"] is None:
                notes[(i, j)] = ("none", "no train\ncrossing")
            elif g["censored"]:
                notes[(i, j)] = ("censored", format_time(g))
            elif g["t_grok"] <= 0:
                notes[(i, j)] = ("nonpositive", str(g["t_grok"]))
            else:
                values[i, j] = g["t_grok"]
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 0.75 * n + 1.5))
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    image = ax.imshow(values, cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax))
    for (i, j), (kind, label) in notes.items():
        if kind == "none":
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color=GRAY)
            continue
        face = "#d9d8d3" if kind == "nonpositive" else "#b5b4ae"
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white",
                                   hatch="///" if kind == "censored" else None, lw=0.5))
        ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
    for i in range(n):
        for j in range(n):
            if not np.isnan(values[i, j]):
                dark = np.log(values[i, j] / vmin) > 0.55 * np.log(vmax / vmin)
                ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if dark else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{t:g}" for t in taus])
    ax.set_xlabel("test-loss threshold"); ax.set_ylabel("training-loss threshold")
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("measured grokking time, steps")
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    cbar.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=10))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    ax.plot([], [], "s", color="#b5b4ae", label="lower bound, test loss never crossed")
    ax.plot([], [], "s", color="#d9d8d3", label="zero or negative, test crossed first")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8, ncol=2)
    ax.set_title(title or "Loss-based grokking time against the two thresholds", loc="left", fontsize=10)
    fig.tight_layout()
    return fig
