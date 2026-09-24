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
from torch.func import functional_call, vmap, grad

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


def build_model(parameterisation, input_dim, hidden_dim, output_dim, activation="relu", init_scale=1.0):
    """Return a fresh model. The caller sets the seed before calling this."""
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


# =====================================================================
# Centred scale and rotation
# =====================================================================
def centre_kernel(K):
    """Return H K H with H = I - (1/n) 1 1^T, Cortes et al. (2012), Lemma 1."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
    return H @ K @ H


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


# =====================================================================
# Training
# =====================================================================
DEFAULTS = dict(parameterisation="mean_field", p=23, hidden_dim=100, alpha=1.0,
                eta_0=100.0, eta_kappa=0.0, steps=60000, eval_interval=250,
                probe_size=256, seed=0, data_seed=42, train_fraction=0.9,
                activation="relu", init_scale=1.0, kernel_save_interval=5000, probe="train",
                checkpoint_steps=None)

# Configuration keys added after some runs were saved, with the value those
# runs were trained with. A saved run that lacks one of them is compared as if
# it had that value.
LEGACY_VALUES = dict(probe="train", checkpoint_steps=None)

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


def probe_pairs(cfg):
    """Return the probe pairs (a, b) of a configuration and the mask of test pairs, as numpy arrays."""
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


def train_run(name, verbose=True, on_checkpoint=None, **overrides):
    """Train one run and return a dictionary with the configuration and history.

    The predictor is the centred and rescaled function of Kumar et al.
    (2024), Appendix 8.1, Equation 7, with learning rate eta_0 over alpha
    squared. Weight decay enters as a factor of one minus eta times kappa per
    step. The model seed is separate from the data seed. The train loss is
    recorded after the update, at the same point as the test loss.

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
    a dictionary of the history values just recorded. Training continues
    from the same model afterwards, so the callback must not change it. The
    callback is not part of the configuration and is not saved.
    """
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown configuration keys: {sorted(unknown)}")
    cfg = normalise_checkpoints({**DEFAULTS, **overrides})
    config = dict(name=name, **cfg)
    p, alpha = cfg["p"], cfg["alpha"]

    (X_train, y_train), (X_test, y_test) = make_modular_addition_dataset(
        p=p, train_fraction=cfg["train_fraction"], seed=cfg["data_seed"])
    X_train, y_train, X_test, y_test = [t.to(DEVICE) for t in (X_train, y_train, X_test, y_test)]

    # The probe set is fixed for the run. select_probe says which pairs each choice of probe takes.
    x_probe, y_probe, probe_is_test = select_probe(X_train, y_train, X_test, y_test,
                                                   cfg["probe"], cfg["probe_size"])
    parts = [~probe_is_test, probe_is_test]

    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["parameterisation"], 2 * p, cfg["hidden_dim"], p,
                        cfg["activation"], cfg["init_scale"]).to(DEVICE)
    model_0 = copy.deepcopy(model)
    for q in model_0.parameters():
        q.requires_grad = False

    lr = cfg["eta_0"] / alpha ** 2
    weight_decay = cfg["eta_kappa"] / lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def predict(X):
        return alpha * (model(X) - model_0(X))

    params_0 = torch.cat([q.detach().flatten() for q in model.parameters()]).clone()
    norm_0 = torch.linalg.norm(params_0).item()
    K_0 = entk(model, x_probe)

    history = {k: [] for k in HISTORY_KEYS}
    saved_steps, saved_kernels = [], []

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            tr, te = predict(X_train), predict(X_test)
            train_loss, test_loss = loss_fn(tr, y_train).item(), loss_fn(te, y_test).item()
            train_acc = (tr.argmax(1) == y_train.argmax(1)).float().mean().item()
            test_acc = (te.argmax(1) == y_test.argmax(1)).float().mean().item()
            params_t = torch.cat([q.flatten() for q in model.parameters()])
            weight_norm = torch.linalg.norm(params_t).item()
            param_dist = (torch.linalg.norm(params_t - params_0) / norm_0).item()
        K_t = entk(model, x_probe)
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
        for mask in parts:
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
        if step % cfg["kernel_save_interval"] == 0:
            saved_steps.append(step)
            saved_kernels.append(K_t.cpu().numpy().astype(np.float32))
        if on_checkpoint is not None:
            on_checkpoint(step, model, K_t, {k: history[k][-1] for k in HISTORY_KEYS})
        model.train()

    checkpoints = None if cfg["checkpoint_steps"] is None else set(cfg["checkpoint_steps"])

    def is_checkpoint(step):
        if checkpoints is None:
            return step % cfg["eval_interval"] == 0
        return step in checkpoints

    t0 = time.time()
    evaluate(0)
    for step in range(1, cfg["steps"] + 1):
        optimizer.zero_grad()
        loss = loss_fn(predict(X_train), y_train)
        loss.backward()
        optimizer.step()
        if is_checkpoint(step):
            evaluate(step)
            if verbose and step % (cfg["eval_interval"] * 40) == 0:
                h = history
                print(f"{name}: step {step:6d}  train {h['train_loss'][-1]:.2e}  "
                      f"test {h['test_loss'][-1]:.2e}  S {h['S_t'][-1]:+.2f}  "
                      f"R {h['R_t'][-1]:.3f}  Rc {h['R_c'][-1]:.3f}  A {h['A_t'][-1]:.3f}")
    run = dict(config=config, history=history, init_weight_norm=norm_0,
               seconds=time.time() - t0)
    (RESULTS / f"{name}.json").write_text(json.dumps(run))
    np.savez_compressed(RESULTS / f"{name}_kernels.npz",
                        steps=np.asarray(saved_steps), K=np.stack(saved_kernels))
    if verbose:
        print(f"{name}: done in {run['seconds']:.0f} s, saved to results/{name}.json")
    return run


def run_or_load(name, rerun=False, verbose=True, on_checkpoint=None, **overrides):
    """Load a saved run if its configuration matches, otherwise train it.

    A saved run that lacks a key in LEGACY_VALUES is compared as if it had
    the value given there, which is the value it was trained with.
    on_checkpoint is passed to train_run, so it is called only when the run
    is trained. Loading a saved run does not call it.
    """
    path = RESULTS / f"{name}.json"
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
    return train_run(name, verbose=verbose, on_checkpoint=on_checkpoint, **overrides)


def load_kernels(name):
    """Return the saved probe kernels of a run as (steps, K) arrays."""
    data = np.load(RESULTS / f"{name}_kernels.npz")
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
def censored_log_fit(X, t, censored, n_boot=300, seed=0):
    """Fit log t = X beta + noise by maximum likelihood with right censoring.

    X is a design matrix with an intercept column, t the grokking times and
    censored a boolean array. A censored time enters through the probability
    that the true value exceeds it, which is the Tobit model the report names.
    The noise is normal on the log scale. Confidence intervals come from a
    bootstrap over cells. Returns the coefficients, their 95 percent
    intervals, and the fitted log standard deviation.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    X = np.asarray(X, float)
    y = np.log(np.asarray(t, float))
    c = np.asarray(censored, bool)

    def nll(params, Xf, yf, cf):
        beta, sigma = params[:-1], np.exp(params[-1])
        mu = Xf @ beta
        return -(norm.logpdf(yf[~cf], mu[~cf], sigma).sum() + norm.logsf(yf[cf], mu[cf], sigma).sum())

    def fit(Xf, yf, cf):
        beta0 = np.linalg.lstsq(Xf, yf, rcond=None)[0]
        res = minimize(lambda p: nll(p, Xf, yf, cf), np.r_[beta0, 0.0], method="BFGS")
        return res.x

    est = fit(X, y, c)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.linalg.matrix_rank(X[idx]) < X.shape[1]:
            continue
        boots.append(fit(X[idx], y[idx], c[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(coef=est[:-1], lo=lo[:-1], hi=hi[:-1], log_sigma=est[-1], n_boot=len(boots))


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
