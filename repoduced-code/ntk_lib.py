"""
Shared code for the grokking experiments in the G2 project.

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

The proposal states that the scale term, the rotation term, and the
alignment are all computed on the centred kernel. RepoducedCode.py computes
the scale and rotation terms on the raw kernel. Both versions are recorded
here so they can be compared.
"""

import copy
import json
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


def build_model(parameterisation, input_dim, hidden_dim, output_dim, activation="relu", init_scale=1.0):
    """Return a fresh model. The caller sets the seed before calling this."""
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
                activation="relu", init_scale=1.0, kernel_save_interval=5000)

HISTORY_KEYS = ["step", "train_loss", "test_loss", "train_acc", "test_acc",
                "S_t", "R_t", "S_c", "R_c", "A_t", "param_dist", "weight_norm",
                "yKy", "K_norm"]


def train_run(name, verbose=True, **overrides):
    """Train one run and return a dictionary with the configuration and history.

    The predictor is the centred and rescaled function of Kumar et al.
    (2024), Appendix 8.1, Equation 7, with learning rate eta_0 over alpha
    squared. Weight decay enters as a factor of one minus eta times kappa per
    step. The model seed is separate from the data seed. The train loss is
    recorded after the update, at the same point as the test loss.

    At each checkpoint the history records the losses, the accuracies, the
    raw scale and rotation terms S_t and R_t, the centred versions S_c and
    R_c, the centred alignment A_t, the relative parameter movement, the
    weight norm, y^T K^+ y on the probe set, and the kernel norm. The probe
    kernel itself is saved every kernel_save_interval steps to a compressed
    file next to the JSON so later analyses can use it.
    """
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown configuration keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **overrides}
    config = dict(name=name, **cfg)
    p, alpha = cfg["p"], cfg["alpha"]

    (X_train, y_train), (X_test, y_test) = make_modular_addition_dataset(
        p=p, train_fraction=cfg["train_fraction"], seed=cfg["data_seed"])
    X_train, y_train, X_test, y_test = [t.to(DEVICE) for t in (X_train, y_train, X_test, y_test)]

    probe_size = min(cfg["probe_size"], X_train.shape[0])
    x_probe, y_probe = X_train[:probe_size], y_train[:probe_size]

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
        K_norm = torch.linalg.norm(K_t, ord="fro").item()
        if K_norm > 0:
            yKy = torch.sum(y_probe * (torch.linalg.pinv(K_t, rtol=1e-6) @ y_probe)).item()
        else:
            yKy = float("nan")
        for k, v in zip(HISTORY_KEYS, [step, train_loss, test_loss, train_acc, test_acc,
                                       S_t, R_t, S_c, R_c, A_t, param_dist, weight_norm,
                                       yKy, K_norm]):
            history[k].append(v)
        if step % cfg["kernel_save_interval"] == 0:
            saved_steps.append(step)
            saved_kernels.append(K_t.cpu().numpy().astype(np.float32))
        model.train()

    t0 = time.time()
    evaluate(0)
    for step in range(1, cfg["steps"] + 1):
        optimizer.zero_grad()
        loss = loss_fn(predict(X_train), y_train)
        loss.backward()
        optimizer.step()
        if step % cfg["eval_interval"] == 0:
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


def run_or_load(name, rerun=False, verbose=True, **overrides):
    """Load a saved run if its configuration matches, otherwise train it."""
    path = RESULTS / f"{name}.json"
    if path.exists() and not rerun:
        run = json.loads(path.read_text())
        saved = {k: v for k, v in run["config"].items() if k != "name"}
        wanted = {**DEFAULTS, **overrides}
        if saved == wanted and "R_c" in run["history"]:
            if verbose:
                print(f"{name}: loaded from results/{name}.json")
            return run
        if verbose:
            print(f"{name}: saved run does not match, training again")
    return train_run(name, verbose=verbose, **overrides)


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
    fig.suptitle(title or run["config"]["name"])
    fig.tight_layout()
    return fig


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
        if key in log_y:
            ax.set_yscale("log")
        if log_x:
            log_step_axis(ax)
        else:
            linear_step_axis(ax)
    axes[legend_panel].legend()
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
