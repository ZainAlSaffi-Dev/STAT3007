"""
Pilot: does a one hidden layer network trained by full-batch gradient descent grok on MNIST?

Liu, Michaud and Tegmark (2023), Omnigrok, arXiv:2210.01117, Section 4 and
Appendix A, induce grokking on MNIST with 1000 training images, a larger
initial weight scale, mean squared error on one-hot targets, a three layer
ReLU network of width 200, and AdamW. The proposal's kernel measures assume
gradient descent, so this pilot keeps the report's training rule and model
and asks whether delayed generalisation still appears. It runs two arms.

- "kumar" is the report's setup: the centred predictor alpha [f - f_0] in
  mean-field parameterisation, with alpha in {1, 8}. A large alpha is the
  lazy regime of Chizat, Oyallon and Bach (2020).
- "omnigrok" keeps the initial function (centred=False) in NTK
  parameterisation, whose initial output is of order one, and multiplies the
  initial weights by init_scale in {1, 3}. This is the large initialisation of
  Omnigrok, Section 4, on the report's network.

Each arm crosses its scale with eta_kappa in {1e-4, 1e-3}. Every run uses the
same seed and data seed. The learning rate is set from the largest eigenvalue
lam_0 of the Gauss-Newton matrix of the loss at initialisation, found by
power iteration. Gradient descent on a quadratic diverges above eta = 2 / lam_0.

- In the "omnigrok" arm eta = 1 / lam_0. The NTK parameterisation keeps the
  curvature near its initial value, so this is half the stability bound.
- In the "kumar" arm eta lam_0 is set equal to its value at the report's
  baseline, eta = 100 on modular addition at alpha = 1, which is about
  1.5e-5. A mean-field network starts with a small output, and its curvature
  grows by orders of magnitude as the weights grow, so eta = 1 / lam_0 would
  diverge. Matching the baseline's ratio carries the report's setting over
  to MNIST.

Two follow-up batches look for the memorise-then-generalise phase that the
pilot did not show. Both use the "omnigrok" arm and 200000 steps.

- "gd" keeps gradient descent with eta = 1 / lam_0 and crosses init_scale in
  {3, 6, 9} with eta_kappa in {0, 1e-5, 3e-5}. Omnigrok, Appendix C, uses an
  initial scale of 9 on MNIST. The weaker decay is meant to let the network
  fit the training set before its norm shrinks.
- "adamw" uses AdamW with learning rate 1e-3, the optimiser and rate of
  Omnigrok, Appendix A, on the report's network, full batch. It crosses
  init_scale in {3, 9} with eta_kappa in {1e-4, 1e-3}, which at that rate
  is a weight decay of 0.1 and 1, the range over which Omnigrok, Appendix C,
  find t proportional to 1 / gamma on MNIST. This batch checks that the delay
  appears at all on this network. The tangent kernel does not give the
  function-space step under AdamW.

Two further batches follow Kumar et al. (2024), arXiv:2310.06110v3, Section 6
and Figure 6a. They start from the large-initialisation network of Omnigrok,
which groks, and multiply the centred predictor by alpha in
{1, 0.5, 0.05, 0.01, 0.001}, the values in that figure. Kumar et al. report
that a smaller alpha removes the delay without changing the initial weight
norm. Their text gives no MNIST settings beyond this, so the rest is taken
from Omnigrok and is an assumption.

- "kumar_a" is the faithful version: OmnigrokMLP, three layers of width 200
  with biases, initial parameters times 9 (Omnigrok, Appendix C), AdamW with
  learning rate 1e-3 and batch size 200 (Omnigrok, Appendix A), and
  eta_kappa = 1e-4, a weight decay of 0.1. The learning rate stays at 1e-3
  for every alpha, because the size of an AdamW step does not depend on the
  scale of the gradient. A sixth run drops the centring at alpha = 1, which
  is Omnigrok's own setup. The kernel is computed by autograd.
- "kumar_b" is the same sweep on the report's network: the bias-free two
  layer ReLU network in NTK parameterisation, initial weights times 3,
  full-batch gradient descent with eta = eta_0 / alpha^2 and eta_0 = 1 / lam_0
  at alpha = 1, and eta_kappa = 1e-5. Under gradient descent the tangent
  kernel sets the function-space step, so this version keeps the kernel
  reading of the report.

The centred runs of "kumar_a" and "kumar_b" lost test accuracy over training.
Weight decay pulls theta towards zero, so alpha [f(theta) - f(theta_0)]
drifts towards -alpha f(theta_0), the large random initial output with its
sign flipped. Kumar et al. tune alpha "without changing parameter weight
norm" and do not name the optimiser of their MNIST run. The batch
"kumar_wd0" therefore repeats "kumar_a" with eta_kappa = 0 under two
optimisers: AdamW with learning rate 1e-3, and plain SGD with no momentum
and eta = eta_0 / alpha^2, eta_0 = 1 / lam_0 at alpha = 1. Both use batch
size 200, and each adds one uncentred run.

The batch "sgd_grok" looks for grokking under SGD. The uncentred SGD run of
"kumar_wd0" fitted the training set and stayed at chance on the test set, as
Omnigrok's account predicts with no regulariser. This batch adds weight
decay, which that account says pulls the weight norm towards the region that
generalises, and momentum, which speeds SGD up. It uses OmnigrokMLP,
uncentred, batch size 200 and 200000 steps, and crosses init_scale in {3, 9}
with eta_kappa in {1e-5, 1e-4, 1e-3} and momentum in {0, 0.9}. The learning
rate is eta = 1 / lam_0 for each init_scale. With momentum beta the heavy-ball
method is stable on a quadratic below 2 (1 + beta) / lam_0, so 1 / lam_0 is
inside the bound for both values.

    uv run python mnist_pilot.py lr [batch]       # print the learning rates only
    uv run python mnist_pilot.py run [batch]      # train a batch in parallel
    uv run python mnist_pilot.py summary [batch]  # print the event steps and kernel terms

The batch is "pilot", "gd", "adamw", "kumar_a", "kumar_b", "kumar_wd0" or "sgd_grok", and "pilot" is the default.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, jvp, vjp

# ntk_lib and RepoducedCode live in repoduced-code at the repository root, and mnist_data in mnist/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repoduced-code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ntk_lib as L

OUT = Path(__file__).resolve().parent / "results" / "mnist_pilot"
STEPS, EVAL = 50_000, 250
COMMON = dict(dataset="mnist", n_train=1000, hidden_dim=100, probe="mixed", probe_size=256,
              kernel_method="closed_form", steps=STEPS, eval_interval=EVAL, kernel_save_interval=5000,
              seed=0, data_seed=42)
ARMS = {
    "kumar": [dict(parameterisation="mean_field", centred=True, alpha=a, init_scale=1.0) for a in (1.0, 8.0)],
    "omnigrok": [dict(parameterisation="ntk", centred=False, alpha=1.0, init_scale=s) for s in (1.0, 3.0)],
}
DECAYS = (1e-4, 1e-3)
# The accuracy levels reported here. The report's levels 0.80, 0.90 and 0.95
# are among them. Omnigrok, Figure 3e and Figure 10b, uses 0.60 and 0.70 on MNIST.
LEVELS = (0.6, 0.7, 0.8, 0.9, 0.95)


def top_gauss_newton_eigenvalue(cfg, iters=50):
    """Largest eigenvalue of (2 / (n C)) J^T J at initialisation, by power iteration.

    J is the Jacobian of the predictor on the training set with respect to the
    parameters, and n C is the number of entries MSELoss averages over.
    """
    cfg = {**L.DEFAULTS, **cfg}
    (X, y), _ = L.make_dataset(cfg)
    torch.manual_seed(cfg["seed"])
    model = L.build_model(cfg["parameterisation"], X.shape[1], cfg["hidden_dim"], y.shape[1],
                          cfg["activation"], cfg["init_scale"])
    params = {k: v.detach() for k, v in model.named_parameters()}
    f0 = functional_call(model, params, X).detach()
    alpha, centred = cfg["alpha"], cfg["centred"]

    def predict(p):
        out = functional_call(model, p, X)
        return alpha * (out - f0) if centred else alpha * out

    v = {k: torch.randn_like(t) for k, t in params.items()}
    lam = 0.0
    for _ in range(iters):
        norm = torch.sqrt(sum((t ** 2).sum() for t in v.values()))
        v = {k: t / norm for k, t in v.items()}
        _, Jv = jvp(predict, (params,), (v,))
        _, pullback = vjp(predict, params)
        (Gv,) = pullback(2.0 * Jv / Jv.numel())
        lam = sum((Gv[k] * v[k]).sum() for k in v).item()
        v = Gv
    return lam


def pilot_configs():
    """Return (name, config, lam_0) for every pilot run, with eta_0 set from the curvature."""
    baseline_ratio = L.DEFAULTS["eta_0"] * top_gauss_newton_eigenvalue({})
    out = []
    for arm, scales in ARMS.items():
        for scale in scales:
            base = {**COMMON, **scale}
            lam = top_gauss_newton_eigenvalue(base)
            ratio = baseline_ratio if arm == "kumar" else 1.0
            # train_run uses eta = eta_0 / alpha^2, so eta_0 = alpha^2 ratio / lam gives eta = ratio / lam.
            eta_0 = base["alpha"] ** 2 * ratio / lam
            for ek in DECAYS:
                tag = f"a{scale['alpha']:g}" if arm == "kumar" else f"s{scale['init_scale']:g}"
                out.append((f"mnist_{arm}_{tag}_ek{ek:g}", dict(base, eta_0=eta_0, eta_kappa=ek), lam))
    return out


LONG = dict(COMMON, parameterisation="ntk", centred=False, alpha=1.0, steps=200_000, eval_interval=500,
            kernel_save_interval=20_000)


def gd_configs():
    """Return (name, config, lam_0) for the gradient-descent batch, with eta = 1 / lam_0."""
    out = []
    for scale in (3.0, 6.0, 9.0):
        base = dict(LONG, init_scale=scale)
        lam = top_gauss_newton_eigenvalue(base)
        for ek in (0.0, 1e-5, 3e-5):
            out.append((f"mnist_gd_s{scale:g}_ek{ek:g}", dict(base, eta_0=1.0 / lam, eta_kappa=ek), lam))
    return out


def adamw_configs():
    """Return (name, config, lam_0) for the AdamW batch, with learning rate 1e-3."""
    out = []
    for scale in (3.0, 9.0):
        base = dict(LONG, init_scale=scale, optimizer="adamw", eta_0=1e-3)
        lam = top_gauss_newton_eigenvalue(base)
        for ek in (1e-4, 1e-3):
            out.append((f"mnist_adamw_s{scale:g}_ek{ek:g}", dict(base, eta_kappa=ek), lam))
    return out


KUMAR_ALPHAS = (1.0, 0.5, 0.05, 0.01, 0.001)


def log_grid(steps, n_log=100, linear=2000):
    """Checkpoints spaced evenly in log step up to steps, together with every linear-th step."""
    return sorted({int(round(t)) for t in np.geomspace(1, steps, n_log)} | set(range(0, steps + 1, linear)))


def kumar_a_configs():
    """Return (name, config, lam_0) for the faithful Kumar batch. lam_0 is not used and is NaN."""
    steps = 200_000
    base = dict(COMMON, parameterisation="omnigrok", hidden_dim=200, init_scale=9.0, optimizer="adamw",
                eta_0=1e-3, eta_kappa=1e-4, batch_size=200, kernel_method="autograd", steps=steps,
                checkpoint_steps=log_grid(steps), kernel_save_interval=20_000)
    out = []
    for a in KUMAR_ALPHAS:
        # train_run divides eta_0 by alpha^2, so eta_0 = alpha^2 1e-3 keeps the AdamW rate at 1e-3.
        out.append((f"mnist_kumarA_a{a:g}", dict(base, alpha=a, eta_0=1e-3 * a ** 2), float("nan")))
    out.append(("mnist_kumarA_uncentred", dict(base, alpha=1.0, centred=False), float("nan")))
    return out


def kumar_b_configs():
    """Return (name, config, lam_0) for the Kumar batch on the report's network."""
    base = dict(LONG, init_scale=3.0, centred=True, eta_kappa=1e-5)
    lam = top_gauss_newton_eigenvalue(dict(base, alpha=1.0))
    return [(f"mnist_kumarB_a{a:g}", dict(base, alpha=a, eta_0=1.0 / lam), lam) for a in KUMAR_ALPHAS]


def kumar_wd0_configs():
    """Return (name, config, lam_0) for the Kumar batch without weight decay, under AdamW and SGD."""
    out = []
    for name, cfg, lam in kumar_a_configs():
        out.append((name.replace("kumarA", "kumarW_adamw"), dict(cfg, eta_kappa=0.0), lam))
    sgd = dict(kumar_a_configs()[0][1], optimizer="sgd", eta_kappa=0.0)
    lam = top_gauss_newton_eigenvalue(dict(sgd, alpha=1.0))
    for a in KUMAR_ALPHAS:
        out.append((f"mnist_kumarW_sgd_a{a:g}", dict(sgd, alpha=a, eta_0=1.0 / lam), lam))
    out.append(("mnist_kumarW_sgd_uncentred", dict(sgd, alpha=1.0, centred=False, eta_0=1.0 / lam), lam))
    return out


def sgd_grok_configs():
    """Return (name, config, lam_0) for the SGD batch with weight decay and momentum."""
    out = []
    base = dict(kumar_a_configs()[0][1], optimizer="sgd", centred=False, alpha=1.0)
    for scale in (3.0, 9.0):
        lam = top_gauss_newton_eigenvalue(dict(base, init_scale=scale))
        for ek in (1e-5, 1e-4, 1e-3):
            for mom in (0.0, 0.9):
                out.append((f"mnist_sgdgrok_s{scale:g}_ek{ek:g}_m{mom:g}",
                            dict(base, init_scale=scale, eta_0=1.0 / lam, eta_kappa=ek, momentum=mom), lam))
    return out


BATCHES = dict(pilot=pilot_configs, gd=gd_configs, adamw=adamw_configs, kumar_a=kumar_a_configs,
               kumar_b=kumar_b_configs, kumar_wd0=kumar_wd0_configs, sgd_grok=sgd_grok_configs)


def _train(item):
    torch.set_num_threads(1)
    name, cfg, _ = item
    return L.run_or_load(name, verbose=False, results_dir=OUT, **cfg)["seconds"]


def summarise(run):
    h = {k: np.asarray(v) for k, v in run["history"].items()}
    step = h["step"]

    def first(key, level):
        hit = np.nonzero(h[key] >= level)[0]
        return int(step[hit[0]]) if hit.size else None

    mem = first("train_acc", 0.99)
    row = dict(mem=mem, test_at_mem=None if mem is None else float(h["test_acc"][step == mem][0]),
               test_end=float(h["test_acc"][-1]), test_max=float(h["test_acc"].max()),
               A0=float(h["A_t"][0]), A_end=float(h["A_t"][-1]), S_end=float(h["S_t"][-1]),
               R_end=float(h["R_t"][-1]), w_ratio_end=float(h["weight_norm"][-1] / run["init_weight_norm"]))
    row.update({f"t{lv:g}": first("test_acc", lv) for lv in LEVELS})
    return row


def main(cmd, batch="pilot"):
    OUT.mkdir(parents=True, exist_ok=True)
    items = BATCHES[batch]()
    if cmd in ("lr", "run"):
        for name, cfg, lam in items:
            eta = cfg["eta_0"] / cfg["alpha"] ** 2
            print(f"{name}: top Gauss-Newton eigenvalue {lam:.4g}, eta {eta:.4g}, eta lam {eta * lam:.3g}")
        if cmd == "run":
            with ProcessPoolExecutor(max_workers=min(len(items), 12)) as pool:
                for (name, _, _), sec in zip(items, pool.map(_train, items)):
                    print(f"{name}: {sec:.0f} s")
    if cmd in ("run", "summary"):
        rows = {}
        for name, _, _ in items:
            rows[name] = summarise(json.loads((OUT / f"{name}.json").read_text()))
        cols = ["mem", "test_at_mem", *[f"t{lv:g}" for lv in LEVELS], "test_max", "test_end",
                "A0", "A_end", "S_end", "R_end", "w_ratio_end"]
        print("\n" + "run".ljust(30) + "".join(c.rjust(12) for c in cols))
        for name, r in rows.items():
            cells = []
            for c in cols:
                v = r[c]
                cells.append(("-" if v is None else f"{v:.3g}" if isinstance(v, float) else str(v)).rjust(12))
            print(name.ljust(30) + "".join(cells))
        (OUT / f"summary_{batch}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "summary", sys.argv[2] if len(sys.argv) > 2 else "pilot")
