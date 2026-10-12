"""
Option B of docs/PLAN_mnist_closed_form.md: grokking on MNIST with a bias-free three layer ReLU network.

The network is OmnigrokMLP without its biases: ntk_lib.DeepBiasFreeMLP with
three linear layers of width 200 in the standard parameterisation. The
predictor is uncentred. Every run uses MSE on one-hot targets, 1000 training
images drawn by data_seed 42, the full test set, the mixed probe of 128
training and 128 test images, plain SGD without momentum, batch size 200 and
the closed-form kernel, unless its batch says otherwise. The learning rate is
eta = ratio / lam_0, where lam_0 is the top eigenvalue of the Gauss-Newton
matrix of the loss at the run's own initial weights. The checkpoints are log
spaced in step together with every 2000th step, as in mnist_pilot.log_grid.

The batches follow section 4 of the plan.

- "b1" crosses init_scale {3, 9} with eta_kappa {1e-5, 3e-5} and seeds
  {0, 1, 2}, 200000 steps, eta = 1 / lam_0.
- "b2" tests fixes for the late-training stall at init_scale 9, eta_kappa
  1e-5, seeds {0, 1} and 500000 steps: a fixed eta = 1.9 / lam_0, a rate
  re-estimated from the curvature every 10000 steps, and a rate that grows
  as the fourth power of the shrinking weight norm, with the factor capped
  at 1000. The rule eta = 1 / lam_0 is run as a reference.
- "b3" maps init_scale {3, 5, 7, 9} against eta_kappa {3e-6, 1e-5, 2e-5}
  with seeds {0, 1} under the rule in B3_RULE.
- "b4" tests the cell in BEST_CELL for robustness.
- "b5" runs the cell in BEST_CELL on OmnigrokMLP with biases and the
  autograd kernel.

    uv run python mnist_optionB.py run b1        # train a batch, at most 12 processes
    uv run python mnist_optionB.py summary b1    # print the criterion table of a batch
    uv run python mnist_optionB.py b6            # kernel analysis and figures of grokking runs

The criterion is that of section 3 of the plan. t_mem is the first checkpoint
with training accuracy at least 0.95. A run shows strong grokking if its test
accuracy at t_mem is at most 0.30, its best test accuracy is at least 0.80,
and t_0.5 / t_mem is at least 3. It shows weak grokking if it is not strong,
its gain after memorisation is at least 0.25, and the delay ratio at the
level halfway through that gain is at least 3.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

# ntk_lib and RepoducedCode live in repoduced-code at the repository root, and mnist_data in mnist/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repoduced-code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ntk_lib as L

OUT = Path(__file__).resolve().parent / "results" / "mnist_optionB"
LEVELS = [round(0.30 + 0.05 * i, 2) for i in range(13)]


def log_grid(steps, n_log=100, linear=2000):
    """Checkpoints spaced evenly in log step up to steps, together with every linear-th step."""
    return sorted({int(round(t)) for t in np.geomspace(1, steps, n_log)} | set(range(0, steps + 1, linear)))


def base_config(steps):
    return dict(dataset="mnist", n_train=1000, data_seed=42, parameterisation="deep_standard", depth=3,
                hidden_dim=200, centred=False, alpha=1.0, probe="mixed", probe_size=256,
                kernel_method="closed_form", optimizer="sgd", momentum=0.0, batch_size=200,
                steps=steps, checkpoint_steps=log_grid(steps), kernel_save_interval=50_000)


def lam_0(cfg):
    """Top Gauss-Newton eigenvalue of the loss at the initial weights that train_run will draw."""
    cfg = {**L.DEFAULTS, **cfg}
    (X, y), _ = L.make_dataset(cfg)
    torch.manual_seed(cfg["seed"])
    model = L.build_model(cfg["parameterisation"], X.shape[1], cfg["hidden_dim"], y.shape[1],
                          cfg["activation"], cfg["init_scale"],
                          **({} if cfg["depth"] is None else dict(depth=cfg["depth"])))
    return L.gauss_newton_top_eigenvalue(model, X, alpha=cfg["alpha"])


def with_rate(name, cfg, ratio=1.0):
    """Return (name, config, lam_0) with eta_0 = ratio / lam_0 for the run's own initial weights."""
    lam = lam_0(cfg)
    return name, dict(cfg, eta_0=ratio / lam), lam


def ek_tag(ek):
    return f"{ek:g}"


# ---------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------
def b1_configs():
    out = []
    for scale in (3.0, 9.0):
        for ek in (1e-5, 3e-5):
            for seed in (0, 1, 2):
                cfg = dict(base_config(200_000), init_scale=scale, eta_kappa=ek, seed=seed)
                out.append(with_rate(f"optB_b1_s{scale:g}_ek{ek_tag(ek)}_seed{seed}", cfg))
    return out


# The learning-rate rules of stage B2. Each maps to (ratio of eta_0 to 1 / lam_0, extra configuration).
RULES = {
    "lam1": (1.0, {}),
    "lam1.9": (1.9, {}),
    "curv": (1.0, dict(lr_schedule="curvature", lr_update_interval=10_000)),
    "norm": (1.0, dict(lr_schedule="norm", lr_norm_power=4.0, lr_cap=1000.0)),
}


def b2_configs():
    out = []
    for rule in ("lam1.9", "curv", "norm", "lam1"):
        ratio, extra = RULES[rule]
        for seed in (0, 1):
            cfg = dict(base_config(500_000), init_scale=9.0, eta_kappa=1e-5, seed=seed, **extra)
            out.append(with_rate(f"optB_b2_{rule}_s9_ek1e-05_seed{seed}", cfg, ratio))
    return out


# Set after B1 and B2 are read. The curvature rule was the only B2 rule that gave strong grokking at
# init_scale 9 (2 of 2 seeds); see REPORT.md.
B3_RULE = "curv"


def b3_steps(scale):
    return 500_000 if scale >= 7 else 300_000


def b3_configs():
    """The longest runs come first, so that they start first. A cell already run in B2 keeps its B2 name,
    so run_or_load loads it instead of training it again."""
    ratio, extra = RULES[B3_RULE]
    out = []
    for scale in (9.0, 7.0, 5.0, 3.0):
        for ek in (3e-6, 1e-5, 2e-5):
            for seed in (0, 1):
                cfg = dict(base_config(b3_steps(scale)), init_scale=scale, eta_kappa=ek, seed=seed, **extra)
                name = f"optB_b3_{B3_RULE}_s{scale:g}_ek{ek_tag(ek)}_seed{seed}"
                if scale == 9.0 and ek == 1e-5:
                    name = f"optB_b2_{B3_RULE}_s9_ek1e-05_seed{seed}"
                out.append(with_rate(name, cfg, ratio))
    return out


# Cells of B3 that get a third seed, set after B3 is read.
B3X_CELLS = [(9.0, 3e-6), (9.0, 1e-5), (7.0, 3e-6), (7.0, 1e-5), (5.0, 3e-6)]


def b3x_configs():
    ratio, extra = RULES[B3_RULE]
    out = []
    for scale, ek in B3X_CELLS:
        cfg = dict(base_config(b3_steps(scale)), init_scale=scale, eta_kappa=ek, seed=2, **extra)
        out.append(with_rate(f"optB_b3_{B3_RULE}_s{scale:g}_ek{ek_tag(ek)}_seed2", cfg, ratio))
    return out


# The cell with the strongest reliable grokking, set after B3 is read. init_scale 7 with eta_kappa 3e-6
# was strong in both seeds with the largest margins on all three strong conditions; see REPORT.md.
BEST_CELL = dict(init_scale=7.0, eta_kappa=3e-6, rule="curv")


def best_base(steps=None):
    scale, ek, rule = BEST_CELL["init_scale"], BEST_CELL["eta_kappa"], BEST_CELL["rule"]
    ratio, extra = RULES[rule]
    steps = steps or b3_steps(scale)
    return dict(base_config(steps), init_scale=scale, eta_kappa=ek, **extra), ratio, \
        f"{rule}_s{scale:g}_ek{ek_tag(ek)}"


def b4_configs():
    cfg, ratio, tag = best_base()
    out = []
    for seed in range(3, 8):
        out.append(with_rate(f"optB_b4_{tag}_seed{seed}", dict(cfg, seed=seed), ratio))
    for seed in (0, 1, 2):
        out.append(with_rate(f"optB_b4_{tag}_fullbatch_seed{seed}", dict(cfg, seed=seed, batch_size=None), ratio))
        for n in (500, 2000):
            out.append(with_rate(f"optB_b4_{tag}_n{n}_seed{seed}", dict(cfg, seed=seed, n_train=n), ratio))
        out.append(with_rate(f"optB_b4_{tag}_w100_seed{seed}", dict(cfg, seed=seed, hidden_dim=100), ratio))
    return out


def b5_configs():
    cfg, ratio, tag = best_base()
    out = []
    for seed in range(6):
        c = dict(cfg, seed=seed, parameterisation="omnigrok", depth=None, kernel_method="autograd")
        out.append(with_rate(f"optB_b5_bias_{tag}_seed{seed}", c, ratio))
    return out


def late_configs():
    """B4, the third seeds of B3 and B5 in one pool, with the slow full-batch runs first."""
    items = b4_configs() + b3x_configs() + b5_configs()
    return sorted(items, key=lambda it: it[1].get("batch_size", 200) is not None)


BATCHES = dict(b1=b1_configs, b2=b2_configs, b3=b3_configs, b3x=b3x_configs, b4=b4_configs, b5=b5_configs,
               late=late_configs)


def _train(item):
    torch.set_num_threads(1)
    name, cfg, _ = item
    return L.run_or_load(name, verbose=False, results_dir=OUT, **cfg)["seconds"]


# ---------------------------------------------------------------------
# The criterion of section 3
# ---------------------------------------------------------------------
def first_at_least(steps, values, level):
    hit = np.nonzero(np.asarray(values) >= level - 1e-12)[0]
    return int(steps[hit[0]]) if hit.size else None


def classify(run):
    """Return the section 3 summary of one run as a dictionary."""
    h = {k: np.asarray(v, float) for k, v in run["history"].items()}
    steps = h["step"].astype(int)
    t_mem = first_at_least(steps, h["train_acc"], 0.95)
    best = float(h["test_acc"].max())
    w_end = float(h["weight_norm"][-1] / run["init_weight_norm"])
    row = dict(t_mem=t_mem, best=best, test_end=float(h["test_acc"][-1]), train_end=float(h["train_acc"][-1]),
               train_max=float(h["train_acc"].max()), w_end=w_end, last=int(steps[-1]),
               diverged=run.get("diverged", False))
    levels = {}
    for lv in LEVELS:
        t = first_at_least(steps, h["test_acc"], lv)
        if t is None:
            levels[lv] = "never"
        elif t_mem is None:
            levels[lv] = f"@{t}"
        elif t <= t_mem:
            levels[lv] = "pre"
        else:
            levels[lv] = t / t_mem
    row["levels"] = levels
    if t_mem is None:
        row.update(test_at_mem=None, gain=None, half_level=None, half_ratio=None, verdict="none")
        # The weight norm relative to the default initialisation is w_end times init_scale.
        # Below 0.3 of the default norm, or at chance training accuracy, the weights have collapsed.
        w_default = w_end * run["config"]["init_scale"]
        row["failure"] = "collapsed" if (w_default < 0.3 or row["train_end"] < 0.2) else "never fitted (stalled)"
        return row
    test_at_mem = float(h["test_acc"][steps == t_mem][0])
    gain = best - test_at_mem
    half = test_at_mem + gain / 2
    t_half = first_at_least(steps, h["test_acc"], half)
    half_ratio = None if t_half is None or t_half <= t_mem else t_half / t_mem
    r50 = levels[0.5]
    strong = test_at_mem <= 0.30 and best >= 0.80 and isinstance(r50, float) and r50 >= 3
    weak = (not strong) and gain >= 0.25 and half_ratio is not None and half_ratio >= 3
    verdict = "strong" if strong else "weak" if weak else "none"
    failure = None
    if verdict == "none":
        if gain < 0.25:
            failure = "fitted, little gain after memorisation"
        else:
            failure = "gain but no delay"
    if row["train_end"] < 0.95:
        failure = (failure + ", " if failure else "") + f"lost fit (train {row['train_end']:.2f} at end)"
    row.update(test_at_mem=test_at_mem, gain=gain, half_level=half, half_ratio=half_ratio, verdict=verdict,
               failure=failure)
    return row


def fmt_level(v):
    if isinstance(v, float):
        return f"{v:.1f}"
    return v


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def summary(batch):
    items = BATCHES[batch]()
    rows = {}
    for name, cfg, lam in items:
        path = OUT / f"{name}.json"
        if not path.exists():
            continue
        r = classify(json.loads(path.read_text()))
        r["lam_0"], r["eta"] = lam, cfg["eta_0"]
        rows[name] = r
    (OUT / f"summary_{batch}.json").write_text(json.dumps(rows, indent=1, default=str))
    print_table(rows)
    return rows


def print_table(rows):
    lv_cols = [0.5, 0.8, 0.9]
    head = ["run", "t_mem", "test@mem", "best", "end test", "end train", "w ratio",
            *[f"r{lv:g}" for lv in lv_cols], "half r", "verdict", "failure"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name, r in rows.items():
        f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"
        cells = [name, str(r["t_mem"]), f(r["test_at_mem"]), f(r["best"]), f(r["test_end"]), f(r["train_end"]),
                 f(r["w_end"]), *[str(fmt_level(r["levels"][lv])) for lv in lv_cols], f(r["half_ratio"], 1),
                 r["verdict"], r.get("failure") or ""]
        print("| " + " | ".join(cells) + " |")


def level_table(rows):
    head = ["run", *[f"{lv:.2f}" for lv in LEVELS]]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name, r in rows.items():
        print("| " + " | ".join([name, *[str(fmt_level(r["levels"][lv])) for lv in LEVELS]]) + " |")


def main(cmd, batch="b1"):
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "run":
        items = BATCHES[batch]()
        for name, cfg, lam in items:
            print(f"{name}: lam_0 {lam:.4g}, eta {cfg['eta_0']:.4g}", flush=True)
        with ProcessPoolExecutor(max_workers=min(len(items), 12)) as pool:
            for (name, _, _), sec in zip(items, pool.map(_train, items)):
                print(f"{name}: {sec:.0f} s", flush=True)
        summary(batch)
    elif cmd == "summary":
        summary(batch)
    elif cmd == "levels":
        rows = json.loads((OUT / f"summary_{batch}.json").read_text())
        rows = {k: dict(v, levels={float(a): (b if isinstance(b, str) else float(b)) for a, b in v["levels"].items()})
                for k, v in rows.items()}
        level_table(rows)
    elif cmd == "alltables":
        # One table of every run in every batch, then the full level table, for REPORT.md.
        # The saved runs are read directly, so lam_0 is not recomputed.
        rows = {}
        for path in sorted(OUT.glob("optB_*.json")):
            rows[path.stem] = classify(json.loads(path.read_text()))
        print_table(rows)
        print()
        level_table(rows)
    elif cmd == "b6":
        import mnist_optionB_b6
        mnist_optionB_b6.main(sys.argv[2:])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "summary", sys.argv[2] if len(sys.argv) > 2 else "b1")
