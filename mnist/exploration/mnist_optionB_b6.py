"""
Stage B6 of docs/PLAN_mnist_closed_form.md: the kernel analysis of the Option B runs that grok.

For each run it reads the history saved by mnist_optionB and reports the
following.

- t_mem, the level table and the gain after memorisation, from
  mnist_optionB.classify.
- S_t, R_t, D_t, A_t and A_t_test at t_mem, t_0.5, t_0.8 and the last step.
  S_t is the log of the kernel norm ratio, as ntk_lib records it. D_t is
  ||K_t - K_0||_F / ||K_0||_F, recovered from S_t and R_t.
- The per-layer alignment A_Wl, log norm ratio S_Wl and trace share
  share_Wl of the closed form at the same steps, and the weight norm ratio.
- The first step after t_mem at which A_t exceeds its value at t_mem by
  0.05, 0.10 and 0.20.
- The slope per step of log ||K_t||_F and of log ||theta_t|| over two late
  windows, against the prediction for a k-homogeneous network of
  Lewkowycz and Gur-Ari (2021), arXiv:2006.08643v2. Their Theorem 1,
  Equation 2, gives d Theta / dt = -2 (k - 1) lambda Theta in the infinite
  width limit under gradient flow on L + (lambda / 2) ||theta||^2. Their
  Equation S5, which holds at finite width, adds terms that are proportional
  to the loss derivative, so the pure decay is expected only when the
  residuals are small. One SGD step with per-step decay eta_kappa is a
  gradient flow time eta with lambda = kappa, so the predicted slope of
  log ||K|| is -2 (k - 1) eta_kappa per step, and of log ||theta|| it is
  -eta_kappa per step when the residual term is negligible. Here k = 3.

    uv run python mnist_optionB_b6.py RUN [RUN ...]
    uv run python mnist_optionB_b6.py --auto      # every strong or weak run in the batch summaries

Figures go to results/mnist_optionB/figures/, one per configuration, with
the seeds of that configuration overlaid.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ntk_lib and RepoducedCode live in repoduced-code at the repository root, and mnist_data in mnist/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repoduced-code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ntk_lib as L
from mnist_optionB import OUT, classify, load

FIG = OUT / "figures"
DELTAS = (0.05, 0.10, 0.20)
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 3))]


def value_at(h, key, step):
    steps = np.asarray(h["step"])
    i = np.where(steps <= step)[0][-1]
    return h[key][i]


def d_from(S, R):
    return float(np.sqrt(max(np.exp(2 * S) + 1 - 2 * np.exp(S) * (1 - R), 0.0)))


def slope(h, key, lo, hi, log=True):
    steps = np.asarray(h["step"], float)
    v = np.asarray(h[key], float)
    m = (steps >= lo) & (steps <= hi) & np.isfinite(v) & (v > 0)
    if m.sum() < 3:
        return None
    y = np.log(v[m]) if log else v[m]
    return float(np.polyfit(steps[m], y, 1)[0])


def analyse(name):
    run = load(name)
    h = run["history"]
    cfg = run["config"]
    c = classify(run)
    t_mem = c["t_mem"]
    steps = np.asarray(h["step"])

    def t_level(lv):
        idx = np.nonzero(np.asarray(h["test_acc"]) >= lv)[0]
        return int(steps[idx[0]]) if idx.size else None

    marks = dict(t_mem=t_mem, t_0_5=t_level(0.5), t_0_8=t_level(0.8), end=int(steps[-1]))
    depth = cfg.get("depth") or 0
    at = {}
    for label, s in marks.items():
        if s is None:
            at[label] = None
            continue
        row = dict(step=s, train_acc=value_at(h, "train_acc", s), test_acc=value_at(h, "test_acc", s),
                   train_loss=value_at(h, "train_loss", s),
                   S_t=value_at(h, "S_t", s), R_t=value_at(h, "R_t", s), A_t=value_at(h, "A_t", s),
                   A_t_test=value_at(h, "A_t_test", s),
                   w_ratio=value_at(h, "weight_norm", s) / run["init_weight_norm"])
        row["D_t"] = d_from(row["S_t"], row["R_t"])
        for l in range(1, depth + 1):
            if f"A_W{l}" in h:
                for k in ("A_W", "S_W", "share_W"):
                    row[f"{k}{l}"] = value_at(h, f"{k}{l}", s)
        at[label] = row
    rise = {}
    if t_mem is not None:
        A_mem = value_at(h, "A_t", t_mem)
        for d in DELTAS:
            idx = np.nonzero((steps >= t_mem) & (np.asarray(h["A_t"]) >= A_mem + d))[0]
            rise[d] = int(steps[idx[0]]) if idx.size else None
    k = depth if depth else None
    ek = cfg["eta_kappa"]
    last = int(steps[-1])
    start_late = marks["t_0_8"] or t_mem or 0
    windows = {"from t_0.8 (or t_mem) to end": (start_late, last), "last quarter": (0.75 * last, last)}
    slopes = {}
    for wname, (lo, hi) in windows.items():
        slopes[wname] = dict(K=slope(h, "K_norm", lo, hi), w=slope(h, "weight_norm", lo, hi),
                             train_loss_lo=value_at(h, "train_loss", lo) if lo > 0 else None,
                             train_loss_hi=value_at(h, "train_loss", hi))
    pred = dict(K=-2 * (k - 1) * ek if k else None, w=-ek)
    return dict(name=name, classify=c, marks=marks, at=at, A_rise=rise, slopes=slopes, prediction=pred, k=k,
                eta_kappa=ek, R_end=h["R_t"][-1])


def config_key(name):
    return re.sub(r"_seed\d+$", "", name)


def figure(names, path):
    L.set_plot_style()
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for i, name in enumerate(names):
        run = load(name)
        h = run["history"]
        s = np.asarray(h["step"])
        ls = LINESTYLES[i % len(LINESTYLES)]
        seed = run["config"]["seed"]
        axes[0].plot(s, h["train_acc"], color=L.BLUE, ls=ls, lw=1.5, label=f"train, seed {seed}")
        axes[0].plot(s, h["test_acc"], color=L.ORANGE, ls=ls, lw=1.5, label=f"test, seed {seed}")
        axes[1].plot(s, h["A_t"], color=L.AQUA, ls=ls, lw=1.5, label=f"A_t, mixed probe, seed {seed}")
        axes[1].plot(s, h["A_t_test"], color=L.ORANGE, ls=ls, lw=1.5, label=f"A_t, test part, seed {seed}")
        axes[2].plot(s, h["S_t"], color=L.RAMP[3], ls=ls, lw=1.5, label=f"S_t, seed {seed}")
        axes[2].plot(s, h["R_t"], color=L.GRAY, ls=ls, lw=1.5, label=f"R_t, seed {seed}")
        axes[3].plot(s, np.asarray(h["weight_norm"]) / run["init_weight_norm"], color=L.GRAY, ls=ls, lw=1.5,
                     label=f"seed {seed}")
        t_mem = classify(run)["t_mem"]
        if t_mem is not None:
            for ax in axes:
                ax.axvline(t_mem, color=L.BLUE, lw=0.8, ls=ls, alpha=0.6)
    titles = ["Accuracy", "Centred alignment with the labels", "Kernel scale and shape change",
              "Weight norm over its initial value"]
    ylabels = ["accuracy", "centred alignment", "S_t (log norm ratio) and R_t", "weight norm ratio"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t, loc="left", fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel(yl)
        L.log_step_axis(ax)
        L.tidy_axes(ax)
        ax.legend(fontsize=6)
    axes[0].set_ylim(0, 1.02)
    fig.suptitle(f"{config_key(names[0])}. Vertical lines mark memorisation at 95 percent training accuracy.",
                 fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def fmt(x, d=3):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{d}g}" if abs(x) < 1e-2 and x != 0 else f"{x:.{d}f}"


def write_tables(results):
    lines = []
    lines.append("| run | point | step | train acc | test acc | S_t | R_t | D_t | A_t | A_t_test | w ratio | "
                 "A_W1 | A_W2 | A_W3 | S_W1 | S_W2 | S_W3 | share_W1 | share_W2 | share_W3 |")
    lines.append("|" + "---|" * 20)
    for r in results:
        for label, row in r["at"].items():
            if row is None:
                lines.append(f"| {r['name']} | {label} | not reached |" + " |" * 17)
                continue
            cells = [r["name"], label, str(row["step"]), fmt(row["train_acc"]), fmt(row["test_acc"]),
                     fmt(row["S_t"]), fmt(row["R_t"]), fmt(row["D_t"]), fmt(row["A_t"]), fmt(row["A_t_test"]),
                     fmt(row["w_ratio"])]
            for k in ("A_W", "S_W", "share_W"):
                for l in (1, 2, 3):
                    cells.append(fmt(row.get(f"{k}{l}")))
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| run | t_mem | t_0.5 | t_0.8 | A_t at t_mem | A_t rise 0.05 | A_t rise 0.10 | A_t rise 0.20 |")
    lines.append("|" + "---|" * 8)
    for r in results:
        m = r["marks"]
        A_mem = r["at"]["t_mem"]["A_t"] if r["at"]["t_mem"] else None
        lines.append("| " + " | ".join([r["name"], fmt(m["t_mem"]), fmt(m["t_0_5"]), fmt(m["t_0_8"]), fmt(A_mem),
                                         *[fmt(r["A_rise"].get(d)) for d in DELTAS]]) + " |")
    lines.append("")
    lines.append("| run | window | slope of log K norm per step | prediction -2(k-1) eta_kappa | ratio | "
                 "slope of log weight norm | prediction -eta_kappa | train loss at window start | at end | R_t at end |")
    lines.append("|" + "---|" * 10)
    for r in results:
        for w, s in r["slopes"].items():
            pk, pw = r["prediction"]["K"], r["prediction"]["w"]
            ratio = None if s["K"] is None or not pk else s["K"] / pk
            lines.append("| " + " | ".join([r["name"], w, fmt(s["K"]), fmt(pk), fmt(ratio, 2), fmt(s["w"]), fmt(pw),
                                             fmt(s["train_loss_lo"]), fmt(s["train_loss_hi"]),
                                             fmt(r["R_end"])]) + " |")
    return "\n".join(lines)


def main(args):
    if args and args[0] == "--auto":
        names = []
        for p in sorted(OUT.glob("summary_b*.json")):
            rows = json.loads(p.read_text())
            names += [n for n, r in rows.items() if r["verdict"] in ("strong", "weak")]
    else:
        names = args
    names = list(dict.fromkeys(names))
    results = [analyse(n) for n in names]
    (OUT / "b6_analysis.json").write_text(json.dumps(results, indent=1, default=str))
    text = write_tables(results)
    (OUT / "b6_tables.md").write_text(text + "\n")
    print(text)
    groups = defaultdict(list)
    for n in names:
        groups[config_key(n)].append(n)
    for key, members in groups.items():
        path = FIG / f"{key}.png"
        figure(members, path)
        print(f"figure: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
