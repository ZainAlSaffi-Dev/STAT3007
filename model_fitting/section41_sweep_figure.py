"""The sweep figure of section 4.1, drawn from the grid database.

Seed 0 of each first-stage cell of the grid (the sweep over alpha at the settings of the
pilot cell) is read from results/grid.duckdb. The left panel shows the training and test MSE
and the right panel the accuracies against step. A hollow point marks memorisation, the
first step at training accuracy 0.99, and a filled point marks grokking, the first step at
test accuracy 0.95, both read from the events table of the database.

Run from the repository root:  uv run python model_fitting/section41_sweep_figure.py
It writes docs/figures/alpha_sweep.pdf.
"""
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "docs" / "figures"
SEED = 0

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.2, "figure.dpi": 150})
ALPHA_COLOURS = {0.5: "#1f77b4", 1.0: "#d95f02", 1.5: "#2ca02c", 2.0: "#7b3294"}


def main():
    con = duckdb.connect(str(ROOT / "results" / "grid.duckdb"), read_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
    for a in sorted(ALPHA_COLOURS):
        h = con.sql(f"""
            SELECT step, train_loss, test_loss, train_acc, test_acc
            FROM metrics NATURAL JOIN cells
            WHERE stage = '1' AND alpha = {a} AND seed = {SEED}
            ORDER BY step
        """).df()
        ev = con.sql(f"""
            SELECT event, level, time, observed
            FROM events NATURAL JOIN cells
            WHERE stage = '1' AND alpha = {a} AND seed = {SEED}
        """).df()
        st = h.step.values.astype(float)
        st[0] = max(st[0], 1.0)
        c = ALPHA_COLOURS[a]
        axes[0].plot(st, h.train_loss, color=c, label=f"$\\alpha={a:g}$")
        axes[0].plot(st, h.test_loss, color=c, ls="--")
        axes[1].plot(st, h.train_acc, color=c)
        axes[1].plot(st, h.test_acc, color=c, ls="--")
        mem = ev[(ev.event == "memorised") & (ev.level == 0.99)]
        if len(mem) and bool(mem.observed.iloc[0]):
            t = float(mem.time.iloc[0])
            axes[1].plot([t], [np.interp(t, st, h.train_acc)], "o", ms=4, mfc="white", mec=c, mew=1.2, zorder=5)
        grok = ev[(ev.event == "grokked") & (ev.level == 0.95)]
        if len(grok) and bool(grok.observed.iloc[0]):
            t = float(grok.time.iloc[0])
            axes[1].plot([t], [np.interp(t, st, h.test_acc)], "o", ms=4, color=c, zorder=5)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("step")
    axes[0].set_yscale("log")
    axes[0].set_title("MSE")
    axes[0].set_ylim(1e-4, 0.1)
    axes[1].set_title("accuracy")
    axes[1].set_ylim(-0.03, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "alpha_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "alpha_sweep.pdf")


if __name__ == "__main__":
    main()
