"""Baseline: the network memorises early, groks late, and its kernel mostly grows.

Slide 18 of the deck, in place of report figure E1. Every curve is a
measurement. The runs are the five seeds of the baseline cell, alpha 1, width
100 and eta lambda 0 (ntk_N100_a1_wd0_s0 to s4), read from
animations/data/n100_checkpoints.npz. Each panel shows the median over the
seeds as a line and the range over the seeds as a band, for S_t on top and R_t
below, on a log step axis. The markers sit at the median memorisation step and
the median grokking step of the five runs, from report_runs.csv. The bar splits
D_t^2 at grokking into the scale part (S_t - 1)^2 and the rotation part
2 S_t R_t (report_v2 Equation eq:identity). Its share is the median over the
seeds of the scale part's share at each run's own grokking step, as in
report_v2/code/E01_baseline.py. The scene asserts that its medians equal the
numbers the slide prints.

Render from animations/:

    uv run manim -qh slides/baseline_kernel.py BaselineKernel
    uv run manim -qh -s slides/baseline_kernel.py BaselineKernel
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, FadeIn, GrowFromEdge, MathTex, Polygon, Rectangle, Text,
                   ValueTracker, VGroup, always_redraw, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, layout, palette

SIZE = palette.LABEL_COLUMN
MATH = SIZE + 8
X_LENGTH, Y_LENGTH = 6.4, 2.3
LEFT_EDGE = -5.0


def band_until(axes, x, low, high, x_end, colour):
    """The range band of a measured curve left of x_end, as a filled polygon."""
    keep = x <= x_end
    if keep.sum() < 2:
        keep[:2] = True
    xs, lo, hi = x[keep], low[keep], high[keep]
    points = [axes.c2p(a, b) for a, b in zip(xs, hi)] + [axes.c2p(a, b) for a, b in zip(xs[::-1], lo[::-1])]
    return Polygon(*points, stroke_width=0, fill_color=colour, fill_opacity=0.22)


class BaselineKernel(layout.QuickScene):
    def construct(self):
        # Measured: the five seeds of the baseline cell at every checkpoint after step 0.
        n100 = data.load_n100()
        cell = (n100["alpha"] == 1.0) & (n100["eta_kappa"] == 0.0)
        assert cell.sum() == 5
        x = np.log10(n100["step"][1:])
        S, R = n100["S_sum"][cell][:, 1:], n100["R_sum"][cell][:, 1:]

        # Measured: the events of the same five runs, and the share of the scale part at grokking.
        runs = data.load_report_runs()
        index = {name: i for i, name in enumerate(runs["run_id"])}
        rows = np.array([index[name] for name in n100["run_id"][cell]])
        assert not runs["censored_grok95"][rows].any()
        t_mem, t_grok = np.median(runs["t_mem"][rows]), np.median(runs["t_grok95"][rows])
        assert t_mem == int(data.deck_number("NfiveAOneMemMedian").replace(",", ""))
        assert t_grok == int(data.deck_number("NfiveAOneGrokMedian").replace(",", ""))
        S_g, R_g = runs["S_t_at_grok95"][rows], runs["R_t_at_grok95"][rows]
        share = np.median((S_g - 1) ** 2 / ((S_g - 1) ** 2 + 2 * S_g * R_g))
        assert round(100 * share) == int(data.deck_number("NtwoAOneGrok").rstrip("%"))

        # The axes leave room for the largest seed: S_t reaches 5.09 and R_t 0.080.
        top, _ = layout.step_axes(X_LENGTH, Y_LENGTH, y_range=(0, 6, 1), y_labels=(), labelled=())
        bottom, numbers = layout.step_axes(X_LENGTH, Y_LENGTH, y_range=(0, 0.09, 0.02), y_labels=(),
                                           labelled=(1, 100, 10_000))
        top.shift([LEFT_EDGE - top.c2p(0, 0)[0], 1.05 - top.c2p(0, 0)[1], 0])
        VGroup(bottom, numbers).shift([LEFT_EDGE - bottom.c2p(0, 0)[0], -2.15 - bottom.c2p(0, 0)[1], 0])
        y_numbers = VGroup(*[Text(f"{v:g}", font_size=SIZE, color=palette.INK).next_to(ax.c2p(0, v), LEFT, buff=0.2)
                             for ax, values in ((top, (1, 5)), (bottom, (0, 0.08))) for v in values])
        x_name = Text("step", font_size=SIZE, color=palette.INK).next_to(VGroup(bottom, numbers), DOWN, buff=0.1)
        s_name = MathTex("S_t", font_size=MATH + 8, color=palette.SCALE_TEXT).move_to(top.c2p(0.6, 4.6))
        r_name = MathTex("R_t", font_size=MATH + 8, color=palette.ROTATION_TEXT).move_to(bottom.c2p(0.6, 0.068))

        cursor = ValueTracker(0.0)
        curves = VGroup()
        for ax, values, colour in ((top, S, palette.SCALE), (bottom, R, palette.ROTATION)):
            median, low, high = np.median(values, axis=0), values.min(axis=0), values.max(axis=0)
            curves.add(always_redraw(lambda ax=ax, low=low, high=high, colour=colour:
                                     band_until(ax, x, low, high, cursor.get_value(), colour)))
            curves.add(always_redraw(lambda ax=ax, median=median, colour=colour:
                                     layout.curve_until(ax, x, median, cursor.get_value(), colour)))

        def markers(at, filled):
            group = VGroup()
            for ax, values, colour in ((top, S, palette.SCALE), (bottom, R, palette.ROTATION)):
                point = ax.c2p(np.log10(at), np.interp(np.log10(at), x, np.median(values, axis=0)))
                group.add(layout.event_marker(point, colour, filled))
            return group

        self.play(FadeIn(top), FadeIn(bottom), FadeIn(numbers), FadeIn(y_numbers), FadeIn(x_name), FadeIn(s_name),
                  FadeIn(r_name), run_time=1)
        self.add(curves)
        self.play(cursor.animate.set_value(np.log10(t_mem)), run_time=6, rate_func=linear)
        self.play(FadeIn(markers(t_mem, filled=False)), run_time=0.5)
        self.play(cursor.animate.set_value(np.log10(t_grok)), run_time=2, rate_func=linear)
        self.play(FadeIn(markers(t_grok, filled=True)), run_time=0.5)
        self.play(cursor.animate.set_value(np.log10(200_000)), run_time=1.6, rate_func=linear)
        self.wait(0.8)

        # The two parts of D_t^2 at grokking, as one bar.
        height, bar_x, base, width = 4.0, 3.3, -2.15, 1.6
        scale_bar = Rectangle(width=width, height=share * height, stroke_width=0, fill_color=palette.SCALE,
                              fill_opacity=1).move_to([bar_x, base + share * height / 2, 0])
        rotation_bar = Rectangle(width=width, height=(1 - share) * height, stroke_width=0, fill_color=palette.ROTATION,
                                 fill_opacity=1).move_to([bar_x, base + share * height + (1 - share) * height / 2, 0])
        title = VGroup(MathTex("D_t^2", font_size=MATH, color=palette.INK),
                       Text("at grokking", font_size=SIZE, color=palette.INK)).arrange(DOWN, buff=0.15)
        title.next_to(rotation_bar, UP, buff=0.25).align_to(rotation_bar, LEFT)
        percent = Text(data.deck_number("NtwoAOneGrok"), font_size=SIZE, color=palette.PAPER).move_to(scale_bar)
        scale_name = MathTex("(S_t - 1)^2", font_size=SIZE, color=palette.SCALE_TEXT).next_to(scale_bar, RIGHT, buff=0.25)
        rotation_name = MathTex("2 S_t R_t", font_size=SIZE, color=palette.ROTATION_TEXT)
        rotation_name.next_to(rotation_bar, RIGHT, buff=0.25)
        self.play(FadeIn(title), GrowFromEdge(scale_bar, DOWN), run_time=1.5)
        self.play(GrowFromEdge(rotation_bar, DOWN), FadeIn(percent), FadeIn(scale_name), FadeIn(rotation_name),
                  run_time=1)
        self.wait(10)
