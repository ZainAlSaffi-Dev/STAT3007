"""In the lazy regime the kernel stays fixed: a larger alpha changes it less, and the lazier network groks later.

Slide 6 of the deck. The first part is a schematic, and it says so on screen.
The same kernel, drawn as an ellipse, is trained at alpha 0.5, 1 and 2, and it
changes by an amount that shrinks as 1 / alpha: it turns by 30 / alpha degrees
and grows by 1 + 0.4 / alpha. This only illustrates the lazy limit that the
slide cites (Chizat, Oyallon and Bach, 2019), in which the parameters move by
an amount of order 1 / alpha. It is not a measurement. In our runs the kernel
still grows at alpha 2, and R_t at the last step is about the same at all
three alphas (0.066, 0.074 and 0.065 for seed 0).

The second part is measured. It shows the test accuracy of the baseline runs
with seed 0, ntk_N100_a{0.5, 1, 2}_wd0_s0, from their committed seed 0 curves,
with the grokking steps from report_runs.csv: 3,662 at alpha 0.5 and 15,395 at
alpha 1. The run at alpha 2 does not reach test accuracy 0.95 within 200,000
steps. The greys are those of the report figure E1.

Render from animations/:

    uv run manim -qh slides/lazy_versus_rich.py LazyVersusRich
    uv run manim -qh -s slides/lazy_versus_rich.py LazyVersusRich
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, DashedVMobject, FadeIn, FadeOut, Line, MathTex, Scene, Text,
                   ValueTracker, VGroup, always_redraw, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, layout, palette

SIZE = palette.LABEL_COLUMN
MATH = SIZE + 8
ALPHAS = (0.5, 1.0, 2.0)
COLUMNS = (-4.4, 0.0, 4.4)
UNIT = 0.55


def turned(K, angle):
    """K turned by the angle: Q K Q^T for the rotation Q."""
    c, s = np.cos(angle), np.sin(angle)
    Q = np.array([[c, -s], [s, c]])
    return Q @ K @ Q.T


class LazyVersusRich(Scene):
    def construct(self):
        self.schematic()
        self.measured()

    def schematic(self):
        K0 = turned(np.diag([2.0, 1.0]), np.deg2rad(20))
        progress = ValueTracker(0.0)

        def kernel(alpha):
            # Schematic: the change shrinks as 1 / alpha.
            t = progress.get_value()
            return (1 + t * 0.4 / alpha) * turned(K0, t * np.deg2rad(30 / alpha))

        centres = [np.array([x, 0.2, 0]) for x in COLUMNS]
        ghosts = VGroup(*[DashedVMobject(layout.kernel_ellipse(K0, UNIT, palette.GRAY, c, width=4)[0], num_dashes=40)
                          for c in centres])
        shapes = VGroup(*[always_redraw(lambda a=a, c=c: layout.kernel_ellipse(kernel(a), UNIT, palette.ALPHA_GREYS[a], c))
                          for a, c in zip(ALPHAS, centres)])
        names = VGroup(*[MathTex(rf"\alpha = {a:g}", font_size=MATH, color=palette.INK).move_to(c + 2.5 * UP)
                         for a, c in zip(ALPHAS, centres)])
        arrow = Arrow(np.array([COLUMNS[0] - 0.8, -2.45, 0]), np.array([COLUMNS[2] + 0.8, -2.45, 0]), buff=0,
                      color=palette.INK, stroke_width=6)
        lazier = Text("lazier", font_size=SIZE, color=palette.INK).next_to(arrow, DOWN, buff=0.2)
        tag = layout.tag("schematic")

        self.play(FadeIn(ghosts), FadeIn(shapes), FadeIn(names), FadeIn(tag), run_time=1.5)
        self.play(FadeIn(arrow), FadeIn(lazier), run_time=1)
        self.play(progress.animate.set_value(1.0), run_time=6)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in (ghosts, shapes, names, arrow, lazier, tag)], run_time=1)

    def measured(self):
        runs = data.load_report_runs()
        names = list(runs["run_id"])
        axes, numbers = layout.step_axes(x_length=10.6, y_length=4.0)
        VGroup(axes, numbers).shift(0.35 * LEFT + (2.15 - axes.c2p(0, 1)[1]) * UP)
        x_name = Text("step", font_size=SIZE, color=palette.INK).next_to(axes.x_axis.get_end(), RIGHT, buff=0.3)
        y_name = Text("test accuracy", font_size=SIZE, color=palette.INK)
        y_name.next_to(axes.c2p(0, 1), UP, buff=0.35).align_to(numbers, LEFT)

        cursor = ValueTracker(0.0)
        lines, markers, keys = VGroup(), VGroup(), VGroup()
        for alpha in ALPHAS:
            # Measured: ntk_N100_a{alpha}_wd0_s0 at every checkpoint after step 0.
            name = data.dataset_run_id(100, alpha, 0.0)
            steps, test = data.history(data.load_curves(name), "step", "test_acc")
            x, y = np.log10(steps[1:]), test[1:]
            colour = palette.ALPHA_GREYS[alpha]
            lines.add(always_redraw(lambda x=x, y=y, colour=colour: layout.curve_until(axes, x, y, cursor.get_value(),
                                                                                      colour)))
            row = names.index(name)
            if not runs["censored_grok95"][row]:
                x_grok = np.log10(runs["t_grok95"][row])
                markers.add(layout.event_marker(axes.c2p(x_grok, np.interp(x_grok, x, y)), colour, filled=True))
            keys.add(VGroup(Line(ORIGIN_LEFT, ORIGIN_LEFT + 0.6 * RIGHT, color=colour, stroke_width=10),
                            MathTex(rf"\alpha = {alpha:g}", font_size=MATH, color=palette.INK)).arrange(RIGHT, buff=0.25))
        keys.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(axes.c2p(1.25, 0.62))

        self.play(FadeIn(axes), FadeIn(numbers), FadeIn(x_name), FadeIn(y_name), FadeIn(keys), run_time=1)
        self.add(lines)
        # The markers appear as the sweep passes each grokking step.
        x_marks = [axes.p2c(m.get_center())[0] for m in markers]
        for x_mark, marker in zip(x_marks, markers):
            self.play(cursor.animate.set_value(x_mark), run_time=max(0.6, 1.6 * (x_mark - cursor.get_value())),
                      rate_func=linear)
            self.play(FadeIn(marker, scale=1.4), run_time=0.4)
        self.play(cursor.animate.set_value(np.log10(200_000)), run_time=2.2, rate_func=linear)
        self.wait(3)


# The left end of a key's line, before the key is arranged.
ORIGIN_LEFT = np.array([0.0, 0.0, 0.0])
