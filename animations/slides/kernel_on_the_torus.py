"""The probe kernel of three cells from step 0 to step 30,000, averaged over offsets.

Every heat map, bar and number here is a measurement. They come from the dense
runs that repoduced-code/kernel_snapshots.py writes. Those runs are alpha 1,
width 100, NTK parameterisation, seed 0 and base rate 100, with eta lambda
equal to 0, 0.0003 and 0.001, checkpointed every 250 steps. They repeat the
Tier 1 grid cells of the same name, and their histories match the saved grid
runs at every shared checkpoint. The probe is the first 256 training pairs,
as in the Tier 1 grid.

A heat map shows the kernel averaged over probe pairs with the same offsets
(a - a', b - b') mod p. See common/kernels.py. The picture has 23 by 23
blocks, and each block averages between 108 and 256 pairs.

Render from animations/:

    uv run manim -qh slides/kernel_on_the_torus.py KernelOnTheTorus
    uv run manim -qh -s slides/kernel_on_the_torus.py KernelOnTheTorus
"""

import sys
from pathlib import Path

import numpy as np
from manim import (ORIGIN, DOWN, LEFT, RIGHT, UL, UP, UR, DL, DR, DashedLine, DecimalNumber, FadeIn,
                   FadeOut, Create, Integer, Line, MathTex, Rectangle, Scene, Transform, ValueTracker,
                   VGroup, Group, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, kernels, layout, notation, palette

# The three dense runs and the value of eta lambda each was trained with.
RUNS = data.DENSE_RUNS

# Colour limits. They were read off the dense runs and are not tuned per frame,
# so a colour means the same value in every frame.
# The step 0 kernel: the cross of shared inputs averages about 0.011 and the
# zero-offset block 0.030, which is drawn at full colour.
VMAX_KERNEL = 0.015
# The change since step 0: 99.9 percent of blocks off the cross, over all
# three runs and all checkpoints, lie within plus or minus 0.001.
VMAX_CHANGE = 0.001

SWEEP_SECONDS = 16


class Run:
    """One dense run, reduced to what the scene draws at each checkpoint."""

    def __init__(self, name, torus):
        run = data.load_run(name)
        steps, K = data.load_kernels(name)
        h = run["history"]
        if list(h["step"]) != list(steps):
            raise ValueError(f"{name}: the kernel checkpoints and the history checkpoints differ.")
        self.steps = np.asarray(steps)
        k_0 = kernels.unit(kernels.centre(K[0].astype(float)))
        self.kernel_0 = torus(k_0)
        # Change of the normalised centred kernel since step 0, at every checkpoint.
        self.change = np.array([torus(kernels.unit(kernels.centre(k.astype(float))) - k_0) for k in K])
        self.norm_ratio = np.exp(np.asarray(h["S_c"]))
        # R at step 0 is recorded as -2.9e-6, float32 rounding of a cosine of 1.
        # Clipping at 0 keeps the counter from showing a minus sign.
        self.R = np.maximum(np.asarray(h["R_c"]), 0.0)
        self.A = np.asarray(h["A_t"])
        self.test_acc = np.asarray(h["test_acc"])


def flipped(image_values):
    """Put the offset a - a' = +11 in the top row, so the vertical axis increases upwards."""
    return image_values[::-1]


class KernelOnTheTorus(Scene):
    def construct(self):
        a, b = data.load_probe(RUNS[0][0])
        p = data.load_run(RUNS[0][0])["config"]["p"]
        torus = kernels.Torus(a, b, p)
        runs = [Run(name, torus) for name, _ in RUNS]
        target = torus(kernels.unit(kernels.target_gram((a + b) % p, p)))

        self.introduce(runs[0].kernel_0, target)
        self.sweep(runs)

    def axes_labels(self, panel):
        """The two offset labels and the zero marks on the edges of a panel."""
        side = panel.height
        left = MathTex(notation.OFFSET_A, color=palette.INK, font_size=palette.LABEL)
        left.rotate(np.pi / 2).next_to(panel, LEFT, buff=0.35)
        bottom = MathTex(notation.OFFSET_B, color=palette.INK, font_size=palette.LABEL)
        bottom.next_to(panel, DOWN, buff=0.35)
        zero_left = MathTex("0", color=palette.GRAY, font_size=palette.LABEL - 6)
        zero_left.next_to(panel.get_left(), LEFT, buff=0.08)
        zero_bottom = MathTex("0", color=palette.GRAY, font_size=palette.LABEL - 6)
        zero_bottom.next_to(panel.get_bottom(), DOWN, buff=0.08)
        return VGroup(left, bottom, zero_left, zero_bottom)

    def introduce(self, kernel_0, target):
        heading = layout.title("The kernel, averaged over offsets")
        side = 4.2

        # Measured: dense_N100_a1_wd0_s0, step 0. The step 0 kernel is the same in all three runs.
        k_panel = layout.heat_map(flipped(kernel_0), VMAX_KERNEL, side).move_to(3.3 * LEFT + 0.1 * DOWN)
        k_name = MathTex(notation.K_UNIT + "_0", color=palette.INK, font_size=palette.CAPTION)
        k_name.next_to(k_panel, UP, buff=0.2)
        k_axes = self.axes_labels(k_panel)

        # Exact: the centred label kernel of the same probe pairs.
        t_panel = layout.heat_map(flipped(target), np.abs(target).max(), side).move_to(3.3 * RIGHT + 0.1 * DOWN)
        t_name = MathTex(notation.TARGET_UNIT, color=palette.INK, font_size=palette.CAPTION)
        t_name.next_to(t_panel, UP, buff=0.2)
        t_axes = self.axes_labels(t_panel)
        t_line = MathTex(notation.SUM_LINE, color=palette.POSITIVE, font_size=palette.LABEL)
        t_line.next_to(t_panel, RIGHT, buff=0.2).shift(0.8 * DOWN)

        self.play(FadeIn(heading))
        self.play(FadeIn(k_panel), FadeIn(k_name), FadeIn(k_axes))
        words = layout.caption("At step 0 the kernel is large where two pairs share a or b.")
        self.play(FadeIn(words))
        self.wait(2.5)

        self.play(FadeIn(t_panel), FadeIn(t_name), FadeIn(t_axes), FadeIn(t_line))
        new_words = layout.caption("The target is large where the two sums agree mod p.")
        self.play(Transform(words, new_words))
        self.wait(2.5)

        new_words = layout.caption("Each block is an average over pairs with the same offsets.")
        self.play(Transform(words, new_words))
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in [heading, k_panel, k_name, k_axes, t_panel, t_name,
                                         t_axes, t_line, words]])

    def sweep(self, runs):
        heading = layout.title("How the kernel turns in three cells")
        what = MathTex(notation.KERNEL_CHANGE, color=palette.INK, font_size=palette.CAPTION)
        what.next_to(heading, DOWN, aligned_edge=LEFT, buff=0.25)

        index = ValueTracker(0)

        def i_now():
            return int(round(index.get_value()))

        step_word = layout.label("step")
        step_value = Integer(0, color=palette.INK, font_size=palette.CAPTION)
        step_value.add_updater(lambda m: m.set_value(int(runs[0].steps[i_now()])))
        # The word stays put and leaves room on its right for "30,000".
        step_word.to_corner(UR, buff=0.5).shift(1.9 * LEFT)
        step_value.next_to(step_word, RIGHT, buff=0.2)
        step_value.add_updater(lambda m: m.next_to(step_word, RIGHT, buff=0.2))
        step_group = VGroup(step_word, step_value)

        side, gap = 3.0, 4.55
        columns = Group()
        panels = []
        for k, (run, (_, decay)) in enumerate(zip(runs, RUNS)):
            centre = (k - 1) * gap * RIGHT + 0.2 * UP
            panel = layout.heat_map(flipped(run.change[0]), VMAX_CHANGE, side).move_to(centre)
            panel.add_updater(lambda m, run=run: layout.set_pixels(m, flipped(run.change[i_now()]), VMAX_CHANGE))
            frame = Rectangle(width=side, height=side, stroke_color=palette.GRID, stroke_width=1.5).move_to(centre)
            name = MathTex(notation.DECAY + " = " + decay, color=palette.INK, font_size=palette.LABEL)
            name.next_to(frame, UP, buff=0.15)

            # Scale: a bar of length e^{S_t}, with a tick at 1, the size at step 0.
            unit_len = 0.5
            origin = frame.get_corner(DL) + 0.5 * DOWN
            bar = Rectangle(width=unit_len, height=0.22, stroke_width=0, fill_color=palette.SCALE,
                            fill_opacity=1)
            bar.move_to(origin, aligned_edge=LEFT)
            bar.add_updater(lambda m, run=run, origin=origin: m.stretch_to_fit_width(
                unit_len * run.norm_ratio[i_now()]).move_to(origin, aligned_edge=LEFT))
            tick = Line(origin + unit_len * RIGHT + 0.2 * UP, origin + unit_len * RIGHT + 0.2 * DOWN,
                        color=palette.INK, stroke_width=2)

            # Rotation, alignment and test accuracy, read from the history.
            r_label = MathTex("R_t =", color=palette.ROTATION, font_size=palette.LABEL)
            r_value = DecimalNumber(run.R[0], num_decimal_places=3, color=palette.ROTATION,
                                    font_size=palette.LABEL)
            a_label = MathTex("A_t =", color=palette.ALIGNMENT, font_size=palette.LABEL)
            a_value = DecimalNumber(run.A[0], num_decimal_places=3, color=palette.ALIGNMENT,
                                    font_size=palette.LABEL)
            numbers = VGroup(r_label, r_value, a_label, a_value).arrange(RIGHT, buff=0.12)
            numbers[2].shift(0.3 * RIGHT)
            numbers[3].shift(0.3 * RIGHT)
            numbers.next_to(origin, DOWN, buff=0.35, aligned_edge=LEFT)
            r_value.add_updater(lambda m, run=run: m.set_value(run.R[i_now()]))
            a_value.add_updater(lambda m, run=run: m.set_value(run.A[i_now()]))

            acc_label = layout.label("test accuracy", palette.TEST)
            acc_value = DecimalNumber(run.test_acc[0], num_decimal_places=2, color=palette.TEST,
                                      font_size=palette.LABEL)
            acc = VGroup(acc_label, acc_value).arrange(RIGHT, buff=0.15)
            acc.next_to(numbers, DOWN, buff=0.25, aligned_edge=LEFT)
            acc_value.add_updater(lambda m, run=run: m.set_value(run.test_acc[i_now()]))

            columns.add(Group(panel, frame, name, bar, tick, numbers, acc))
            panels.append(frame)

        bar_name = MathTex(notation.NORM_RATIO, color=palette.SCALE, font_size=palette.LABEL)
        bar_name.next_to(columns[0][3], LEFT, buff=0.2)

        self.play(FadeIn(heading), FadeIn(what), FadeIn(step_group), FadeIn(columns), FadeIn(bar_name))
        words = layout.caption("Colour shows how the direction of the kernel has changed.")
        self.play(FadeIn(words))
        self.wait(1.5)
        self.play(index.animate.set_value(len(runs[0].steps) - 1), run_time=SWEEP_SECONDS, rate_func=linear)
        self.wait(1)

        # The two lines through the zero offset, drawn on the middle panel, where they are strongest. The rows run
        # from a - a' = +11 at the top, so the sum line a + b = a' + b' runs from the
        # top left to the bottom right and the difference line from the bottom left
        # to the top right.
        middle = panels[1]
        sum_line = DashedLine(middle.get_corner(UL), middle.get_corner(DR), color=palette.INK,
                              stroke_width=2.5, dash_length=0.12)
        diff_line = DashedLine(middle.get_corner(DL), middle.get_corner(UR), color=palette.GRAY,
                               stroke_width=2.5, dash_length=0.12)
        sum_key = DashedLine(ORIGIN, 0.6 * RIGHT, color=palette.INK, stroke_width=2.5, dash_length=0.12)
        sum_name = MathTex(notation.SUM_LINE, color=palette.INK, font_size=palette.LABEL)
        diff_key = DashedLine(ORIGIN, 0.6 * RIGHT, color=palette.GRAY, stroke_width=2.5, dash_length=0.12)
        diff_name = MathTex(notation.DIFFERENCE_LINE, color=palette.GRAY, font_size=palette.LABEL)
        legend = VGroup(VGroup(sum_key, sum_name).arrange(RIGHT, buff=0.2),
                        VGroup(diff_key, diff_name).arrange(RIGHT, buff=0.2)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_edge(RIGHT, buff=0.5).align_to(step_word, UP).shift(0.75 * DOWN)
        # Mean change on each line at step 30,000, against the spread of the other blocks
        # (dense runs): eta lambda 0 has +0.00036 and +0.00034 against a standard deviation
        # of 0.00017, 0.0003 has +0.00087 and +0.00078 against 0.00019, and 0.001 has
        # +0.00062 and +0.00045 against 0.00022. So both lines are there in every cell.
        new_words = layout.caption("In all three cells the change runs along two lines.")
        self.play(Create(sum_line), Create(diff_line), FadeIn(legend),
                  Transform(words, new_words))
        self.wait(2.5)
        new_words = layout.caption("Only the sum line appears in the target.")
        self.play(Transform(words, new_words))
        self.wait(3)
