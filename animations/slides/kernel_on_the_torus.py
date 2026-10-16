"""On this task, the change of shape has a visible pattern: the kernel averaged over offsets.

Slide 13 of the deck. Every image is a measurement. The kernel is the report's,
the tangent kernel of the summed logits over sqrt(p), on all 529 pairs. It is
rebuilt with kernels.sum_kernel from the saved weights of the baseline run
ntk_N100_a1_wd0_s0 (alpha 1, width 100, eta lambda 0, seed 0). The report
measures the kernel on the 53 test pairs of a run, which are too few for this
picture, so the scene uses the same kernel on every pair.

An image averages the kernel over pairs of pairs with the same offsets
(a - a', b - b') mod 23, as kernels.Torus does. With all 529 pairs, each block
averages 529 entries. At step 0 the normalised kernel K_0 / ||K_0|| is a cross,
because two pairs that share a or b are alike. The label kernel Y Y^T is one on
the line a + b = a' + b' mod 23. The change of the normalised kernel,
K_t / ||K_t|| - K_0 / ||K_0||, is the change of shape that R_t measures, since
its squared norm is 2 R_t. It falls on the cross and rises everywhere else,
most on two diagonals: the sum line and the difference line
a - b = a' - b', which the labels do not contain. The scene asserts that at
the last step shown both lines lie more than two SD above the other blocks
off the cross, and that the cross falls. At step 31,000 the two lines average
+3.5e-4 and +3.4e-4, against +2.1e-4 (SD 0.4e-4) for the other blocks and
-9.0e-4 on the cross.

Render from animations/:

    uv run manim -qh slides/kernel_on_the_torus.py KernelOnTheTorus
    uv run manim -qh -s slides/kernel_on_the_torus.py KernelOnTheTorus
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, DashedLine, DecimalNumber, FadeIn, FadeOut, Create, Integer, MathTex,
                   Rectangle, Scene, Square, Text, ValueTracker, VGroup, Group)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, kernels, layout, notation, palette

RUN = data.dataset_run_id(100, 1.0, 0.0)
P = 23
LAST_STEP = 31_000
SIZE = palette.LABEL_WIDE + 4
MATH = SIZE + 6


def flipped(image_values):
    """Put the offset a - a' = +11 in the top row, so the vertical axis increases upwards."""
    return image_values[::-1]


class KernelOnTheTorus(Scene):
    def construct(self):
        # Measured: the weights of ntk_N100_a1_wd0_s0 at its saved steps up to step 31,000.
        steps, W1, W2 = data.load_weights(RUN)
        shown = int(np.flatnonzero(steps == LAST_STEP)[0]) + 1
        steps, W1, W2 = steps[:shown], W1[:shown], W2[:shown]
        X, a, b = kernels.one_hot_pairs(P)
        torus = kernels.Torus(a, b, P)
        k0 = kernels.unit(kernels.sum_kernel(W1[0], W2[0], X))
        start = torus(k0)
        changes = np.array([torus(kernels.unit(kernels.sum_kernel(W1[k], W2[k], X)) - k0) for k in range(shown)])
        Y = np.eye(P)[(a + b) % P]
        target = torus(Y @ Y.T)

        # Measured: the test accuracy of the same run at the saved steps.
        curve_steps, test = data.history(data.load_curves(RUN), "step", "test_acc")
        at = np.searchsorted(curve_steps, steps)
        assert np.array_equal(curve_steps[at], steps), "every saved step is a checkpoint of the curve"
        test = test[at]

        # The claim of the caption, checked on the last step shown.
        off_a, off_b = np.meshgrid(np.arange(P) - P // 2, np.arange(P) - P // 2, indexing="ij")
        cross = (off_a == 0) | (off_b == 0)
        sum_line = ((off_a + off_b) % P == 0) & ~cross
        diff_line = ((off_a - off_b) % P == 0) & ~cross
        rest = ~(cross | sum_line | diff_line)
        last = changes[-1]
        floor = last[rest].mean() + 2 * last[rest].std()
        assert last[sum_line].mean() > floor and last[diff_line].mean() > floor and last[cross].mean() < 0

        # Colour limits read off the data, the same in every frame. The cross of K_0 is drawn at 80 percent
        # ink. The change is at full colour at 1.15 times the larger of the two line means at step 31,000,
        # so the cross, which falls further, saturates in grey.
        vmax_start = start[cross & ~((off_a == 0) & (off_b == 0))].mean() / 0.8
        vmax_change = 1.15 * max(last[sum_line].mean(), last[diff_line].mean())

        self.introduce(start, target, vmax_start)
        self.sweep(steps, changes, test, vmax_change)

    def axis_names(self, panel):
        rows = MathTex(notation.OFFSET_A, font_size=MATH, color=palette.INK).rotate(np.pi / 2)
        rows.next_to(panel, LEFT, buff=0.25)
        columns = MathTex(notation.OFFSET_B, font_size=MATH, color=palette.INK).next_to(panel, DOWN, buff=0.25)
        return VGroup(rows, columns)

    def introduce(self, start, target, vmax_start):
        side = 4.6
        left = layout.grey_map(flipped(start), vmax_start, side).move_to([-3.4, 0.1, 0])
        right = layout.grey_map(flipped(target), 1.0, side).move_to([3.4, 0.1, 0])
        frames = Group(*[Square(side, stroke_color=palette.GRID, stroke_width=2).move_to(p) for p in (left, right)])
        left_title = Text("the kernel at step 0", font_size=SIZE, color=palette.INK).next_to(left, UP, buff=0.3)
        right_title = VGroup(Text("the label kernel", font_size=SIZE, color=palette.INK),
                             MathTex(r"YY^\top", font_size=MATH, color=palette.INK)).arrange(RIGHT, buff=0.25)
        right_title.next_to(right, UP, buff=0.3)
        left_note = VGroup(Text("pairs that share", font_size=SIZE, color=palette.GRAY),
                           MathTex("a", font_size=MATH, color=palette.GRAY),
                           Text("or", font_size=SIZE, color=palette.GRAY),
                           MathTex("b", font_size=MATH, color=palette.GRAY)).arrange(RIGHT, buff=0.18)
        right_note = MathTex(notation.SUM_LINE, font_size=MATH, color=palette.GRAY)
        left_axes, right_axes = self.axis_names(left), self.axis_names(right)
        left_note.next_to(left_axes[1], DOWN, buff=0.2)
        right_note.next_to(right_axes[1], DOWN, buff=0.2)

        self.play(FadeIn(left), FadeIn(frames[0]), FadeIn(left_title), FadeIn(left_axes), run_time=1.2)
        self.play(FadeIn(left_note), run_time=0.8)
        self.wait(1.8)
        self.play(FadeIn(right), FadeIn(frames[1]), FadeIn(right_title), FadeIn(right_axes), FadeIn(right_note),
                  run_time=1.2)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in (left, right, frames, left_title, right_title, left_note, right_note,
                                         left_axes, right_axes)], run_time=0.8)

    def sweep(self, steps, changes, test, vmax):
        side = 5.6
        index = ValueTracker(0)

        def now():
            return int(round(index.get_value()))

        panel = layout.heat_map(flipped(changes[0]), vmax, side, palette.GRAY, palette.ROTATION)
        panel.move_to([-2.6, -0.05, 0])
        panel.add_updater(lambda m: layout.set_pixels(m, flipped(changes[now()]), vmax, palette.GRAY,
                                                      palette.ROTATION))
        frame = Square(side, stroke_color=palette.GRID, stroke_width=2).move_to(panel)
        title = VGroup(Text("change of shape since step 0", font_size=SIZE, color=palette.INK))
        title.next_to(panel, UP, buff=0.25)
        axes = self.axis_names(panel)

        x0 = 1.6
        step_word = Text("step", font_size=SIZE, color=palette.INK).move_to([x0, 2.6, 0], aligned_edge=LEFT)
        step_value = Integer(0, font_size=MATH, color=palette.INK)
        step_value.add_updater(lambda m: m.set_value(int(steps[now()])).next_to(step_word, RIGHT, buff=0.25))
        acc_word = Text("test accuracy", font_size=SIZE, color=palette.INK).move_to([x0, 1.8, 0], aligned_edge=LEFT)
        acc_value = DecimalNumber(test[0], num_decimal_places=2, font_size=MATH, color=palette.INK)
        acc_value.add_updater(lambda m: m.set_value(test[now()]).next_to(acc_word, RIGHT, buff=0.25))
        step_value.next_to(step_word, RIGHT, buff=0.25)
        acc_value.next_to(acc_word, RIGHT, buff=0.25)

        def swatch(colour):
            return Rectangle(width=0.5, height=0.4, stroke_width=0, fill_color=colour, fill_opacity=1)

        colours = VGroup(VGroup(swatch(palette.ROTATION), Text("more alike", font_size=SIZE, color=palette.INK)),
                         VGroup(swatch(palette.GRAY), Text("less alike", font_size=SIZE, color=palette.INK)))
        for item in colours:
            item.arrange(RIGHT, buff=0.25)
        colours.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([x0, 0.55, 0], aligned_edge=LEFT)

        self.play(FadeIn(panel), FadeIn(frame), FadeIn(title), FadeIn(axes), FadeIn(step_word), FadeIn(step_value),
                  FadeIn(acc_word), FadeIn(acc_value), FadeIn(colours), run_time=1)
        self.play(index.animate.set_value(len(steps) - 1), run_time=12, rate_func=lambda t: t)
        self.wait(0.8)

        # The two lines through the zero offset. Rows run from a - a' = +11 at the top, so the sum line
        # runs from the top left to the bottom right, and the difference line from the bottom left to the top right.
        sum_line = DashedLine(frame.get_corner(UP + LEFT), frame.get_corner(DOWN + RIGHT), color=palette.INK,
                              stroke_width=3, dash_length=0.14)
        diff_line = DashedLine(frame.get_corner(DOWN + LEFT), frame.get_corner(UP + RIGHT), color=palette.GRAY,
                               stroke_width=3, dash_length=0.14)
        keys = VGroup(VGroup(DashedLine(ORIGIN_LEFT, ORIGIN_LEFT + 0.6 * RIGHT, color=palette.INK, stroke_width=3,
                                        dash_length=0.12), MathTex(notation.SUM_LINE, font_size=MATH, color=palette.INK)),
                      VGroup(DashedLine(ORIGIN_LEFT, ORIGIN_LEFT + 0.6 * RIGHT, color=palette.GRAY, stroke_width=3,
                                        dash_length=0.12), MathTex(notation.DIFFERENCE_LINE, font_size=MATH,
                                                                   color=palette.GRAY)))
        for item in keys:
            item.arrange(RIGHT, buff=0.25)
        keys.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([x0, -1.55, 0], aligned_edge=LEFT)
        self.play(Create(sum_line), Create(diff_line), FadeIn(keys), run_time=1.5)
        self.wait(6.4)


# The left end of a key's line, before the key is arranged.
ORIGIN_LEFT = np.array([0.0, 0.0, 0.0])
