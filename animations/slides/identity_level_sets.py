"""The distance mixes growth and rotation: lines of equal D_t in the plane of the two parts.

A backup slide of the deck. By report_v2 Equation eq:identity,
D_t^2 = (S_t - 1)^2 + 2 S_t R_t. So in the plane of the scale part (S_t - 1)^2
across and the shape part 2 S_t R_t up, drawn on one scale, every line of
equal D_t is the diagonal x + y = D_t^2. The grey lines are exact.

The first part is a schematic at D_t = 0.5, the distance of slide 8. Growth by
1.5 times puts all of D_t^2 = 0.25 in the scale part. A change of shape at a
fixed size puts all of it in the shape part, with R_t = 0.125. The second
part zooms out to the measured path of the baseline run ntk_N100_a1_wd0_s0
(alpha 1, width 100, eta lambda 0, seed 0) over its 307 checkpoints, read from
its committed seed 0 curve. At its last step the scale part is 12.5 and the
shape part 0.67, so the path stays close to the growth axis. The scene
asserts that the recorded D_sum of the run satisfies the identity at every
checkpoint.

Render from animations/:

    python -m manim -qh slides/identity_level_sets.py IdentityLevelSets
    python -m manim -qh -s slides/identity_level_sets.py IdentityLevelSets
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, DashedLine, Dot, FadeIn, FadeOut, Line, MathTex, Text,
                   ValueTracker, VGroup, VMobject, always_redraw, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, layout, palette

RUN = data.dataset_run_id(100, 1.0, 0.0)
SIZE = palette.LABEL_COLUMN
MATH = SIZE + 8
SIDE = 5.1
CORNER = np.array([-5.2, -2.3, 0])
SMALL, BIG = 0.3, 13.5
D_SCHEMATIC = 0.5
LEVELS = (0.25, 0.5, 1, 2, 3)


class IdentityLevelSets(layout.QuickScene):
    def construct(self):
        # Measured: every checkpoint of ntk_N100_a1_wd0_s0.
        run = data.load_curves(RUN)
        S, R, D = data.history(run, "S_sum", "R_sum", "D_sum")
        scale_part, rotation_part = (S - 1) ** 2, 2 * S * R
        assert np.allclose(D ** 2, scale_part + rotation_part, rtol=1e-5, atol=1e-6), "eq:identity fails"

        window = ValueTracker(SMALL)
        growth = ValueTracker(0.0)
        shape = ValueTracker(0.0)
        sweep = ValueTracker(0.0)

        def point(x, y):
            m = window.get_value()
            return CORNER + np.array([x / m * SIDE, y / m * SIDE, 0])

        def levels():
            m = window.get_value()
            group = VGroup()
            for d in LEVELS:
                c = d ** 2
                if c > 2 * m or c < 0.04 * m:
                    continue
                x1, x2 = max(0.0, c - m), min(c, m)
                group.add(DashedLine(point(x1, c - x1), point(x2, c - x2), color=palette.GRAY, stroke_width=3,
                                     dash_length=0.1))
            return group

        def path(x_end, y_end, colour):
            return Line(point(0, 0), point(x_end, y_end), color=colour, stroke_width=10)

        def measured():
            j = int(round(sweep.get_value()))
            m = window.get_value()
            keep = (scale_part[:j + 1] <= m) & (rotation_part[:j + 1] <= m)
            points = [point(x, y) for x, y in zip(scale_part[:j + 1][keep], rotation_part[:j + 1][keep])]
            line = VMobject(stroke_color=palette.INK, stroke_width=8)
            line.set_points_as_corners(points if len(points) > 1 else [point(0, 0)] * 2)
            return line

        axes = VGroup(Line(CORNER, CORNER + SIDE * RIGHT), Line(CORNER, CORNER + SIDE * UP)).set_stroke(palette.INK, 3)
        x_name = VGroup(Text("scale part", font_size=SIZE, color=palette.INK),
                        MathTex("(S_t - 1)^2", font_size=MATH, color=palette.SCALE_TEXT)).arrange(RIGHT, buff=0.25)
        x_name.next_to(axes[0], DOWN, buff=0.85)
        y_name = VGroup(Text("shape part", font_size=SIZE, color=palette.INK),
                        MathTex("2 S_t R_t", font_size=MATH, color=palette.ROTATION_TEXT)).arrange(RIGHT, buff=0.25)
        y_name.next_to(axes[1], UP, buff=0.3).align_to(axes[1], LEFT)

        grid = always_redraw(levels)
        grow_line = always_redraw(lambda: path(growth.get_value(), 0, palette.SCALE))
        shape_line = always_redraw(lambda: path(0, shape.get_value(), palette.ROTATION))
        run_line = always_redraw(measured)

        def key(colour, words):
            return VGroup(Line(ORIGIN_LEFT, ORIGIN_LEFT + 0.6 * RIGHT, color=colour, stroke_width=10),
                          Text(words, font_size=SIZE, color=palette.INK)).arrange(RIGHT, buff=0.25)

        level_key = VGroup(DashedLine(ORIGIN_LEFT, ORIGIN_LEFT + 0.6 * RIGHT, color=palette.GRAY, stroke_width=4,
                                      dash_length=0.1),
                           MathTex(r"D_t = 1,\ 2,\ 3", font_size=MATH, color=palette.GRAY)).arrange(RIGHT, buff=0.25)
        keys = VGroup(key(palette.SCALE, "change of scale"), key(palette.ROTATION, "change of shape"),
                      key(palette.INK, "baseline run"), level_key).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        keys.move_to([3.6, 0.6, 0])
        small_label = MathTex(r"D_t = 0.5", font_size=MATH, color=palette.GRAY)
        small_label.next_to(CORNER + 0.5 * SIDE * (UP + RIGHT) * 0.25 / SMALL, UP + RIGHT, buff=0.1)
        tag = layout.tag("schematic")

        self.play(FadeIn(axes), FadeIn(x_name), FadeIn(y_name), FadeIn(grid), FadeIn(small_label), FadeIn(tag),
                  run_time=1.5)
        self.add(grow_line, shape_line)
        self.play(growth.animate.set_value(D_SCHEMATIC ** 2), FadeIn(keys[0]), run_time=2.5)
        self.play(shape.animate.set_value(D_SCHEMATIC ** 2), FadeIn(keys[1]), run_time=2.5)
        self.wait(1.5)

        # Zoom out to the measured run.
        self.play(FadeOut(small_label), FadeOut(tag), run_time=0.5)
        self.play(window.animate.set_value(BIG), run_time=3,
                  rate_func=lambda t: (np.power(BIG / SMALL, t) - 1) / (BIG / SMALL - 1))
        ticks = VGroup()
        for value in (5, 10):
            ticks.add(Text(f"{value}", font_size=SIZE, color=palette.INK).next_to(point(value, 0), DOWN, buff=0.2))
            ticks.add(Text(f"{value}", font_size=SIZE, color=palette.INK).next_to(point(0, value), LEFT, buff=0.2))
        ticks.add(Text("0", font_size=SIZE, color=palette.INK).next_to(point(0, 0), DOWN + LEFT, buff=0.1))
        self.play(FadeIn(ticks), FadeIn(keys[3]), run_time=0.5)

        self.add(run_line)
        self.play(sweep.animate.set_value(len(S) - 1), FadeIn(keys[2]), run_time=4.5, rate_func=linear)
        self.add(Dot(point(scale_part[-1], rotation_part[-1]), radius=0.1, color=palette.INK))
        self.wait(3.5)


# The left end of a key's line, before the key is arranged.
ORIGIN_LEFT = np.array([0.0, 0.0, 0.0])
