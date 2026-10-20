"""Seen as arrows, the three numbers are a length and two angles.

Slide 10 of the deck. This is a schematic, and it says so on screen. With the
Frobenius inner product, each kernel matrix is one arrow. By report_v2
Equation eq:statistics, the scale S_t is the ratio of the lengths of the
arrows of K_t and K_0, the rotation R_t is one minus the cosine of the angle
between them, and the alignment A_t is the cosine of the angle between the
centred kernel C K_t C and the centred label kernel C Y Y^T C. The plane only
pictures these arrows, and the angles are chosen for the picture: 20 degrees
between K_t and K_0, so R_t = 0.06, and 66 degrees between the centred arrows,
so A_t = 0.41. The numbers on screen are computed from the drawn arrows. The
last beat stretches the kernel by 1.4 times, which changes S_t and leaves both
angles, and so R_t and A_t, as they were.

Render from animations/:

    uv run manim -qh slides/kernel_as_arrows.py KernelAsArrows
    uv run manim -qh -s slides/kernel_as_arrows.py KernelAsArrows
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Angle, Arrow, DecimalNumber, FadeIn, GrowArrow, Line, MathTex, 
                   Text, ValueTracker, VGroup, always_redraw)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import layout, palette

SIZE = palette.LABEL_WIDE + 4
MATH = SIZE + 8
LEFT_ORIGIN = np.array([-6.2, -1.5, 0])
RIGHT_ORIGIN = np.array([1.2, -1.5, 0])
K0_LENGTH, KT_LENGTH = 2.6, 3.64
K0_ANGLE, KT_ANGLE = np.deg2rad(18), np.deg2rad(38)
CENTRED_ANGLE, LABEL_ANGLE = np.deg2rad(12), np.deg2rad(78)
STRETCH = 1.4


def direction(angle):
    return np.array([np.cos(angle), np.sin(angle), 0])


def arrow(origin, length, angle, colour):
    return Arrow(origin, origin + length * direction(angle), buff=0, color=colour, stroke_width=7,
                 max_tip_length_to_length_ratio=0.12)


class KernelAsArrows(layout.QuickScene):
    def construct(self):
        stretch = ValueTracker(1.0)
        k0 = arrow(LEFT_ORIGIN, K0_LENGTH, K0_ANGLE, palette.GRAY)
        kt = always_redraw(lambda: arrow(LEFT_ORIGIN, stretch.get_value() * KT_LENGTH, KT_ANGLE, palette.INK))
        k0_name = MathTex("K_0", font_size=MATH, color=palette.GRAY).next_to(k0.get_end(), RIGHT, buff=0.15)
        kt_name = always_redraw(lambda: MathTex("K_t", font_size=MATH, color=palette.INK)
                                .next_to(kt.get_end(), UP + RIGHT, buff=0.1))
        theta = Angle(Line(LEFT_ORIGIN, LEFT_ORIGIN + direction(K0_ANGLE)),
                      Line(LEFT_ORIGIN, LEFT_ORIGIN + direction(KT_ANGLE)), radius=1.3, color=palette.ROTATION,
                      stroke_width=6)
        theta_name = MathTex(r"\theta", font_size=MATH, color=palette.ROTATION_TEXT)
        theta_name.move_to(LEFT_ORIGIN + 1.7 * direction((K0_ANGLE + KT_ANGLE) / 2))

        ct = always_redraw(lambda: arrow(RIGHT_ORIGIN, stretch.get_value() * KT_LENGTH * 0.75, CENTRED_ANGLE,
                                         palette.INK))
        label = arrow(RIGHT_ORIGIN, 3.0, LABEL_ANGLE, palette.ALIGNMENT)
        ct_name = always_redraw(lambda: MathTex(r"CK_tC", font_size=MATH, color=palette.INK)
                                .next_to(ct.get_end(), DOWN + RIGHT, buff=0.1))
        label_name = MathTex(r"CYY^\top C", font_size=MATH, color=palette.ALIGNMENT_TEXT)
        label_name.next_to(label.get_end(), RIGHT, buff=0.15)
        phi = Angle(Line(RIGHT_ORIGIN, RIGHT_ORIGIN + direction(CENTRED_ANGLE)),
                    Line(RIGHT_ORIGIN, RIGHT_ORIGIN + direction(LABEL_ANGLE)), radius=1.1, color=palette.ALIGNMENT,
                    stroke_width=6)
        phi_name = MathTex(r"\phi", font_size=MATH, color=palette.ALIGNMENT_TEXT)
        phi_name.move_to(RIGHT_ORIGIN + 1.5 * direction((CENTRED_ANGLE + LABEL_ANGLE) / 2))
        right_title = Text("after centring", font_size=SIZE, color=palette.INK).move_to([3.4, 3.3, 0])
        left_title = Text("the kernel", font_size=SIZE, color=palette.INK).move_to([-3.6, 3.3, 0])

        # The three numbers, computed from the drawn arrows.
        def value(compute):
            number = DecimalNumber(compute(), num_decimal_places=2, font_size=MATH, color=palette.INK)
            number.add_updater(lambda m: m.set_value(compute()))
            return number

        readouts = VGroup(
            VGroup(MathTex(r"S_t = \frac{\lVert K_t \rVert_F}{\lVert K_0 \rVert_F} =", font_size=MATH,
                           color=palette.SCALE_TEXT),
                   value(lambda: stretch.get_value() * KT_LENGTH / K0_LENGTH)),
            VGroup(MathTex(r"R_t = 1 - \cos\theta =", font_size=MATH, color=palette.ROTATION_TEXT),
                   value(lambda: 1 - np.cos(KT_ANGLE - K0_ANGLE))),
            VGroup(MathTex(r"A_t = \cos\phi =", font_size=MATH, color=palette.ALIGNMENT_TEXT),
                   value(lambda: np.cos(LABEL_ANGLE - CENTRED_ANGLE))))
        for item in readouts:
            item.arrange(RIGHT, buff=0.2)
            item[1].add_updater(lambda m, item=item: m.next_to(item[0], RIGHT, buff=0.2))
        readouts[0].move_to([-3.6, -2.55, 0])
        readouts[1].move_to([-3.6, -3.45, 0])
        readouts[2].move_to([3.4, -2.55, 0])
        tag = layout.tag("schematic")

        self.play(FadeIn(left_title), GrowArrow(k0), FadeIn(k0_name), FadeIn(tag), run_time=1.5)
        self.play(GrowArrow(kt), FadeIn(kt_name), run_time=1.5)
        self.add(kt, kt_name)
        self.play(FadeIn(readouts[0]), run_time=1)
        self.wait(1.2)
        self.play(FadeIn(theta), FadeIn(theta_name), FadeIn(readouts[1]), run_time=1.2)
        self.wait(1.5)
        self.play(FadeIn(right_title), GrowArrow(label), FadeIn(label_name), FadeIn(ct), FadeIn(ct_name), run_time=1.5)
        self.add(ct, ct_name)
        self.play(FadeIn(phi), FadeIn(phi_name), FadeIn(readouts[2]), run_time=1.2)
        self.wait(1.4)

        # Stretching the kernel changes its length and neither angle.
        self.play(stretch.animate.set_value(STRETCH), run_time=2.5)
        self.wait(5.4)
