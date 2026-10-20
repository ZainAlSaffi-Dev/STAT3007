"""A kernel can grow, or it can change shape, and both can travel the same distance.

Slide 8 of the deck. This is a schematic, and it says so on screen. Each
ellipse is an exact 2 by 2 kernel, drawn by layout.kernel_ellipse: its axes are
the eigenvectors and their lengths are the eigenvalues. The start is K_0 with
eigenvalues 2 and 1. On the left the kernel grows, K_t = s K_0 with s from 1 to
1.5. On the right it turns at a fixed size, K_t = Q K_0 Q^T for a rotation Q by
the angle phi. For this K_0 the turn gives S_t = 1 and R_t = 0.2 sin^2 phi, so
D_t^2 = 2 R_t by report_v2 Equation eq:identity. The turn stops at the angle
where D_t = 0.5, the distance of the growth path, which is phi = 52.2 degrees.
The counters compute D_t from the drawn matrices with kernels.statistics.

Render from animations/:

    uv run manim -qh slides/scale_versus_rotation.py ScaleVersusRotation
    uv run manim -qh -s slides/scale_versus_rotation.py ScaleVersusRotation
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Create, DashedVMobject, DecimalNumber, FadeIn, FadeOut, Indicate, MathTex,
                   Text, ValueTracker, VGroup, always_redraw)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import kernels, layout, palette

SIZE = palette.LABEL_WIDE
MATH = SIZE + 8
UNIT = 0.75
EIGENVALUES = (2.0, 1.0)
D_END = 0.5
LEFT_CENTRE = np.array([-3.6, 0.3, 0])
RIGHT_CENTRE = np.array([3.6, 0.3, 0])


def turned(K, angle):
    """K turned by the angle: Q K Q^T for the rotation Q."""
    c, s = np.cos(angle), np.sin(angle)
    Q = np.array([[c, -s], [s, c]])
    return Q @ K @ Q.T


class ScaleVersusRotation(layout.QuickScene):
    def construct(self):
        K0 = turned(np.diag(EIGENVALUES), np.deg2rad(20))

        # The growth stops at D_t = 0.5. The turn stops at the angle with the same D_t,
        # from R_t = share sin^2 phi and D_t^2 = 2 R_t.
        l1, l2 = EIGENVALUES
        share = (l1 - l2) ** 2 / (l1 ** 2 + l2 ** 2)
        s_end, phi_end = 1 + D_END, np.arcsin(D_END / np.sqrt(2 * share))
        assert abs(kernels.statistics(s_end * K0, K0)["D_t"] - D_END) < 1e-12
        assert abs(kernels.statistics(turned(K0, phi_end), K0)["D_t"] - D_END) < 1e-12

        start = layout.kernel_ellipse(K0, UNIT, palette.INK, centre=[0, 0.3, 0])
        start_name = MathTex("K_0", font_size=MATH, color=palette.INK).next_to(start, UP, buff=0.3)
        lines = VGroup(Text("The axes are the eigenvectors.", font_size=SIZE, color=palette.INK),
                       Text("Their lengths are the eigenvalues.", font_size=SIZE, color=palette.INK))
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(start, DOWN, buff=0.7)
        tag = layout.tag("schematic")
        self.play(Create(start), FadeIn(start_name), FadeIn(tag), run_time=1.5)
        self.play(FadeIn(lines), run_time=1)
        self.wait(3)

        # The start splits into two copies, one per path.
        copy = start.copy()
        self.play(copy.animate.move_to(LEFT_CENTRE).set_color(palette.SCALE),
                  start.animate.move_to(RIGHT_CENTRE).set_color(palette.ROTATION),
                  FadeOut(lines), FadeOut(start_name), run_time=1.5)

        scale = ValueTracker(1.0)
        angle = ValueTracker(0.0)
        grow = always_redraw(lambda: layout.kernel_ellipse(scale.get_value() * K0, UNIT, palette.SCALE, LEFT_CENTRE))
        turn = always_redraw(lambda: layout.kernel_ellipse(turned(K0, angle.get_value()), UNIT, palette.ROTATION,
                                                            RIGHT_CENTRE))
        self.remove(copy, start)
        self.add(grow, turn)

        ghosts = VGroup(*[DashedVMobject(layout.kernel_ellipse(K0, UNIT, palette.GRAY, centre, width=4)[0],
                                         num_dashes=48) for centre in (LEFT_CENTRE, RIGHT_CENTRE)])
        names = VGroup(Text("growth", font_size=SIZE, color=palette.SCALE_TEXT).move_to(LEFT_CENTRE + 2.6 * UP),
                       Text("change of shape", font_size=SIZE, color=palette.ROTATION_TEXT)
                       .move_to(RIGHT_CENTRE + 2.6 * UP))

        def counter(kernel_now, centre):
            label = MathTex("D_t =", font_size=MATH, color=palette.INK)
            value = DecimalNumber(0, num_decimal_places=2, font_size=MATH, color=palette.INK)
            VGroup(label, value).arrange(RIGHT, buff=0.2).move_to(centre + 2.7 * DOWN)
            value.add_updater(lambda m: m.set_value(kernels.statistics(kernel_now(), K0)["D_t"])
                              .next_to(label, RIGHT, buff=0.2))
            return VGroup(label, value)

        counts = VGroup(counter(lambda: scale.get_value() * K0, LEFT_CENTRE),
                        counter(lambda: turned(K0, angle.get_value()), RIGHT_CENTRE))

        self.play(FadeIn(ghosts), FadeIn(names), FadeIn(counts), run_time=1)
        self.play(scale.animate.set_value(s_end), run_time=4)
        self.wait(1.5)
        self.play(angle.animate.set_value(phi_end), run_time=4)
        self.wait(1.5)
        same = Text("the same distance", font_size=SIZE, color=palette.INK).move_to(3.35 * DOWN)
        self.play(FadeIn(same), Indicate(counts[0], color=palette.INK), Indicate(counts[1], color=palette.INK),
                  run_time=1.2)
        self.wait(4.8)
