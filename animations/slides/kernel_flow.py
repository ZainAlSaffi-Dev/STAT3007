"""In theory, weight decay only rescales the kernel, and only T_t can turn it.

Slide 14 of the deck. This is a schematic, and it says so on screen. It draws
the two terms of the kernel flow of Lewkowycz and Gur-Ari (2020), Equation S5
of arXiv:2006.08643v2, checked against the paper on 16 October 2026:

    dTheta_t(x, x')/dt = -2(k - 1) lambda Theta_t(x, x') + T_t(x, x') + T_t(x', x).

The first term is proportional to the kernel itself, so on its own it shrinks
the kernel evenly and keeps its axes. Only T_t, Equation S6, can turn it. The
paper states that E[T_0] is of order 1 / n in the width at initialisation, a
result it takes from Dyer and Gur-Ari (2020). The last beat draws the turn
from T_t shrinking as 1 / N, from 70 degrees at N = 100 to 4.4 degrees at
N = 1600. That T_t stays of order 1 / N during training is our reading, as
the slide says, and the clip labels it so.

Render from animations/:

    uv run manim -qh slides/kernel_flow.py KernelFlow
    uv run manim -qh -s slides/kernel_flow.py KernelFlow
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arc, DashedVMobject, FadeIn, FadeOut, MathTex, Text,
                   ValueTracker, VGroup, always_redraw)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import layout, palette

SIZE = palette.LABEL_COLUMN + 8
MATH = SIZE + 8
CENTRE = np.array([-2.7, -0.3, 0])
UNIT = 1.0
SHRINK = 0.75
TURN = np.deg2rad(35)
RADIUS = 2.7
ARC_START = np.deg2rad(-35)
ARC_AT_100 = np.deg2rad(70)


def turned(K, angle):
    """K turned by the angle: Q K Q^T for the rotation Q."""
    c, s = np.cos(angle), np.sin(angle)
    Q = np.array([[c, -s], [s, c]])
    return Q @ K @ Q.T


def turn_arc(span, colour, dashed=False):
    """The arrow of the T_t term: an arc around the kernel that turns it by the span."""
    arc = Arc(radius=RADIUS, start_angle=ARC_START, angle=span, arc_center=CENTRE, color=colour, stroke_width=8)
    arc.add_tip(tip_length=0.3, tip_width=0.3)
    return DashedVMobject(arc, num_dashes=24) if dashed else arc


class KernelFlow(layout.QuickScene):
    def construct(self):
        K0 = turned(np.diag([2.0, 1.0]), np.deg2rad(15))
        scale = ValueTracker(1.0)
        angle = ValueTracker(0.0)
        kernel = always_redraw(lambda: layout.kernel_ellipse(scale.get_value() * turned(K0, angle.get_value()), UNIT,
                                                             palette.INK, CENTRE))
        ghost = DashedVMobject(layout.kernel_ellipse(K0, UNIT, palette.GRAY, CENTRE, width=4)[0], num_dashes=48)
        name = MathTex(r"\Theta_t", font_size=MATH, color=palette.INK).move_to(CENTRE + 2.6 * UP + 1.6 * LEFT)
        tag = layout.tag("schematic")

        self.play(FadeIn(kernel), FadeIn(name), FadeIn(tag), run_time=1.5)
        self.add(ghost)

        # The decay term shrinks the kernel evenly and keeps its axes.
        decay = VGroup(MathTex(r"-2(k-1)\lambda\,\Theta_t", font_size=MATH, color=palette.INK),
                       Text("shrinks it evenly", font_size=SIZE, color=palette.GRAY))
        decay.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.7, 2.4, 0])
        self.play(FadeIn(decay), run_time=0.8)
        self.play(scale.animate.set_value(SHRINK), run_time=2.5)
        self.wait(0.7)

        # T_t turns it.
        arc = turn_arc(ARC_AT_100, palette.INK)
        twist = VGroup(MathTex(r"T_t", font_size=MATH, color=palette.INK),
                       Text("turns it", font_size=SIZE, color=palette.GRAY))
        twist.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.6, -0.1, 0])
        self.play(FadeIn(arc), FadeIn(twist), run_time=0.8)
        self.play(angle.animate.set_value(TURN), run_time=2.5)
        self.wait(0.7)

        # Our reading: T_t stays of order 1 / N, so a wider network turns its kernel more slowly.
        width = ValueTracker(100.0)

        def tip(span):
            return CENTRE + RADIUS * np.array([np.cos(ARC_START + span), np.sin(ARC_START + span), 0])

        ghost_arc = turn_arc(ARC_AT_100, palette.GRAY, dashed=True)
        ghost_name = MathTex(r"N = 100", font_size=MATH, color=palette.GRAY).next_to(tip(ARC_AT_100), RIGHT, buff=0.3)
        live_arc = always_redraw(lambda: turn_arc(ARC_AT_100 * 100 / width.get_value(), palette.INK))
        live_name = always_redraw(lambda: MathTex(rf"N = {int(round(width.get_value()))}", font_size=MATH,
                                                  color=palette.INK)
                                  .next_to(tip(ARC_AT_100 * 100 / width.get_value()), RIGHT, buff=0.3))
        reading = Text("our reading", font_size=SIZE, color=palette.GRAY).move_to([3.7, -2.5, 0])
        self.play(FadeOut(twist[1]), run_time=0.4)
        self.remove(arc)
        self.add(ghost_arc, live_arc, live_name)
        self.play(FadeIn(ghost_name), FadeIn(reading), run_time=0.8)
        self.play(width.animate.set_value(1600.0), run_time=4,
                  rate_func=lambda t: (np.power(16, t) - 1) / 15)
        self.wait(5.3)
