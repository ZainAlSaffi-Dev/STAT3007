"""In theory, weight decay only rescales the kernel, and a wider network changes its shape less.

Slide 12 of the deck, shown as a still. This is a schematic, and it says so on
screen. It draws the two terms of the kernel flow of Lewkowycz and Gur-Ari
(2020), Equation S5 of arXiv:2006.08643v2, checked against the paper on
16 October 2026. The deck writes their Theta_t as K_t:

    dK_t(x, x')/dt = -2(k - 1) lambda K_t(x, x') + T_t(x, x') + T_t(x', x).

The last frame has two panels, one per term.

- Left, the scale part. The first term is proportional to the kernel itself,
  so on its own it shrinks the kernel evenly and keeps its axes. The solid
  ellipse is the start scaled by 0.75.
- Right, the shape part. Only T_t, Equation S6, can change the shape. The
  paper states that E[T_0] is of order 1 / n in the width at initialisation,
  a result it takes from Dyer and Gur-Ari (2020). The solid ellipse is the
  start turned by 35 degrees at N = 100. The light one is the turn scaled by
  100 / N at N = 1600, about 2.2 degrees, so it barely leaves the start. That
  T_t stays of order 1 / N during training is our prediction, and the clip
  labels it so.

Render from animations/:

    python -m manim -qh slides/kernel_flow.py KernelFlow
    python -m manim -qh -s slides/kernel_flow.py KernelFlow
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, DashedVMobject, FadeIn, Line, MathTex, Text, ValueTracker, VGroup,
                   always_redraw)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import layout, notation, palette

SIZE = palette.LABEL_COLUMN
MATH = SIZE + 8
KEY = SIZE - 8
UNIT = 0.95
SHRINK = 0.75
TURN = np.deg2rad(35)
WIDE = 1600
LEFT_CENTRE = np.array([-3.7, 0.25, 0])
RIGHT_CENTRE = np.array([3.5, 0.25, 0])
NOTE = SIZE - 16
FAINT = 0.45


def turned(K, angle):
    """K turned by the angle: Q K Q^T for the rotation Q."""
    c, s = np.cos(angle), np.sin(angle)
    Q = np.array([[c, -s], [s, c]])
    return Q @ K @ Q.T


def heading(math, words, colour, centre):
    """The term of the flow and the name of its part, above a panel."""
    group = VGroup(MathTex(math, font_size=MATH, color=palette.INK), Text(words, font_size=SIZE, color=colour))
    return group.arrange(DOWN, buff=0.15).move_to(centre + 2.85 * UP)


def key(colour, opacity, math):
    """One entry of the key below the shape panel: a short line in the colour of an ellipse and its width."""
    line = Line(ORIGIN, 0.5 * RIGHT, color=colour, stroke_width=8).set_stroke(opacity=opacity)
    return VGroup(line, MathTex(math, font_size=KEY, color=palette.INK)).arrange(RIGHT, buff=0.15)


class KernelFlow(layout.QuickScene):
    def construct(self):
        K0 = turned(np.diag([2.0, 1.0]), np.deg2rad(15))

        def start(centre):
            return DashedVMobject(layout.kernel_ellipse(K0, UNIT, palette.GRAY, centre, width=4)[0], num_dashes=48)

        scale = ValueTracker(1.0)
        angle = ValueTracker(0.0)
        shrunk = always_redraw(lambda: layout.kernel_ellipse(scale.get_value() * K0, UNIT, palette.SCALE, LEFT_CENTRE))
        turned_100 = always_redraw(lambda: layout.kernel_ellipse(turned(K0, angle.get_value()), UNIT, palette.ROTATION,
                                                                 RIGHT_CENTRE))
        turned_wide = layout.kernel_ellipse(turned(K0, TURN * 100 / WIDE), UNIT, palette.ROTATION, RIGHT_CENTRE)
        turned_wide.set_stroke(opacity=FAINT)

        scale_head = heading(notation.FLOW_SCALE, "scale part", palette.SCALE_TEXT, LEFT_CENTRE)
        shape_head = heading(notation.FLOW_SHAPE, "shape part", palette.ROTATION_TEXT, RIGHT_CENTRE)
        self.play(FadeIn(start(LEFT_CENTRE)), FadeIn(start(RIGHT_CENTRE)), FadeIn(scale_head), FadeIn(shape_head),
                  FadeIn(Text("schematic", font_size=palette.TAG, color=palette.GRAY).to_corner(DOWN + LEFT, buff=0.3)),
                  run_time=1.2)

        # The scale part shrinks the kernel evenly and keeps its axes.
        evenly = Text("shrinks it evenly", font_size=KEY, color=palette.GRAY).move_to(LEFT_CENTRE + 2.45 * DOWN)
        self.add(shrunk)
        self.play(scale.animate.set_value(SHRINK), FadeIn(evenly), run_time=2)
        self.wait(0.5)

        # The shape part turns it: by 35 degrees at N = 100.
        keys = VGroup(key(palette.ROTATION, 1.0, r"N = 100"), key(palette.ROTATION, FAINT, r"N = 1600"))
        keys.arrange(RIGHT, buff=0.5).move_to(RIGHT_CENTRE + 2.45 * DOWN)
        self.add(turned_100)
        self.play(angle.animate.set_value(TURN), FadeIn(keys[0]), run_time=2)
        self.wait(0.5)

        # Our prediction: with T_t of order 1 / N, the turn at N = 1600 is 100 / 1600 of it.
        less = Text("a wider network turns less", font_size=NOTE, color=palette.GRAY)
        less.move_to(RIGHT_CENTRE + 3.1 * DOWN)
        note = Text("our prediction", font_size=palette.TAG, color=palette.GRAY).next_to(less, DOWN, buff=0.12)
        self.play(FadeIn(turned_wide), FadeIn(keys[1]), FadeIn(less), FadeIn(note), run_time=1.2)
        self.wait(5)
