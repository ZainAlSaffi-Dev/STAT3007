"""What the two clocks would look like: the predicted slopes of Tier 2 beside the fitted ones.

Slide 23 of the deck. The fit is that of report_v2/code/E06_aft.py, row "main"
of E06.json, at alpha 1 over N from 50 to 1600 and eta lambda from 1e-5 to 1e-4:

    log T = c + a x + b y + gamma x y,   x = log N - x0,   y = log(eta lambda) - y0,

with natural logs centred at the middle of that grid. The clip draws two
slices through the centre of the fit. On the left, log T against log N at
y = 0: the fitted line has slope a. On the right, log T against log(eta lambda)
at x = 0: the fitted line has slope b. The gamma term vanishes on both slices,
so the ink lines are exactly the fitted slopes. Every number comes from E06.json,
and the labels print the strings of the slide.

The orange and grey lines are predictions, drawn through the same centre so
that only their slopes differ. The weight-decay clock has b = -1, since its
grokking time scales as 1 / (eta lambda) (Kim, 2026, as slide 4 cites), and
a = 0, which is our reading. The rotation clock has a = 1 and b = 0, our
reading of slide 14: if T_t stays of order 1 / N, the time to rotate grows in
proportion to N. The note on screen says that both are predictions.

A first version drew the three planes in 3D. Manim's Cairo renderer showed the
planes nearly edge-on and could not keep the labels readable, so the clip shows
the two slices instead.

Render from animations/:

    uv run manim -qh slides/two_clocks.py TwoClocks
    uv run manim -qh -s slides/two_clocks.py TwoClocks
"""

import sys
from pathlib import Path

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, FadeIn, Line, MathTex, Rectangle, Text, VGroup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, layout, palette

SIZE = palette.LABEL_WIDE + 4
MATH = SIZE + 8
WIDTHS = (50, 1600)
DECAYS = (1e-5, 1e-4)
Y_RANGE = (8.4, 12.2)
PANEL_WIDTH, PANEL_HEIGHT = 3.8, 4.4


class TwoClocks(layout.QuickScene):
    def construct(self):
        # Measured: the fit of E06.json, row "main", and its centre.
        e06 = data.report_numbers("E06")
        x0, y0 = e06["centre"]["x0_logN"], e06["centre"]["y0_log_decay"]
        est = e06["rows"]["main"]["estimate"]
        c, a, b = est["c"], est["a"], est["b"]
        assert data.deck_tex("EsixMainA") == f"{a:+.2f}" and data.deck_tex("EsixMainB") == f"{b:+.2f}"

        x_lo, x_hi = np.log(WIDTHS[0]) - x0, np.log(WIDTHS[1]) - x0
        y_lo, y_hi = np.log(DECAYS[0]) - y0, np.log(DECAYS[1]) - y0

        def panel(origin, lo, hi):
            """A map from (slice coordinate, log T) to the frame, the axes, and a dashed line at the centre."""
            origin = np.asarray(origin, float)

            def point(u, v):
                return origin + np.array([(u - lo) / (hi - lo) * PANEL_WIDTH,
                                          (v - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * PANEL_HEIGHT, 0])

            frame = VGroup(Line(point(lo, Y_RANGE[0]), point(hi, Y_RANGE[0])),
                           Line(point(lo, Y_RANGE[0]), point(lo, Y_RANGE[1]))).set_stroke(palette.INK, 3)
            centre = DashedLine(point(0, Y_RANGE[0]), point(0, Y_RANGE[1]), color=palette.GRAY, stroke_width=2,
                                dash_length=0.1)
            return point, VGroup(frame, centre)

        left, left_frame = panel([-5.9, -2.0, 0], x_lo, x_hi)
        right, right_frame = panel([0.75, -2.0, 0], y_lo, y_hi)

        # The tick labels are the ends of the grid and two grokking steps, placed at their logarithms.
        ticks = VGroup(
            Text("50", font_size=SIZE, color=palette.INK).next_to(left(x_lo, Y_RANGE[0]), DOWN, buff=0.2),
            Text("1600", font_size=SIZE, color=palette.INK).next_to(left(x_hi, Y_RANGE[0]), DOWN, buff=0.2),
            MathTex(r"10^{-5}", font_size=MATH, color=palette.INK).next_to(right(y_lo, Y_RANGE[0]), DOWN, buff=0.2),
            MathTex(r"10^{-4}", font_size=MATH, color=palette.INK).next_to(right(y_hi, Y_RANGE[0]), DOWN, buff=0.2))
        for power in (4, 5):
            ticks.add(MathTex(rf"10^{power}", font_size=MATH, color=palette.INK)
                      .next_to(left(x_lo, np.log(10.0 ** power)), LEFT, buff=0.2))
        x_names = VGroup(MathTex("N", font_size=MATH, color=palette.INK).next_to(left(0, Y_RANGE[0]), DOWN, buff=0.3),
                         MathTex(r"\eta\lambda", font_size=MATH, color=palette.INK)
                         .next_to(right(0, Y_RANGE[0]), DOWN, buff=0.3))
        y_name = MathTex("T", font_size=MATH, color=palette.INK).next_to(left(x_lo, Y_RANGE[1]), UP, buff=0.2)

        def line(point, lo, hi, slope, colour):
            return Line(point(lo, c + slope * lo), point(hi, c + slope * hi), color=colour, stroke_width=7)

        def end_label(point, hi, slope, colour, tex, nudge=0.0):
            return MathTex(tex, font_size=MATH, color=colour).next_to(point(hi, c + slope * hi), RIGHT, buff=0.15) \
                .shift(nudge * UP)

        # Left panel: slopes in log N at the middle weight decay. Right panel: slopes in log(eta lambda).
        rotation = VGroup(line(left, x_lo, x_hi, 1.0, palette.ROTATION), line(right, y_lo, y_hi, 0.0, palette.ROTATION),
                          end_label(left, x_hi, 1.0, palette.ROTATION_TEXT, "a = 1"),
                          end_label(right, y_hi, 0.0, palette.ROTATION_TEXT, "b = 0", 0.15))
        decay = VGroup(line(left, x_lo, x_hi, 0.0, palette.GRAY), line(right, y_lo, y_hi, -1.0, palette.GRAY),
                       end_label(left, x_hi, 0.0, palette.GRAY, "a = 0", -0.17),
                       end_label(right, y_hi, -1.0, palette.GRAY, "b = -1", -0.17))
        fitted = VGroup(line(left, x_lo, x_hi, a, palette.INK), line(right, y_lo, y_hi, b, palette.INK),
                        end_label(left, x_hi, a, palette.INK, f"a = {data.deck_tex('EsixMainA')}", 0.17),
                        end_label(right, y_hi, b, palette.INK, f"b = {data.deck_tex('EsixMainB')}", 0.17))

        def key(colour, words):
            return VGroup(Rectangle(width=0.5, height=0.28, stroke_width=0, fill_color=colour, fill_opacity=1),
                          Text(words, font_size=SIZE - 6, color=palette.INK)).arrange(RIGHT, buff=0.2)

        keys = VGroup(key(palette.ROTATION, "rotation clock"), key(palette.GRAY, "weight-decay clock"),
                      key(palette.INK, "our fit")).arrange(RIGHT, buff=0.45).to_edge(DOWN, buff=0.3)
        note = Text("Both clocks are predictions through the centre of the fit.", font_size=SIZE - 6,
                    color=palette.GRAY).to_edge(UP, buff=0.3)

        self.play(Create(left_frame), Create(right_frame), FadeIn(ticks), FadeIn(x_names), FadeIn(y_name),
                  run_time=1.5)
        self.play(FadeIn(rotation), FadeIn(keys[0]), run_time=1.5)
        self.wait(1.5)
        self.play(FadeIn(decay), FadeIn(keys[1]), FadeIn(note), run_time=1.5)
        self.wait(1.5)
        self.play(FadeIn(fitted), FadeIn(keys[2]), run_time=1.5)
        self.wait(11)
