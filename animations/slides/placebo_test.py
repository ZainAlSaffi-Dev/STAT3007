"""At a fixed width, rotation is the statistic most tied to grokking: the placebo test.

Slide 19 of the deck. Every dot is a measurement. It is one of the 78 runs of
width 100 that reach test accuracy 0.95, read from
animations/data/n100_checkpoints.npz, with its grokking step from
report_runs.csv. The method is that of report_v2/code/E03_placebo.py. Each run
is read at its own grokking step, and then at the grokking step of another
run, assigned by a shuffle that keeps no run in place. A run is read at the
last checkpoint at or before a step. The scene draws the same 2,000 shuffles as
E03, from numpy's default_rng(0), and asserts that its spreads and tightening
ratios equal those in report_v2/code/out/numbers/E03.json. The ratios on
screen are the strings that the slide prints.

Each panel shows one statistic on its own vertical scale. The dark band spans
the mean plus or minus one SD at the runs' own grokking steps. The light band
does the same after a shuffle. The dots after the shuffle are the one shuffle
of the 2,000 whose SD is closest to the median SD, so the picture matches the
ratio.

Render from animations/:

    uv run manim -qh slides/placebo_test.py PlaceboTest
    uv run manim -qh -s slides/placebo_test.py PlaceboTest
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Dot, FadeIn, MathTex, Rectangle, Scene, Square, Text, Transform, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, palette

SIZE = palette.LABEL_COLUMN
MATH = SIZE + 12
N_SHUFFLES = 2_000
SEED = 0
# The statistic key of E03.json, the checkpoint column, the colours, the symbol and the ratio's variable.
STATS = (("R", "R_sum", palette.ROTATION, palette.ROTATION_TEXT, "R_t", "EthreeNhundredRRatio"),
         ("S", "S_sum", palette.SCALE, palette.SCALE_TEXT, "S_t", "EthreeNhundredSRatio"),
         ("A", "A_sum", palette.ALIGNMENT, palette.ALIGNMENT_TEXT, "A_t", "EthreeNhundredARatio"))
CENTRES = (-4.6, 0.0, 4.6)
WIDTH, TOP, BOTTOM = 3.4, 2.5, -2.1


def derangements(n, k, rng):
    """k permutations of n runs with no run in place, drawn in the order that E03_placebo.py draws them."""
    out = np.empty((k, n), dtype=int)
    i = 0
    while i < k:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            out[i] = p
            i += 1
    return out


class Panel:
    """One statistic: its values at the runs' own grokking steps and after the shown shuffle."""

    def __init__(self, true, shown, colour, centre, jitter):
        lo, hi = min(true.min(), shown.min()), max(true.max(), shown.max())
        lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
        self.y = lambda v: BOTTOM + (v - lo) / (hi - lo) * (TOP - BOTTOM)
        self.centre, self.colour = centre, colour
        self.true_dots = self.dots(true, jitter)
        self.shown_dots = self.dots(shown, jitter)
        self.true_band = self.band(true, 0.35)
        self.shown_band = self.band(shown, 0.12)

    def dots(self, values, jitter):
        return VGroup(*[Dot([self.centre + x, self.y(v), 0], radius=0.06, color=self.colour, fill_opacity=0.85)
                        for x, v in zip(jitter, values)])

    def band(self, values, opacity):
        mean, sd = values.mean(), values.std(ddof=1)
        low, high = self.y(mean - sd), self.y(mean + sd)
        return Rectangle(width=WIDTH, height=high - low, stroke_width=0, fill_color=self.colour,
                         fill_opacity=opacity).move_to([self.centre, (low + high) / 2, 0])


class PlaceboTest(Scene):
    def construct(self):
        # Measured: the 120 runs at width 100, of which the 78 that grok at level 0.95.
        n100 = data.load_n100()
        runs = data.load_report_runs()
        index = {name: i for i, name in enumerate(runs["run_id"])}
        rows = np.array([index[name] for name in n100["run_id"]])
        groks = ~runs["censored_grok95"][rows].astype(bool)
        T = runs["t_grok95"][rows][groks]
        n = len(T)
        assert n == int(data.deck_number("EthreeNhundredRuns"))

        grid = n100["step"]
        own = np.searchsorted(grid, T, side="right") - 1
        perms = derangements(n, N_SHUFFLES, np.random.default_rng(SEED))
        other = np.searchsorted(grid, T[perms], side="right") - 1
        e03 = data.report_numbers("E03")["groups"]["N100"]
        jitter = np.random.default_rng(1).uniform(-0.5, 0.5, n) * (WIDTH - 0.4)

        panels, names = [], VGroup()
        for (key, column, colour, text_colour, symbol, ratio_name), centre in zip(STATS, CENTRES):
            W = n100[column][groks]
            true = W[np.arange(n), own]
            placebo = W[np.arange(n)[None, :], other]
            sd_true, sd_placebo = true.std(ddof=1), placebo.std(ddof=1, axis=1)
            assert np.isclose(sd_true, e03[key]["sd"]["at_grok"], rtol=1e-12)
            assert np.isclose(np.median(sd_placebo) / sd_true, e03[key]["sd"]["tightening"], rtol=1e-12)
            shown = placebo[np.argmin(np.abs(sd_placebo - np.median(sd_placebo)))]
            panels.append(Panel(true, shown, colour, centre, jitter))
            name = VGroup(MathTex(symbol, font_size=MATH, color=text_colour),
                          MathTex(r"\times\," + data.deck_tex(ratio_name),
                                  font_size=MATH, color=palette.INK))
            names.add(name.arrange(RIGHT, buff=0.35).move_to([centre, BOTTOM - 0.7, 0]))

        own_key = VGroup(Square(0.38, stroke_width=0, fill_color=palette.GRAY, fill_opacity=0.55),
                         Text("own step", font_size=SIZE, color=palette.INK)).arrange(RIGHT, buff=0.2)
        other_key = VGroup(Square(0.38, stroke_width=0, fill_color=palette.GRAY, fill_opacity=0.18),
                           Text("another run's step", font_size=SIZE, color=palette.INK)).arrange(RIGHT, buff=0.2)
        VGroup(own_key, other_key).arrange(RIGHT, buff=0.9).move_to(3.45 * UP)

        # Rotation first: at its own grokking step, then at another run's.
        r = panels[0]
        self.play(FadeIn(r.true_dots, lag_ratio=0.02), FadeIn(names[0][0]), run_time=2)
        self.play(FadeIn(r.true_band), FadeIn(own_key), run_time=1)
        self.wait(2.5)
        self.play(Transform(r.true_dots, r.shown_dots), FadeIn(r.shown_band), FadeIn(other_key), run_time=3)
        self.play(FadeIn(names[0][1]), run_time=0.8)
        self.wait(3)

        # Then scale and alignment, both at once.
        rest = panels[1:]
        self.play(*[FadeIn(p.true_dots, lag_ratio=0.02) for p in rest], *[FadeIn(p.true_band) for p in rest],
                  *[FadeIn(nm[0]) for nm in names[1:]], run_time=1.8)
        self.wait(1.5)
        self.play(*[Transform(p.true_dots, p.shown_dots) for p in rest], *[FadeIn(p.shown_band) for p in rest],
                  run_time=3)
        self.play(*[FadeIn(nm[1]) for nm in names[1:]], run_time=0.8)
        self.wait(5.6)
