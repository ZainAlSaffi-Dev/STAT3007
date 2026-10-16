"""The task: addition on a clock with 23 hours, and the 53 pairs a run must fill in.

Slide 3 of the deck. Row a and column b of the table hold (a + b) mod 23,
shaded from light at 0 to dark at 22, so equal sums form diagonal stripes that
wrap around. The shades stay neutral, because the deck keeps blue, orange and
aqua for the three statistics. The split is a measurement: the 476 training
pairs and the 53 test pairs of data seed 42, the split of every seed 0 run,
read from animations/data/split_seed42.npz.

Render from animations/:

    uv run manim -qh slides/modular_addition_table.py ModularAdditionTable
    uv run manim -qh -s slides/modular_addition_table.py ModularAdditionTable
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Create, FadeIn, FadeOut, LaggedStart, MathTex, Scene, Square,
                   ManimColor, SurroundingRectangle, Text, VGroup, interpolate_color)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, palette

P = 23
SIZE = palette.LABEL_COLUMN
MATH = SIZE + 8
CELL = 0.245
LIGHT, FADED = "#d9d8d4", "#f3f2ef"
EXAMPLE = (3, 21)


class ModularAdditionTable(Scene):
    def construct(self):
        split = data.load_split()
        a, b, is_test = split["a"], split["b"], split["is_test"]
        assert is_test.sum() == 53 and (~is_test).sum() == 476

        origin = np.array([-6.0, 2.75, 0])
        cells = VGroup()
        for i, j in zip(a, b):
            shade = interpolate_color(ManimColor(LIGHT), ManimColor(palette.INK), ((i + j) % P) / (P - 1))
            cells.add(Square(CELL, stroke_width=0, fill_color=shade, fill_opacity=1)
                      .move_to(origin + j * CELL * RIGHT + i * CELL * DOWN))
        rows = MathTex("a", font_size=MATH, color=palette.INK).next_to(cells, LEFT, buff=0.25)
        columns = MathTex("b", font_size=MATH, color=palette.INK).next_to(cells, UP, buff=0.2)

        # Fill the table along the stripes of equal sum.
        order = np.argsort((a + b) % P, kind="stable")
        self.play(LaggedStart(*[FadeIn(cells[k]) for k in order], lag_ratio=0.004), FadeIn(rows), FadeIn(columns),
                  run_time=3)

        # One worked cell.
        k = EXAMPLE[0] * P + EXAMPLE[1]
        box = SurroundingRectangle(cells[k], color=palette.INK, buff=0.03, stroke_width=6)
        example = MathTex(rf"({EXAMPLE[0]} + {EXAMPLE[1]}) \bmod 23 = {sum(EXAMPLE) % P}", font_size=MATH,
                          color=palette.INK).next_to(cells, RIGHT, buff=0.5).align_to(cells, UP)
        self.play(Create(box), FadeIn(example), run_time=1)
        self.wait(1.5)
        self.play(FadeOut(box), FadeOut(example), run_time=0.6)

        # The split of the seed 0 runs: the training pairs fade and the test pairs keep their shade.
        test_cells = VGroup(*[cells[k] for k in np.flatnonzero(is_test)])
        train_cells = VGroup(*[cells[k] for k in np.flatnonzero(~is_test)])
        train_key = VGroup(Square(0.4, stroke_width=0, fill_color=FADED, fill_opacity=1).set_stroke(palette.GRID, 2),
                           Text("476 training pairs", font_size=SIZE, color=palette.INK)).arrange(RIGHT, buff=0.25)
        test_key = VGroup(Square(0.4, stroke_color=palette.INK, stroke_width=4, fill_color=LIGHT, fill_opacity=1),
                          Text("53 test pairs", font_size=SIZE, color=palette.INK)).arrange(RIGHT, buff=0.25)
        keys = VGroup(train_key, test_key).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        keys.next_to(cells, RIGHT, buff=0.6).align_to(cells, UP).shift(0.3 * DOWN)
        self.play(train_cells.animate.set_fill(FADED), test_cells.animate.set_stroke(palette.INK, width=3),
                  FadeIn(keys), run_time=2)
        self.wait(1.5)

        question = VGroup(Text("Can the network", font_size=SIZE, color=palette.INK),
                          Text("fill these in?", font_size=SIZE, color=palette.INK))
        question.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(keys, DOWN, buff=1.0, aligned_edge=LEFT)
        self.play(FadeIn(question), run_time=1)
        self.wait(4.5)
