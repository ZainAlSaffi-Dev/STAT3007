"""We trained 720 networks across laziness, width and weight decay: the grid of settings.

Slide 16 of the deck. The grid is read from report_runs.csv of the report
extract and is not typed: 3 values of alpha, 6 widths and 8 values of eta lambda
give 144 cells, and each cell has 5 seeds, so 720 runs. The scene asserts these
counts. Each panel is one value of alpha, with the width N down and the weight
decay eta lambda across, in the order of the slide's table. The slide's
column is narrow, so the labels are larger than in the other clips.

Render from animations/:

    uv run manim -qh slides/grid_of_runs.py GridOfRuns
    uv run manim -qh -s slides/grid_of_runs.py GridOfRuns
"""

import sys
from pathlib import Path

from manim import DOWN, LEFT, RIGHT, UP, Arrow, FadeIn, LaggedStart, MathTex, Square, Text, VGroup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, layout, palette

SIZE = palette.LABEL_COLUMN + 16
MATH = SIZE + 8
CELL = 0.42
GAP = 0.6
FILL = "#e4e3df"


class GridOfRuns(layout.QuickScene):
    def construct(self):
        runs = data.load_report_runs()
        alphas = sorted(set(runs["alpha"]))
        widths = sorted(set(runs["width"]))
        decays = sorted(set(runs["eta_kappa"]))
        cells = set(zip(runs["alpha"], runs["width"], runs["eta_kappa"]))
        seeds = sorted(set(runs["seed"]))
        assert (len(alphas), len(widths), len(decays), len(seeds)) == (3, 6, 8, 5)
        assert len(cells) == 144 and len(runs["run_id"]) == 720

        panels, names = VGroup(), VGroup()
        width = len(decays) * CELL
        for k, alpha in enumerate(alphas):
            panel = VGroup(*[Square(CELL, stroke_color=palette.INK, stroke_width=2, fill_color=FILL, fill_opacity=1)
                             .move_to([j * CELL, -i * CELL, 0])
                             for i in range(len(widths)) for j in range(len(decays))])
            panel.move_to([(k - 1) * (width + GAP), 1.25, 0])
            panels.add(panel)
            names.add(MathTex(rf"\alpha = {alpha:g}", font_size=MATH, color=palette.INK).next_to(panel, UP, buff=0.3))

        down = Arrow(panels[0].get_corner(UP + LEFT) + 0.35 * LEFT, panels[0].get_corner(DOWN + LEFT) + 0.35 * LEFT,
                     buff=0, color=palette.INK, stroke_width=6)
        across = Arrow(panels[0].get_corner(DOWN + LEFT) + 0.35 * DOWN, panels[2].get_corner(DOWN + RIGHT) + 0.35 * DOWN,
                       buff=0, color=palette.INK, stroke_width=6)
        n_name = MathTex("N", font_size=MATH, color=palette.INK).next_to(down, LEFT, buff=0.15)
        decay_name = MathTex(r"\eta\lambda", font_size=MATH, color=palette.INK).next_to(across, DOWN, buff=0.15)

        self.play(LaggedStart(*[FadeIn(p) for p in panels], lag_ratio=0.3), FadeIn(names), run_time=2.5)
        self.play(FadeIn(down), FadeIn(n_name), FadeIn(across), FadeIn(decay_name), run_time=1.2)
        self.wait(1.8)

        count = VGroup(VGroup(MathTex(rf"{len(alphas)} \times {len(widths)} \times {len(decays)} = {len(cells)}", font_size=MATH,
                                      color=palette.INK),
                              Text("cells", font_size=SIZE, color=palette.INK)).arrange(RIGHT, buff=0.3),
                       Text(f"{len(seeds)} seeds each, so {len(runs['run_id'])} runs", font_size=SIZE, color=palette.INK))
        count.arrange(DOWN, buff=0.3).next_to(decay_name, DOWN, buff=0.45)
        self.play(FadeIn(count[0]), run_time=0.8)
        self.wait(1.2)
        self.play(FadeIn(count[1]), run_time=0.8)
        self.wait(6.6)
