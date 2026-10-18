"""A gradient step on one input moves the output on another, and the size of that move is the kernel.

Slide 5 of the deck. The first part is a schematic of two outputs of one
network, f(x) and f(x'). One gradient step on the loss at x moves f(x) towards
its target. To first order it moves f(x') by an amount proportional to
K(x', x), and f(x) itself by an amount proportional to K(x, x), as the formula
on the slide states (Jacot, Gabriel and Hongler, 2020). The bar heights are
not measured.

The second part is measured. It is the kernel of the report, the tangent kernel
of the summed logits over sqrt(p), on all 529 pairs at step 0 of the run
ntk_N100_a1_wd0_s0, rebuilt from its saved weights with kernels.sum_kernel. The
pairs are ordered by a and then by b. A block on the diagonal holds the pairs
with the same a, and a thin diagonal line holds the pairs with the same b. At
step 0 a pair's kernel with itself averages 0.042. Pairs that share one number
average 0.019 (the same a) and 0.020 (the same b). Pairs that share neither
average 0.0064, with SD 0.0018.

Render from animations/:

    uv run manim -qh slides/ntk_from_scratch.py NTKFromScratch
    uv run manim -qh -s slides/ntk_from_scratch.py NTKFromScratch
"""

import sys
from pathlib import Path

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Brace, Create, DashedLine, FadeIn, FadeOut, GrowFromEdge, Line,
                   MathTex, Rectangle, Square, Text, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, kernels, layout, palette

RUN = data.dataset_run_id(100, 1.0, 0.0)
P = 23
SIZE = palette.LABEL_COLUMN
MATH = SIZE + 8
# The kernel is drawn at full ink from 0.03, about 1.5 times the kernel of two
# pairs that share one number. The kernel of a pair with itself saturates.
VMAX = 0.03
# The zoom shows the pairs with a from 0 to 4, the first 5 * 23 rows.
ZOOM_ROWS = 5 * P


class NTKFromScratch(layout.QuickScene):
    def construct(self):
        self.one_step()
        self.every_pair()

    def one_step(self):
        tag = layout.tag("schematic")
        baseline, width, xs = -2.0, 1.6, (-1.8, 1.8)
        heights, moves = (1.5, 1.8), (1.9, 0.9)
        floor = Line([-3.2, baseline, 0], [3.2, baseline, 0], color=palette.INK, stroke_width=3)
        bars, names, grown = VGroup(), VGroup(), VGroup()
        for x, h, m, name in zip(xs, heights, moves, ("f(x)", "f(x')")):
            bars.add(Rectangle(width=width, height=h, stroke_width=0, fill_color=palette.INK, fill_opacity=1)
                     .move_to([x, baseline + h / 2, 0]))
            names.add(MathTex(name, font_size=MATH, color=palette.INK).next_to([x, baseline, 0], DOWN, buff=0.3))
            grown.add(Rectangle(width=width, height=m, stroke_width=0, fill_color=palette.TRAIN, fill_opacity=1)
                      .move_to([x, baseline + h + m / 2, 0]))

        # One step moves f(x) towards its target, not onto it.
        target_y = baseline + heights[0] + moves[0] + 0.5
        target = DashedLine([xs[0] - 1.1, target_y, 0], [xs[0] + 1.1, target_y, 0], color=palette.GRAY,
                            stroke_width=5)
        target_name = Text("target", font_size=SIZE, color=palette.GRAY).next_to(target, UP, buff=0.15)
        step_name = VGroup(Text("one step on", font_size=SIZE, color=palette.INK),
                           MathTex("x", font_size=MATH, color=palette.INK)).arrange(RIGHT, buff=0.2)
        step_name.move_to(3.2 * UP)

        self_brace = Brace(grown[0], direction=LEFT, color=palette.INK)
        self_name = MathTex(r"\propto K(x, x)", font_size=MATH, color=palette.INK).next_to(self_brace, LEFT, buff=0.15)
        cross_brace = Brace(grown[1], direction=RIGHT, color=palette.INK)
        cross_name = MathTex(r"\propto K(x', x)", font_size=MATH, color=palette.INK)
        cross_name.next_to(cross_brace, RIGHT, buff=0.15)

        self.play(FadeIn(floor), FadeIn(bars), FadeIn(names), FadeIn(tag), run_time=1.5)
        self.play(Create(target), FadeIn(target_name), run_time=1.2)
        self.wait(0.5)
        self.play(FadeIn(step_name), run_time=0.8)
        self.play(GrowFromEdge(grown[0], DOWN), GrowFromEdge(grown[1], DOWN), run_time=2.5)
        self.play(FadeIn(cross_brace), FadeIn(cross_name), run_time=1)
        self.wait(1.5)
        self.play(FadeIn(self_brace), FadeIn(self_name), run_time=1)
        self.wait(2)
        self.play(*[FadeOut(m) for m in (floor, bars, names, grown, target, target_name, step_name, self_brace,
                                         self_name, cross_brace, cross_name, tag)], run_time=1)

    def every_pair(self):
        # Measured: ntk_N100_a1_wd0_s0 at step 0, all 529 pairs in the order of kernels.one_hot_pairs.
        steps, W1, W2 = data.load_weights(RUN)
        assert steps[0] == 0
        X, _, _ = kernels.one_hot_pairs(P)
        K = kernels.sum_kernel(W1[0], W2[0], X)

        side = 4.2
        full = layout.grey_map(K, VMAX, side).move_to([-3.9, 0.2, 0])
        rows = MathTex("x", font_size=MATH, color=palette.INK).next_to(full, LEFT, buff=0.25)
        columns = MathTex("x'", font_size=MATH, color=palette.INK).next_to(full, UP, buff=0.2)
        full_name = Text("all 529 pairs", font_size=SIZE, color=palette.INK).next_to(full, DOWN, buff=0.3)
        tag = layout.tag("measured: one network at step 0")

        self.play(FadeIn(full), FadeIn(rows), FadeIn(columns), FadeIn(tag), run_time=1.5)
        self.play(FadeIn(full_name), run_time=0.8)
        self.wait(3)

        cell = side / len(K)
        outline = Square(side_length=ZOOM_ROWS * cell, color=palette.INK, stroke_width=5)
        outline.move_to(full.get_corner(UP + LEFT), aligned_edge=UP + LEFT)
        zoom = layout.grey_map(K[:ZOOM_ROWS, :ZOOM_ROWS], VMAX, side).move_to([1.8, 0.2, 0])
        zoom_frame = Square(side_length=side, color=palette.INK, stroke_width=5).move_to(zoom)
        links = VGroup(Line(outline.get_corner(UP + RIGHT), zoom_frame.get_corner(UP + LEFT)),
                       Line(outline.get_corner(DOWN + RIGHT), zoom_frame.get_corner(DOWN + LEFT)))
        links.set_stroke(palette.INK, width=3, opacity=0.6)
        zoom_name = VGroup(Text("pairs with", font_size=SIZE, color=palette.INK),
                           MathTex(r"a \le 4", font_size=MATH, color=palette.INK)).arrange(RIGHT, buff=0.2)
        zoom_name.next_to(zoom, DOWN, buff=0.3)

        self.play(Create(outline), run_time=1)
        self.play(Create(links), FadeIn(zoom), Create(zoom_frame), run_time=1.5)
        self.play(FadeIn(zoom_name), run_time=0.8)
        self.wait(1)

        # A block on the diagonal holds the pairs with a = a'. The block of a = a' = 4 is the last one.
        block = side / 5
        same_a_point = zoom.get_corner(DOWN + RIGHT) + 0.5 * block * (UP + LEFT)
        # In the block of rows a = 0 and columns a' = 4, the pairs with b = b' lie on its diagonal.
        same_b_point = zoom.get_corner(UP + RIGHT) + 0.5 * block * (DOWN + LEFT)
        marks = VGroup()
        for point, text in ((same_b_point, "b = b'"), (same_a_point, "a = a'")):
            arrow = Arrow(point + 1.25 * RIGHT, point, buff=0.05, color=palette.INK, stroke_width=6,
                          max_tip_length_to_length_ratio=0.3)
            name = MathTex(text, font_size=MATH, color=palette.INK).next_to(arrow, RIGHT, buff=0.1)
            marks.add(VGroup(arrow, name))
        self.play(FadeIn(marks[0]), run_time=0.8)
        self.wait(1.2)
        self.play(FadeIn(marks[1]), run_time=0.8)
        self.wait(4.5)
