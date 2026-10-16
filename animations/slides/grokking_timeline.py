"""The baseline run fits its training pairs long before it generalises.

Slide 2 of the deck. Both curves are measurements of the run ntk_N100_a1_wd0_s0
of the sweep of 3 October 2026: alpha 1, width 100, eta lambda 0 and seed 0,
read from its committed seed 0 curve. Its two events come from report_runs.csv
of the report extract. The run memorises at step 2,983, the first checkpoint
with training accuracy at least 0.99. Its grokking step T is 15,395, the first
checkpoint with test accuracy at least 0.95. The slide prints the medians over
the 5 seeds, 2,983 and 23,000, so the clip marks the two events without
printing their steps.

Render from animations/:

    uv run manim -qh slides/grokking_timeline.py GrokkingTimeline
    uv run manim -qh -s slides/grokking_timeline.py GrokkingTimeline
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, BraceBetweenPoints, Create, DashedLine, FadeIn, GrowFromCenter, MathTex,
                   Scene, Text, ValueTracker, VGroup, always_redraw, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, layout, notation, palette

RUN = data.dataset_run_id(100, 1.0, 0.0)
SIZE = palette.LABEL_COLUMN


class GrokkingTimeline(Scene):
    def construct(self):
        # Measured: every checkpoint of ntk_N100_a1_wd0_s0 after step 0, which has no place on a log axis.
        run = data.load_curves(RUN)
        steps, train, test = data.history(run, "step", "train_acc", "test_acc")
        x, train, test = np.log10(steps[1:]), train[1:], test[1:]

        # Measured: the events of the same run, report_runs.csv.
        runs = data.load_report_runs()
        row = list(runs["run_id"]).index(RUN)
        assert not runs["censored_grok95"][row], "the baseline run with seed 0 groks within the budget"
        x_mem, x_grok = np.log10(runs["t_mem"][row]), np.log10(runs["t_grok95"][row])

        axes, numbers = layout.step_axes(x_length=10.6, y_length=4.0)
        VGroup(axes, numbers).shift(0.35 * LEFT + (2.65 - axes.c2p(0, 1)[1]) * UP)
        x_name = Text("step", font_size=SIZE, color=palette.INK).next_to(axes.x_axis.get_end(), RIGHT, buff=0.3)
        y_name = Text("accuracy", font_size=SIZE, color=palette.INK)
        y_name.next_to(axes.c2p(0, 1), UP, buff=0.35).align_to(numbers, LEFT)

        cursor = ValueTracker(0)
        train_line = always_redraw(lambda: layout.curve_until(axes, x, train, cursor.get_value(), palette.TRAIN))
        test_line = always_redraw(lambda: layout.curve_until(axes, x, test, cursor.get_value(), palette.TEST))
        train_name = Text("training", font_size=SIZE, color=palette.TRAIN).move_to(axes.c2p(0.95, 0.78))
        test_name = Text("test", font_size=SIZE, color=palette.TEST).move_to(axes.c2p(2.55, 0.14))

        self.play(FadeIn(axes), FadeIn(numbers), FadeIn(x_name), FadeIn(y_name), run_time=1)
        self.add(train_line, test_line)
        self.play(cursor.animate.set_value(2.6), run_time=5, rate_func=linear)
        self.play(cursor.animate.set_value(x_mem), FadeIn(train_name), FadeIn(test_name), run_time=1.8,
                  rate_func=linear)

        # Memorisation: training accuracy first reaches 0.99.
        mem_point = axes.c2p(x_mem, np.interp(x_mem, x, train))
        mem_drop = DashedLine(mem_point, axes.c2p(x_mem, 0), color=palette.TRAIN, stroke_width=3)
        mem_marker = layout.event_marker(mem_point, palette.TRAIN, filled=False)
        mem_name = Text("memorised", font_size=SIZE, color=palette.INK).next_to(mem_point, UP, buff=0.3)
        self.play(Create(mem_drop), FadeIn(mem_marker), FadeIn(mem_name), run_time=1)
        self.wait(0.6)

        self.play(cursor.animate.set_value(x_grok), run_time=2.6, rate_func=linear)

        # Grokking: test accuracy first reaches 0.95, at the step T.
        grok_point = axes.c2p(x_grok, np.interp(x_grok, x, test))
        grok_drop = DashedLine(grok_point, axes.c2p(x_grok, 0), color=palette.TEST, stroke_width=3)
        grok_marker = layout.event_marker(grok_point, palette.TEST, filled=True)
        self.play(Create(grok_drop), FadeIn(grok_marker), run_time=1)

        self.play(cursor.animate.set_value(x[-1]), run_time=2, rate_func=linear)

        # The grokking step T is counted from the start of training.
        brace = BraceBetweenPoints(axes.c2p(0, 0), axes.c2p(x_grok, 0), direction=DOWN, color=palette.INK)
        brace.next_to(numbers, DOWN, buff=0.15).align_to(axes.c2p(0, 0), LEFT)
        brace_name = VGroup(Text("grokking step", font_size=SIZE, color=palette.INK),
                            MathTex(notation.GROK_TIME, font_size=SIZE + 8, color=palette.INK))
        brace_name.arrange(RIGHT, buff=0.2).next_to(brace, DOWN, buff=0.15)
        self.play(GrowFromCenter(brace), FadeIn(brace_name), run_time=1.2)
        self.add(layout.tag("one run, seed 0"))
        self.wait(3.5)
