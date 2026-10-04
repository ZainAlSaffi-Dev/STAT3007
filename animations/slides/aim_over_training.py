"""Rotation, aim and alignment over the first 30,000 steps of three cells.

Every curve is a measurement from the dense runs that
repoduced-code/kernel_snapshots.py writes: alpha 1, width 100, NTK
parameterisation, seed 0, eta lambda 0, 0.0003 and 0.001, checkpoints every
250 steps. The rotation and alignment are the recorded history keys R_c and
A_t. The aim is computed from them with the identity in
repoduced-code/term_dependence.py, and it is undefined at step 0, so its
curves start at step 250.

Render from animations/:

    uv run manim -qh slides/aim_over_training.py AimOverTraining
    uv run manim -qh -s slides/aim_over_training.py AimOverTraining
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, UR, Axes, FadeIn, Integer, Line, MathTex, Scene,
                   Transform, ValueTracker, VGroup, VMobject, always_redraw, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, kernels, layout, notation, palette

# The three dense runs from common/data.py, coloured along the ordered ramp so
# that stronger decay is darker.
RUNS = [(name, decay, colour) for (name, decay), colour in
        zip(data.DENSE_RUNS, (palette.RAMP[0], palette.RAMP[2], palette.RAMP[4]))]
# The pause shows the early turn of the strongest decay. At step 2,000 its
# rotation is 0.046 against 0.028 without decay, and its aim is 0.013 against
# 0.080 (dense runs, history keys R_c and A_t).
PAUSE_STEP = 2000


class AimOverTraining(Scene):
    def construct(self):
        runs = []
        for name, decay, colour in RUNS:
            h = data.load_run(name)["history"]
            R, A = np.asarray(h["R_c"]), np.asarray(h["A_t"])
            runs.append(dict(steps=np.asarray(h["step"]), R=R, A=A, gamma=kernels.aim(R, A, A[0]),
                             decay=decay, colour=colour))
        steps = runs[0]["steps"]
        last = int(steps[-1])

        heading = layout.title("Rotation, aim and alignment in three cells")
        panels = [
            ("R_t", "R", palette.ROTATION, (0, 0.08, 0.02)),
            (notation.AIM, "gamma", palette.AIM, (0, 0.2, 0.05)),
            ("A_t", "A", palette.ALIGNMENT, (0.08, 0.16, 0.02)),
        ]
        index = ValueTracker(1)

        def i_now():
            return int(round(index.get_value()))

        axes_group = VGroup()
        curves = VGroup()
        for k, (symbol, key, colour, y_range) in enumerate(panels):
            axes = Axes(x_range=(0, last, 10000), y_range=y_range, x_length=3.9, y_length=3.6,
                        tips=False, axis_config=dict(color=palette.INK, stroke_width=2, font_size=22,
                                                     include_numbers=True),
                        x_axis_config=dict(numbers_to_include=[0, 10000, 20000, 30000],
                                           decimal_number_config=dict(group_with_commas=True,
                                                                      num_decimal_places=0, color=palette.INK)),
                        y_axis_config=dict(decimal_number_config=dict(num_decimal_places=2, color=palette.INK)))
            axes.move_to((k - 1) * 4.6 * RIGHT + 0.1 * DOWN)
            name = MathTex(symbol, color=colour, font_size=palette.CAPTION).next_to(axes, UP, buff=0.2)
            step_name = layout.label("step").scale(0.8).next_to(axes, DOWN, buff=0.15)
            axes_group.add(VGroup(axes, name, step_name))
            for run in runs:
                values = run[key]

                def draw(axes=axes, values=values, run=run):
                    # The aim is undefined at step 0, so every curve starts at checkpoint 1.
                    j = max(i_now(), 1)
                    xs, ys = run["steps"][1:j + 1], values[1:j + 1]
                    points = [axes.c2p(x, y) for x, y in zip(xs, ys)]
                    line = VMobject(color=run["colour"], stroke_width=3.5)
                    line.set_points_as_corners(points if len(points) > 1 else points * 2)
                    return line
                curves.add(always_redraw(draw))

        legend = VGroup()
        for run in runs:
            swatch = Line(0.25 * LEFT, 0.25 * RIGHT, color=run["colour"], stroke_width=6)
            text = MathTex(notation.DECAY + " = " + run["decay"], color=palette.INK, font_size=palette.LABEL)
            legend.add(VGroup(swatch, text).arrange(RIGHT, buff=0.15))
        legend.arrange(RIGHT, buff=0.6).next_to(heading, DOWN, aligned_edge=LEFT, buff=0.3)

        step_word = layout.label("step")
        step_word.to_corner(UR, buff=0.5).shift(1.9 * LEFT)
        # The curves start at checkpoint 1, so the counter does too.
        step_value = Integer(int(steps[1]), color=palette.INK, font_size=palette.CAPTION)
        step_value.add_updater(lambda m: m.set_value(int(steps[i_now()])).next_to(step_word, RIGHT, buff=0.2))
        step_value.next_to(step_word, RIGHT, buff=0.2)

        self.play(FadeIn(heading), FadeIn(axes_group), FadeIn(legend), FadeIn(step_word), FadeIn(step_value))
        self.add(curves)

        pause = int(np.searchsorted(steps, PAUSE_STEP))
        self.play(index.animate.set_value(pause), run_time=3, rate_func=linear)
        words = layout.caption("The strongest decay turns the kernel early with almost no aim.")
        self.play(FadeIn(words))
        self.wait(3)
        self.play(index.animate.set_value(len(steps) - 1), run_time=12, rate_func=linear)
        new_words = layout.caption("The three cells turn about as far and differ in aim.")
        self.play(Transform(words, new_words))
        self.wait(3)
