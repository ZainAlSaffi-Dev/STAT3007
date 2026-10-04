"""Train and test accuracy of one saved run. Copy this file to start a new scene.

This is a working scene and a template at once. It loads the Tier 0 run
mean_field_alpha1 (alpha 1, width 100, mean-field parameterisation, seed 0)
and draws its train and test accuracy against the step. Every curve is a
measurement read from repoduced-code/results/mean_field_alpha1.json.

To start a new scene, copy this file into slides/ or explainer/ and give it a
name that says what it shows. Then change four things.

1. This docstring. Say what the scene shows, which runs it loads, and the
   render commands with the new file and class names.
2. RUN, or a list of runs. data.available() lists every saved run, and
   data.DENSE_RUNS names the three runs that save the kernel every 250 steps.
3. The class name. Name it for the idea, such as ScaleVersusRotation.
4. The body of construct().

Then add a row for the scene to the table in README.md, and work through the
checklist at the end of CLAUDE.md in this folder.

Render from animations/:

    uv run manim -ql scene_template.py AccuracyOverTraining      # while working
    uv run manim -qh scene_template.py AccuracyOverTraining      # for the deck
    uv run manim -qh -s scene_template.py AccuracyOverTraining   # last frame as a still
"""

import sys
from pathlib import Path

from manim import DOWN, LEFT, RIGHT, UP, Axes, Create, FadeIn, Line, Scene, VGroup, linear

# Manim loads this file on its own, so the folder that holds common/ has to be
# put on the path by hand. A scene in slides/ or explainer/ needs parents[1],
# because common/ is one folder up. This file sits next to common/, so it
# needs parent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data, layout, palette

RUN = "mean_field_alpha1"
# Both accuracies reach 0.99 by step 3,750 in this run, so the axis stops at
# step 10,000 to leave the rise visible.
LAST_STEP = 10_000


class AccuracyOverTraining(Scene):
    def construct(self):
        # Load the measurement. history() raises with the list of keys if one is missing.
        run = data.load_run(RUN)
        steps, train, test = data.history(run, "step", "train_acc", "test_acc")
        keep = steps <= LAST_STEP
        steps, train, test = steps[keep], train[keep], test[keep]

        heading = layout.title("Train and test accuracy")
        number_style = dict(color=palette.INK, num_decimal_places=0)
        axes = Axes(x_range=(0, LAST_STEP, 2_500), y_range=(0, 1, 0.25), x_length=9, y_length=4.8, tips=False,
                    axis_config=dict(color=palette.INK, stroke_width=2, font_size=24),
                    x_axis_config=dict(numbers_to_include=[5_000, 10_000],
                                       decimal_number_config=dict(number_style, group_with_commas=True)),
                    y_axis_config=dict(numbers_to_include=[0.5, 1],
                                       decimal_number_config=dict(number_style, num_decimal_places=1)))
        axes.shift(0.2 * UP)
        x_name = layout.label("step").next_to(axes.x_axis, RIGHT, buff=0.3)

        # Every mobject needs a colour from the palette, because the page is white.
        curves = VGroup()
        for values, colour in ((train, palette.TRAIN), (test, palette.TEST)):
            curve = axes.plot_line_graph(steps, values, add_vertex_dots=False, line_color=colour, stroke_width=4)
            curves.add(curve)

        legend = VGroup()
        for word, colour in (("train", palette.TRAIN), ("test", palette.TEST)):
            swatch = Line(0.25 * LEFT, 0.25 * RIGHT, color=colour, stroke_width=6)
            legend.add(VGroup(swatch, layout.label(word)).arrange(RIGHT, buff=0.15))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(axes, RIGHT, buff=0.3).align_to(axes, UP)
        legend.shift(0.2 * LEFT)

        self.play(FadeIn(heading), FadeIn(axes), FadeIn(x_name), FadeIn(legend))
        self.play(Create(curves[0]), Create(curves[1]), run_time=4, rate_func=linear)
        # A caption is one short sentence. layout.caption() raises if it does not fit on one line.
        self.play(FadeIn(layout.caption(f"The saved run {RUN}.")))
        self.wait(1)
