"""Scale against rotation along three runs, with lines of equal kernel movement.

Every path is a measurement from the dense runs that
repoduced-code/kernel_snapshots.py writes: alpha 1, width 100, NTK
parameterisation, seed 0, eta lambda 0, 0.0003 and 0.001, checkpoints every
250 steps up to 30,000. The axes are the recorded history keys R_c and S_c.

The grey lines are exact. They are the level sets of the movement statistic
D = ||K_t - K_0||_F / ||K_0||_F, which Equation eq:decomp in docs/report.tex
writes as a function of S and R alone. It holds for the centred kernel
with the centred terms, which is what the axes show:

    D^2 = e^{2S} + 1 - 2 e^{S} (1 - R).

Solving for e^S gives e^S = (1 - R) +/- sqrt((1 - R)^2 + D^2 - 1). For D of at
least 1 only the plus sign gives a positive root. For D below 1 both roots are
positive where the square root is real, so the level set has an upper and a
lower branch.

term_dependence.py measures the lockstep over the whole Tier 1 grid. The
straight-line R squared of S_t on R_t along a run has median 0.993 over the 20
runs without decay, and 0.045 in the one run at eta lambda 0.001.

Render from animations/:

    uv run manim -qh slides/scale_rotation_lockstep.py ScaleRotationLockstep
    uv run manim -qh -s slides/scale_rotation_lockstep.py ScaleRotationLockstep
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, UR, Axes, Dot, FadeIn, Integer, Line, MathTex, Scene, Transform,
                   ValueTracker, VGroup, VMobject, always_redraw, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, layout, notation, palette

# The three dense runs from common/data.py, coloured along the ordered ramp so
# that stronger decay is darker.
RUNS = [(name, decay, colour) for (name, decay), colour in
        zip(data.DENSE_RUNS, (palette.RAMP[0], palette.RAMP[2], palette.RAMP[4]))]
R_RANGE = (0, 0.08, 0.02)
S_RANGE = (-1, 2, 0.5)
LEVELS = [0.5, 1, 2, 4]


def level_set(D, R):
    """The branches of e^S on the level set D, as a list of arrays of S over R. NaN where undefined."""
    disc = (1 - R) ** 2 + D ** 2 - 1
    root = np.sqrt(np.where(disc >= 0, disc, np.nan))
    branches = [np.log((1 - R) + root)]
    if D < 1:
        branches.append(np.log((1 - R) - root))
    return branches


class ScaleRotationLockstep(Scene):
    def construct(self):
        runs = []
        for name, decay, colour in RUNS:
            h = data.load_run(name)["history"]
            runs.append(dict(steps=np.asarray(h["step"]), R=np.asarray(h["R_c"]), S=np.asarray(h["S_c"]),
                             decay=decay, colour=colour))
        steps = runs[0]["steps"]

        heading = layout.title("Scale against rotation")
        axes = Axes(x_range=R_RANGE, y_range=S_RANGE, x_length=7.5, y_length=5.2, tips=False,
                    axis_config=dict(color=palette.INK, stroke_width=2, font_size=24, include_numbers=True,
                                     decimal_number_config=dict(num_decimal_places=2, color=palette.INK)))
        axes.move_to(0.9 * LEFT + 0.35 * DOWN)
        x_name = MathTex("R_t", color=palette.ROTATION, font_size=palette.CAPTION).next_to(axes.x_axis, RIGHT, buff=0.2)
        y_name = MathTex("S_t", color=palette.SCALE, font_size=palette.CAPTION).next_to(axes.y_axis, UP, buff=0.2)

        # Exact level sets of D from Equation eq:decomp.
        grid_R = np.linspace(R_RANGE[0], R_RANGE[1], 200)
        levels = VGroup()
        level_names = VGroup()
        for D in LEVELS:
            for S in level_set(D, grid_R):
                keep = np.isfinite(S) & (S >= S_RANGE[0]) & (S <= S_RANGE[1])
                if keep.sum() < 2:
                    continue
                line = VMobject(color=palette.GRAY, stroke_width=1.5, stroke_opacity=0.6)
                line.set_points_smoothly([axes.c2p(r, s) for r, s in zip(grid_R[keep], S[keep])])
                levels.add(line)
                name = MathTex(f"D = {D:g}", color=palette.GRAY, font_size=22)
                name.next_to(line.get_end(), RIGHT, buff=0.1)
                level_names.add(name)

        # Each run has its own tracker, the checkpoint its path has reached, so one
        # path can stay drawn while the others are traced. The step counter follows clock.
        upto = [ValueTracker(0) for _ in runs]
        clock = ValueTracker(0)

        paths = VGroup()
        heads = VGroup()
        for run, tracker in zip(runs, upto):
            def draw(run=run, tracker=tracker):
                j = int(round(tracker.get_value()))
                points = [axes.c2p(r, s) for r, s in zip(run["R"][:j + 1], run["S"][:j + 1])]
                line = VMobject(color=run["colour"], stroke_width=4)
                line.set_points_as_corners(points if len(points) > 1 else points * 2)
                return line

            def head(run=run, tracker=tracker):
                j = int(round(tracker.get_value()))
                return Dot(axes.c2p(run["R"][j], run["S"][j]), radius=0.07, color=run["colour"])
            paths.add(always_redraw(draw))
            heads.add(always_redraw(head))

        legend = VGroup()
        for run in runs:
            swatch = Line(0.25 * LEFT, 0.25 * RIGHT, color=run["colour"], stroke_width=6)
            text = MathTex(notation.DECAY + " = " + run["decay"], color=palette.INK, font_size=palette.LABEL)
            legend.add(VGroup(swatch, text).arrange(RIGHT, buff=0.15))
        movement = MathTex(notation.MOVEMENT_CENTRED, color=palette.GRAY, font_size=palette.LABEL)
        legend.add(movement)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(axes, RIGHT, buff=0.9).shift(0.6 * UP)

        step_word = layout.label("step")
        step_word.to_corner(UR, buff=0.5).shift(1.9 * LEFT)
        step_value = Integer(0, color=palette.INK, font_size=palette.CAPTION)
        step_value.add_updater(lambda m: m.set_value(int(steps[int(round(clock.get_value()))])).next_to(step_word, RIGHT, buff=0.2))
        step_value.next_to(step_word, RIGHT, buff=0.2)

        self.play(FadeIn(heading), FadeIn(axes), FadeIn(x_name), FadeIn(y_name), FadeIn(step_word),
                  FadeIn(step_value))
        # First the run without decay on its own.
        self.play(FadeIn(legend[0]), FadeIn(paths[0]), FadeIn(heads[0]))
        self.play(upto[0].animate.set_value(len(steps) - 1), clock.animate.set_value(len(steps) - 1),
                  run_time=7, rate_func=linear)
        words = layout.caption("Without decay, scale and rotation rise together.")
        self.play(FadeIn(words))
        self.wait(2.5)

        # Then the two runs with decay, from step 0 again.
        clock.set_value(0)
        self.play(FadeIn(legend[1]), FadeIn(legend[2]), FadeIn(paths[1:]), FadeIn(heads[1:]))
        self.play(upto[1].animate.set_value(len(steps) - 1), upto[2].animate.set_value(len(steps) - 1),
                  clock.animate.set_value(len(steps) - 1), run_time=9, rate_func=linear)
        new_words = layout.caption("With decay the two move apart.")
        self.play(Transform(words, new_words))
        self.wait(2.5)

        self.play(FadeIn(levels), FadeIn(level_names), FadeIn(legend[3]))
        new_words = layout.caption("Each grey line has one value of the movement statistic.")
        self.play(Transform(words, new_words))
        self.wait(3)
