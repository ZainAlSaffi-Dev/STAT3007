"""Why alignment is set by rotation and aim, and why scale never enters it.

The first half builds the identity

    A_t = A_0 (1 - R_t) + gamma_t sqrt(1 - A_0^2) sqrt(R_t (2 - R_t)).

It is the group's own algebra, written out in the docstring of
repoduced-code/term_dependence.py. It is not from a paper. It holds for any
kernel, because it only splits the normalised centred kernel into its part
along the step 0 kernel and the rest.

The second half draws three unit vectors: the step 0 kernel, the target and
the kernel at step t. Their three pairwise cosines are 1 - R_t, A_t and A_0,
and three unit vectors with known cosines can be placed exactly in three
dimensions. The first axis is the step 0 kernel. The second is the direction
in which the target leaves it. The third holds whatever else the kernel turned
into. That third direction is a different direction at each step, so the
picture is exact at every frame, but the path the arrow tip traces is not a
measurement of how far apart two checkpoints are. The scene draws no trail
for that reason.

The measured run is dense_N100_a1_wd0.0003_s0 from
repoduced-code/kernel_snapshots.py: alpha 1, width 100, NTK parameterisation,
seed 0, eta lambda 0.0003, checkpoints every 250 steps up to 30,000.

Render from animations/:

    uv run manim -qh explainer/turn_and_aim.py TurnAndAim
    uv run manim -qh -s explainer/turn_and_aim.py TurnAndAim
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DEGREES, DOWN, LEFT, ORIGIN, OUT, RIGHT, UP, UL, UR, Arrow3D, Circle, Create,
                   DashedLine, DecimalNumber, FadeIn, FadeOut, Integer, Line, MathTex, SurroundingRectangle,
                   ThreeDScene, Transform, ValueTracker, VGroup, VMobject, Write, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data, kernels, layout, notation, palette

RUN = "dense_N100_a1_wd0.0003_s0"
SWEEP_SECONDS = 14
ARROW = dict(thickness=0.012, height=0.18, base_radius=0.05)


class TurnAndAim(ThreeDScene):
    def construct(self):
        self.algebra()
        self.geometry()

    def algebra(self):
        heading = layout.title("Alignment from rotation and aim")
        split = MathTex(notation.SPLIT, color=palette.INK, font_size=40).shift(1.6 * UP)
        definition = MathTex(notation.AIM_DEFINITION, color=palette.INK, font_size=40).shift(0.2 * UP)
        identity = MathTex(notation.ALIGNMENT_IDENTITY, color=palette.INK, font_size=44).shift(1.3 * DOWN)
        box = SurroundingRectangle(identity, color=palette.ALIGNMENT, buff=0.2)

        self.add_fixed_in_frame_mobjects(heading)
        self.play(FadeIn(heading))

        self.add_fixed_in_frame_mobjects(split)
        self.play(Write(split))
        words = layout.caption("Split the kernel into its start direction and a new part.")
        self.add_fixed_in_frame_mobjects(words)
        self.play(FadeIn(words))
        self.wait(3)

        self.add_fixed_in_frame_mobjects(definition)
        self.play(Write(definition))
        new_words = layout.caption("The aim says how directly the new part points at the target.")
        self.add_fixed_in_frame_mobjects(new_words)
        self.play(FadeOut(words), FadeIn(new_words))
        words = new_words
        self.wait(3)

        self.add_fixed_in_frame_mobjects(identity, box)
        self.play(Write(identity), Create(box))
        new_words = layout.caption("Alignment follows from rotation and aim, and scale never enters.")
        self.add_fixed_in_frame_mobjects(new_words)
        self.play(FadeOut(words), FadeIn(new_words))
        self.wait(4)

        self.play(*[FadeOut(m) for m in [heading, split, definition, identity, box, new_words]])

    def geometry(self):
        run = data.load_run(RUN)
        h = run["history"]
        steps = np.asarray(h["step"])
        R = np.asarray(h["R_c"])
        A = np.asarray(h["A_t"])
        A_0 = A[0]
        gamma = kernels.aim(R, A, A_0)
        # The aim is undefined at step 0, where the kernel has not turned, so the
        # picture starts at the first checkpoint, step 250.
        first = 1
        norm_ratio = np.exp(np.asarray(h["S_c"]))

        def tip(i):
            # Exact coordinates of the unit kernel at checkpoint i. See the module docstring.
            r = R[i]
            s = np.sqrt(r * (2 - r))
            return np.array([1 - r, gamma[i] * s, np.sqrt(1 - gamma[i] ** 2) * s])

        target_tip = np.array([A_0, np.sqrt(1 - A_0 ** 2), 0.0])
        scale = 3.0  # scene units per unit length

        self.set_camera_orientation(phi=68 * DEGREES, theta=-35 * DEGREES, zoom=1.0)

        index = ValueTracker(first)

        def i_now():
            return int(round(index.get_value()))

        # Faint guides: the unit circle in the flat plane of start and target, and the
        # quarter circle from the start towards the third axis.
        flat = Circle(radius=scale, color=palette.GRID, stroke_width=2)
        up_arc = Circle(radius=scale, color=palette.GRID, stroke_width=2).rotate(90 * DEGREES, RIGHT)
        axis_x = Line(ORIGIN, 1.15 * scale * RIGHT, color=palette.GRID, stroke_width=2)
        axis_y = Line(ORIGIN, 1.15 * scale * UP, color=palette.GRID, stroke_width=2)
        axis_z = Line(ORIGIN, 1.15 * scale * OUT, color=palette.GRID, stroke_width=2)

        start = Arrow3D(ORIGIN, scale * RIGHT, color=palette.GRAY, **ARROW)
        target = Arrow3D(ORIGIN, scale * target_tip, color=palette.ALIGNMENT, **ARROW)
        now = Arrow3D(ORIGIN, scale * tip(first), color=palette.ROTATION, **ARROW)
        now.add_updater(lambda m: m.become(Arrow3D(ORIGIN, scale * tip(i_now()), color=palette.ROTATION,
                                                    **ARROW)))

        # Only the shadow of the kernel in the flat plane counts towards A_t,
        # because the target has no component along the third axis.
        def drop():
            t = scale * tip(i_now())
            if t[2] < 0.05:
                # Too short to draw as a dashed line, which needs a positive length.
                return VMobject()
            return DashedLine(t, np.array([t[0], t[1], 0.0]), color=palette.ROTATION, stroke_width=2,
                              dash_length=0.08)
        shadow = drop()
        shadow.add_updater(lambda m: m.become(drop()))

        start_name = MathTex(notation.K_UNIT + "_0", color=palette.GRAY, font_size=36)
        start_name.move_to(1.22 * scale * RIGHT + 0.35 * OUT)
        target_name = MathTex(notation.TARGET_UNIT, color=palette.ALIGNMENT, font_size=36)
        target_name.move_to(1.15 * scale * target_tip + 0.35 * OUT)
        now_name = MathTex(notation.K_UNIT + "_t", color=palette.ROTATION, font_size=36)
        now_name.add_updater(lambda m: m.move_to(scale * tip(i_now()) * 1.12 + 0.3 * OUT))
        self.add_fixed_orientation_mobjects(start_name, target_name, now_name)

        heading = layout.title("The kernel as a direction")
        step_word = layout.label("step")
        step_word.to_corner(UR, buff=0.5).shift(1.9 * LEFT)
        step_value = Integer(int(steps[first]), color=palette.INK, font_size=palette.CAPTION)
        step_value.add_updater(lambda m: m.set_value(int(steps[i_now()])).next_to(step_word, RIGHT, buff=0.2))

        # The four numbers at the current checkpoint of RUN.
        def row(symbol, values, colour, places=3):
            name = MathTex(symbol + " =", color=colour, font_size=palette.CAPTION)
            value = DecimalNumber(values[first], num_decimal_places=places, color=colour, font_size=palette.CAPTION)
            value.add_updater(lambda m: m.set_value(values[i_now()]).next_to(name, RIGHT, buff=0.15))
            value.next_to(name, RIGHT, buff=0.15)
            return VGroup(name, value)

        numbers = VGroup(
            row("R_t", R, palette.ROTATION),
            row(notation.AIM, gamma, palette.AIM),
            row("A_t", A, palette.ALIGNMENT),
            row(notation.NORM_RATIO, norm_ratio, palette.SCALE, places=2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        numbers.to_corner(UL, buff=0.5).shift(1.3 * DOWN)
        run_name = MathTex(notation.DECAY + " = 0.0003", color=palette.INK, font_size=palette.LABEL)
        run_name.next_to(numbers, DOWN, aligned_edge=LEFT, buff=0.4)

        self.add_fixed_in_frame_mobjects(heading, step_word, step_value, numbers, run_name)
        self.play(FadeIn(heading), FadeIn(step_word), FadeIn(step_value), FadeIn(numbers), FadeIn(run_name))
        self.play(Create(flat), Create(up_arc), Create(axis_x), Create(axis_y), Create(axis_z))
        self.play(FadeIn(start), FadeIn(start_name), FadeIn(target), FadeIn(target_name))
        words = layout.caption("The angles to the start and to the target are measured.")
        self.add_fixed_in_frame_mobjects(words)
        self.play(FadeIn(words))
        self.wait(2.5)

        self.add(now, shadow, now_name)
        new_words = layout.caption("The upward axis holds every other direction the kernel turns into.")
        self.add_fixed_in_frame_mobjects(new_words)
        self.play(FadeOut(words), FadeIn(new_words))
        words = new_words
        self.begin_ambient_camera_rotation(rate=0.04)
        self.play(index.animate.set_value(len(steps) - 1), run_time=SWEEP_SECONDS, rate_func=linear)
        self.stop_ambient_camera_rotation()

        new_words = layout.caption("Most of the turn goes upwards and does not raise the alignment.")
        self.add_fixed_in_frame_mobjects(new_words)
        self.play(FadeOut(words), FadeIn(new_words))
        self.wait(3)
