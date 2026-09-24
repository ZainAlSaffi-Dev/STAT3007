"""One throwaway scene that checks the toolchain.

It draws a circle and one formula. The circle proves that Manim can write an
MP4. The formula proves that MathTex found latex and dvisvgm. Run it first on a
new machine:

    uv run manim -ql smoke.py SmokeTest
"""

import sys
from pathlib import Path

from manim import Circle, Create, MathTex, Scene, Write

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import notation, palette


class SmokeTest(Scene):
    def construct(self):
        circle = Circle(radius=1.5, color=palette.BLUE)
        formula = MathTex(notation.DECOMPOSITION, color=palette.INK).scale(0.8)
        formula.next_to(circle, direction=[0, -1, 0], buff=0.8)
        self.play(Create(circle))
        self.play(Write(formula))
        self.wait(0.5)
