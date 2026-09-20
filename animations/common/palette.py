"""Colours and sizes shared by every scene.

The hex values are the ones in the notebook helper ntk_lib.py on the sam
branch, so a curve keeps its colour when it moves from a figure in the report
to a clip in the deck. Train is blue and test is orange everywhere.

The background is white, set in manim.cfg. Manim draws in white by default,
which is invisible on a white page, so every mobject in a scene has to be
given a colour. Use INK for anything that would otherwise be black.
"""

# Categorical colours, in the order ntk_lib.py fixes them.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#52514e"

# Ordered ramp, for sweeps such as alpha or the weight-decay grid.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

# Page colours.
PAPER = "#ffffff"
INK = "#1a1a1a"
GRID = "#e6e5e1"

# One colour per kernel term, used in every scene that shows more than one.
SCALE = BLUE
ROTATION = ORANGE
ALIGNMENT = AQUA

# One colour per loss curve.
TRAIN = BLUE
TEST = ORANGE

# Type sizes at 1920 by 1080. A caption that needs more than one line at
# CAPTION size is too long for a slide.
TITLE = 48
CAPTION = 32
LABEL = 28
