"""Colours and sizes shared by every scene.

The hex values are those of the report figures, report_v2/code/common.py, and
of the deck theme, presentation/theme/g2.scss, so a curve keeps its colour when
it moves from a figure in the report to a clip in the deck. Each statistic has
one hue: scale is blue, rotation is orange and alignment is aqua. Accuracy is
grey for training and ink for test, so that it never reads as a statistic.

The background is white, set in manim.cfg. Manim draws in white by default,
which is invisible on a white page, so every mobject in a scene has to be
given a colour. Use INK for anything that would otherwise be black.
"""

# Categorical colours, in the order ntk_lib.py fixes them.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#52514e"

# Ordered ramp, for sweeps such as alpha or the weight-decay grid. It is blue,
# the colour of scale, so the deck clips do not use it.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

# Page colours.
PAPER = "#ffffff"
INK = "#0b0b0b"
GRID = "#e4e3df"

# One colour per kernel term, used in every scene that shows more than one.
SCALE = BLUE
ROTATION = ORANGE
ALIGNMENT = AQUA

# Darker steps of the same hues for symbols and words, as the deck colours
# \Sc, \Rc and \Ac, so that coloured text stays readable.
SCALE_TEXT = "#1c5cab"
ROTATION_TEXT = "#c9501f"
ALIGNMENT_TEXT = "#11704e"

# Heat maps of a signed quantity run from NEGATIVE through PAPER at zero to
# POSITIVE.
NEGATIVE = BLUE
POSITIVE = ORANGE

# The aim, the fourth quantity in the turn and aim scenes. ntk_lib has no
# colour for it, so this one is new.
AIM = "#7c3aed"

# One colour per accuracy or loss curve.
TRAIN = "#8a8986"
TEST = INK

# The test accuracy of the three values of alpha, light to dark, as in the
# report figure E1 (report_v2/code/E01_baseline.py).
ALPHA_GREYS = {0.5: "#b5b4b0", 1.0: "#6e6d69", 2.0: "#0b0b0b"}

# The font of every Text object. The deck sets Source Sans 3 with Helvetica Neue
# as its fallback. Pango finds only the second on our machines, and its own
# default is a serif font.
FONT = "Helvetica Neue"

# Type sizes at 1920 by 1080. A caption that needs more than one line at
# CAPTION size is too long for a slide.
TITLE = 48
CAPTION = 32
LABEL = 28

# Labels in the deck clips. A clip in a column of a slide is shown at about
# half size, so it needs LABEL_COLUMN. A clip across the whole slide uses
# LABEL_WIDE. The small grey tag such as "schematic" uses TAG.
LABEL_COLUMN = 56
LABEL_WIDE = 40
TAG = 36
