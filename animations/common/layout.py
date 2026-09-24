"""Titles, captions and heat maps drawn the same way in every scene.

A caption is one short sentence on one line. caption() raises when the text
would not fit on one line of the frame, so a long caption fails the render
instead of reaching the deck.
"""

import numpy as np
from manim import DOWN, LEFT, UP, RESAMPLING_ALGORITHMS, ImageMobject, Text, config
from manim.utils.images import change_to_rgba_array

from . import palette

MARGIN = 0.5


def _check_width(mob, what, text):
    room = config.frame_width - 2 * MARGIN
    if mob.width > room:
        raise ValueError(f"The {what} '{text}' is {mob.width:.2f} units wide and the frame allows {room:.2f}.")
    return mob


def title(text):
    """A title in the top left corner."""
    mob = Text(text, font_size=palette.TITLE, color=palette.INK)
    _check_width(mob, "title", text)
    return mob.to_corner(UP + LEFT, buff=MARGIN)


def caption(text):
    """A one-line caption at the bottom of the frame."""
    mob = Text(text, font_size=palette.CAPTION, color=palette.INK)
    _check_width(mob, "caption", text)
    return mob.to_edge(DOWN, buff=MARGIN)


def label(text, color=palette.INK):
    """A short plain-text label."""
    return Text(text, font_size=palette.LABEL, color=color)


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def diverging_pixels(values, vmax):
    """Map a 2D array to RGB pixels, NEGATIVE below zero, PAPER at zero, POSITIVE above.

    Values beyond plus or minus vmax are drawn at full colour.
    """
    x = np.clip(np.asarray(values, float) / vmax, -1.0, 1.0)[..., None]
    paper, neg, pos = _rgb(palette.PAPER), _rgb(palette.NEGATIVE), _rgb(palette.POSITIVE)
    rgb = np.where(x >= 0, paper + x * (pos - paper), paper - x * (neg - paper))
    return rgb.round().astype(np.uint8)


def heat_map(values, vmax, side):
    """A square image of a 2D array with one sharp block per entry."""
    image = ImageMobject(diverging_pixels(values, vmax))
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    image.height = side
    return image


def set_pixels(image, values, vmax):
    """Redraw a heat map in place, for use inside an updater."""
    image.pixel_array = change_to_rgba_array(diverging_pixels(values, vmax))
